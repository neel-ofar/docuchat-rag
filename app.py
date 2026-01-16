"""
Google Docs AI Chatbot - Flask Application
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import os
from dotenv import load_dotenv
import uuid
from gdocs_fetcher import GoogleDocsFetcher
from rag_engine import RAGEngine

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', os.urandom(24))
CORS(app, supports_credentials=True)

# Initialize components
print("=" * 60)
print("🚀 Google Docs AI Chatbot Starting...")
print("=" * 60)

gdocs_fetcher = GoogleDocsFetcher()
rag_engine = RAGEngine()

print("=" * 60)
print("✅ Application Ready!")
print("=" * 60 + "\n")


@app.route('/')
def index():
    """Serve main page"""
    return render_template('index.html')


@app.route('/api/load-document', methods=['POST'])
def load_document():
    """Load Google Doc and ingest into RAG"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'success': False, 'error': 'URL is required'}), 400
        
        # Validate URL
        validation = gdocs_fetcher.validate_url(url)
        if not validation['valid']:
            return jsonify({'success': False, 'error': validation['error']}), 400
        
        # Fetch document
        print(f"\n📥 Fetching document from: {url}")
        result = gdocs_fetcher.fetch_document(url)
        
        if not result['success']:
            return jsonify({'success': False, 'error': result['error']}), 400
        
        # Ingest into RAG
        ingest_result = rag_engine.ingest_document(
            content=result['content'],
            title=result['title'],
            doc_id=result['doc_id']
        )
        
        if not ingest_result['success']:
            return jsonify({'success': False, 'error': ingest_result['error']}), 500
        
        # Initialize session
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        return jsonify({
            'success': True,
            'title': result['title'],
            'chunks': ingest_result['chunks'],
            'message': f'Document "{result["title"]}" loaded successfully with {ingest_result["chunks"]} chunks.'
        })
    
    except Exception as e:
        print(f"❌ Load error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/query', methods=['POST'])
def query():
    """Query the chatbot"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'success': False, 'error': 'Question is required'}), 400
        
        # Get or create session
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        # Query RAG engine
        result = rag_engine.query(question)
        
        if result.get('error'):
            return jsonify({'success': False, 'error': result['answer']}), 400
        
        return jsonify({
            'success': True,
            'answer': result['answer'],
            'sources': result['sources'],
            'fallback': result.get('fallback', False)
        })
    
    except Exception as e:
        print(f"❌ Query error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/clear', methods=['POST'])
def clear():
    """Clear current document and conversation"""
    try:
        rag_engine.clear()
        session.pop('session_id', None)
        
        return jsonify({
            'success': True,
            'message': 'Document and conversation cleared'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def status():
    """Get current status"""
    try:
        has_document = rag_engine.current_doc is not None
        
        status_info = {
            'has_document': has_document,
            'history_length': len(rag_engine.conversation_history)
        }
        
        if has_document:
            status_info['document'] = {
                'title': rag_engine.current_doc['title'],
                'chunks': rag_engine.current_doc['total_chunks']
            }
        
        return jsonify(status_info)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'google-docs-chatbot'})


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    print(f"\n🌐 Server running on: http://localhost:{port}")
    print("📝 Ready to load Google Docs!\n")
    
    app.run(debug=debug, host='0.0.0.0', port=port)
