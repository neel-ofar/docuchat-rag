from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from rag_engine import RAGEngine
import uuid
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Enable credentials for session support
CORS(app, supports_credentials=True)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize RAG engine
print("🚀 Starting DocuChat RAG Application...")
rag_engine = RAGEngine()

# Store conversation history in memory (in production, use a database)
conversations = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_document():
    """Upload and process documents for RAG"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            original_filename = secure_filename(file.filename)

            # Prevent overwriting files with unique names
            unique_filename = f"{uuid.uuid4()}_{original_filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

            file.save(filepath)
            
            # Process document with RAG engine
            success = rag_engine.add_document(filepath, original_filename)
            
            if success:
                # Initialize conversation for this session
                if 'session_id' not in session:
                    session['session_id'] = str(uuid.uuid4())
                
                if session['session_id'] not in conversations:
                    conversations[session['session_id']] = []
                
                return jsonify({
                    'success': True,
                    'message': f'Document "{original_filename}" uploaded and processed successfully',
                    'filename': original_filename,
                    'session_id': session['session_id']
                })
            else:
                return jsonify({'error': 'Failed to process document'}), 500
        
        return jsonify({'error': 'Invalid file type. Please upload TXT, PDF, or DOCX files.'}), 400
    
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/query', methods=['POST'])
def query():
    """Query the RAG system"""
    try:
        # Safe JSON parsing
        data = request.get_json(silent=True) or {}
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        # Get or create session
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        session_id = session['session_id']
        
        # Get conversation history
        history = conversations.get(session_id, [])
        
        # Query RAG engine
        answer, sources = rag_engine.query(question, history)
        
        # Store in conversation history
        conversation_entry = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer,
            'sources': sources
        }
        
        if session_id not in conversations:
            conversations[session_id] = []
        
        conversations[session_id].append(conversation_entry)
        
        return jsonify({
            'answer': answer,
            'sources': sources,
            'session_id': session_id
        })
    
    except Exception as e:
        print(f"❌ Query error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get conversation history"""
    try:
        session_id = session.get('session_id')
        if not session_id or session_id not in conversations:
            return jsonify({'history': []})
        
        return jsonify({'history': conversations[session_id]})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def clear_session():
    """Clear current session"""
    try:
        session_id = session.get('session_id')
        if session_id and session_id in conversations:
            del conversations[session_id]
        
        session.pop('session_id', None)
        rag_engine.clear_documents()
        
        return jsonify({'success': True, 'message': 'Session cleared successfully'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    """Get list of uploaded documents"""
    try:
        documents = rag_engine.get_document_list()
        return jsonify({'documents': documents})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 DocuChat RAG Application Started!")
    print("="*50)
    print("📚 Upload documents and ask questions!")
    
    # Get port from environment (for cloud deployment)
    port = int(os.environ.get('PORT', 5000))
    
    # Use debug mode only in development
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    print(f"🔗 Access at: http://localhost:{port}")
    print("="*50 + "\n")
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
