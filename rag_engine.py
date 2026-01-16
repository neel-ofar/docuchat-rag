"""
RAG Engine for Google Docs Chatbot
Implements chunking, embedding, retrieval, and generation
"""

import os
import re
from typing import List, Dict, Tuple, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class RAGEngine:
    """Complete RAG pipeline for document Q&A"""
    
    def __init__(self):
        print("🔧 Initializing RAG Engine...")
        
        # Configuration
        self.chunk_size = int(os.getenv('CHUNK_SIZE', 800))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 100))
        self.top_k = int(os.getenv('TOP_K_RESULTS', 3))
        
        # Initialize embeddings
        print("📥 Loading embedding model...")
        embedding_model = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        self.embedder = SentenceTransformer(embedding_model)
        print("✅ Embeddings ready!")
        
        # Initialize ChromaDB
        print("🗄️  Setting up vector store...")
        self.chroma_client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))
        self.collection = None
        print("✅ Vector store ready!")
        
        # Initialize LLM (Groq)
        groq_key = os.getenv('GROQ_API_KEY')
        if groq_key:
            self.llm_client = Groq(api_key=groq_key)
            self.llm_model = os.getenv('LLM_MODEL', 'llama-3.1-8b-instant')
            print("✅ LLM ready (Groq)!")
        else:
            self.llm_client = None
            print("⚠️  No LLM API key found")
        
        # Storage
        self.current_doc = None
        self.conversation_history = []
        
        print("✅ RAG Engine initialized!\n")
    
    def chunk_text(self, text: str) -> List[Dict]:
        """
        Chunk text into semantic segments with metadata
        Returns list of {text, chunk_id, section}
        """
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        # Detect sections (lines starting with numbers or all caps)
        section_pattern = re.compile(r'^(\d+\.|\d+\)|\w+:|\b[A-Z\s]{5,}\b)')
        current_section = "Introduction"
        
        for para in paragraphs:
            # Check if this is a section header
            if section_pattern.match(para) and len(para) < 100:
                current_section = para
                continue
            
            # Add to current chunk
            if len(current_chunk) + len(para) < self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                # Save current chunk
                if current_chunk.strip():
                    chunks.append({
                        'text': current_chunk.strip(),
                        'chunk_id': chunk_id,
                        'section': current_section
                    })
                    chunk_id += 1
                
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_words = words[-self.chunk_overlap:] if len(words) > self.chunk_overlap else words
                current_chunk = ' '.join(overlap_words) + "\n\n" + para + "\n\n"
        
        # Add last chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'chunk_id': chunk_id,
                'section': current_section
            })
        
        return chunks
    
    def ingest_document(self, content: str, title: str, doc_id: str) -> Dict:
        """Ingest document into vector store"""
        try:
            print(f"📄 Ingesting document: {title}")
            
            # Chunk document
            chunks = self.chunk_text(content)
            print(f"✂️  Created {len(chunks)} chunks")
            
            if len(chunks) == 0:
                return {'success': False, 'error': 'No chunks created from document'}
            
            # Reset collection
            try:
                self.chroma_client.delete_collection('documents')
            except:
                pass
            
            self.collection = self.chroma_client.create_collection(
                name='documents',
                metadata={'doc_id': doc_id, 'title': title}
            )
            
            # Generate embeddings and store
            print("🔄 Creating embeddings...")
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()
            
            # Add to ChromaDB
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                ids=[f"chunk_{i}" for i in range(len(chunks))],
                metadatas=[{
                    'chunk_id': chunk['chunk_id'],
                    'section': chunk['section']
                } for chunk in chunks]
            )
            
            # Store document info
            self.current_doc = {
                'title': title,
                'doc_id': doc_id,
                'content': content,
                'chunks': chunks,
                'total_chunks': len(chunks)
            }
            
            # Clear conversation history
            self.conversation_history = []
            
            print(f"✅ Document ingested successfully!\n")
            
            return {
                'success': True,
                'chunks': len(chunks),
                'title': title
            }
        
        except Exception as e:
            print(f"❌ Ingestion error: {e}")
            return {'success': False, 'error': str(e)}
    
    def retrieve_context(self, query: str) -> List[Dict]:
        """Retrieve relevant chunks for query"""
        if not self.collection:
            return []
        
        # Generate query embedding
        query_embedding = self.embedder.encode([query], show_progress_bar=False).tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=self.top_k
        )
        
        # Format results
        contexts = []
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            contexts.append({
                'text': doc,
                'section': metadata.get('section', 'Unknown'),
                'chunk_id': metadata.get('chunk_id', i),
                'distance': results['distances'][0][i] if 'distances' in results else 0
            })
        
        return contexts
    
    def generate_answer(self, query: str, contexts: List[Dict]) -> Dict:
        """Generate answer using LLM with citations"""
        
        if not contexts:
            return {
                'answer': "I couldn't find relevant information in the document to answer your question. Could you rephrase or ask about a different topic?",
                'sources': [],
                'fallback': True
            }
        
        # Build context string with section markers
        context_text = ""
        for ctx in contexts:
            context_text += f"[Section: {ctx['section']}]\n{ctx['text']}\n\n"
        
        # Build conversation history
        history_text = ""
        if self.conversation_history:
            recent_history = self.conversation_history[-5:]  # Last 5 exchanges
            for entry in recent_history:
                history_text += f"User: {entry['question']}\nAssistant: {entry['answer']}\n\n"
        
        # Create prompt
        prompt = self._create_prompt(query, context_text, history_text)
        
        # Generate with LLM
        if self.llm_client:
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided document context. Always cite the section name when referencing information."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                answer = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"⚠️  LLM error: {e}")
                answer = self._extractive_answer(query, contexts)
        else:
            answer = self._extractive_answer(query, contexts)
        
        return {
            'answer': answer,
            'sources': contexts,
            'fallback': False
        }
    
    def _create_prompt(self, query: str, context: str, history: str) -> str:
        """Create prompt for LLM"""
        prompt = f"""Based on the following document context, answer the user's question accurately and concisely.

IMPORTANT RULES:
1. Only use information from the provided context
2. Always cite the section name when referencing information (e.g., "According to Section X...")
3. If the answer isn't in the context, say "This information is not available in the document"
4. Be direct and concise (2-3 sentences max)
5. Use previous conversation for context if relevant

Document Context:
{context[:2000]}

"""
        
        if history:
            prompt += f"Previous Conversation:\n{history}\n\n"
        
        prompt += f"User Question: {query}\n\nAnswer:"
        
        return prompt
    
    def _extractive_answer(self, query: str, contexts: List[Dict]) -> str:
        """Fallback extractive answer"""
        # Find most relevant sentences
        query_words = set(query.lower().split())
        
        best_sentences = []
        for ctx in contexts:
            sentences = ctx['text'].split('. ')
            for sent in sentences:
                sent_words = set(sent.lower().split())
                overlap = len(query_words & sent_words)
                if overlap > 2:
                    best_sentences.append((overlap, sent, ctx['section']))
        
        best_sentences.sort(reverse=True)
        
        if best_sentences:
            answer = f"According to {best_sentences[0][2]}: {best_sentences[0][1]}"
            if len(best_sentences) > 1:
                answer += f". Also, from {best_sentences[1][2]}: {best_sentences[1][1]}"
            return answer
        
        return "I found relevant sections but couldn't extract a specific answer. Could you rephrase your question?"
    
    def query(self, question: str) -> Dict:
        """Main query method"""
        try:
            if not self.current_doc:
                return {
                    'answer': 'Please upload a Google Doc first.',
                    'sources': [],
                    'error': True
                }
            
            print(f"\n🔍 Query: {question}")
            
            # Retrieve relevant contexts
            contexts = self.retrieve_context(question)
            print(f"📚 Retrieved {len(contexts)} relevant chunks")
            
            # Generate answer
            result = self.generate_answer(question, contexts)
            
            # Store in history
            self.conversation_history.append({
                'question': question,
                'answer': result['answer'],
                'sources': [ctx['section'] for ctx in contexts]
            })
            
            print(f"✅ Answer generated\n")
            
            return result
        
        except Exception as e:
            print(f"❌ Query error: {e}")
            return {
                'answer': f'Error processing query: {str(e)}',
                'sources': [],
                'error': True
            }
    
    def clear(self):
        """Clear current document and history"""
        self.current_doc = None
        self.conversation_history = []
        if self.collection:
            try:
                self.chroma_client.delete_collection('documents')
                self.collection = None
            except:
                pass
        print("🗑️  Cleared document and history")
