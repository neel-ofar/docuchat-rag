import os
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Fixed imports for Windows
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import PyPDF2
try:
    import docx
except ImportError:
    import python_docx as docx

from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

class RAGEngine:
    """RAG Engine using Hugging Face models"""
    
    def __init__(self):
        print("🔧 Initializing RAG Engine...")
        
        try:
            # Initialize embeddings model from Hugging Face
            print("📥 Loading embedding model (this may take a minute)...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("✅ Embedding model loaded!")
            
            # Initialize Hugging Face Inference Client for LLM
            hf_token = os.getenv('HUGGINGFACE_TOKEN', '')
            if not hf_token:
                print("⚠️  Warning: No Hugging Face token found. LLM features may be limited.")
            
            self.client = InferenceClient(token=hf_token) if hf_token else None
            
            # Use a good open-source model (can be changed)
            self.llm_model = "HuggingFaceH4/zephyr-7b-beta"
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len,
            )
            
            # Vector store
            self.vectorstore = None
            self.documents = []
            self.document_metadata = []
            
            print("✅ RAG Engine initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing RAG Engine: {e}")
            raise
    
    def extract_text_from_file(self, filepath: str) -> str:
        """Extract text from various file formats"""
        ext = filepath.lower().split('.')[-1]
        
        try:
            if ext == 'txt':
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            
            elif ext == 'pdf':
                text = ""
                try:
                    with open(filepath, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        for page in pdf_reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                except Exception as e:
                    print(f"⚠️  PDF extraction error: {e}")
                return text
            
            elif ext == 'docx':
                try:
                    doc = docx.Document(filepath)
                    return "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
                except Exception as e:
                    print(f"⚠️  DOCX extraction error: {e}")
                    return ""
            
            else:
                return ""
        
        except Exception as e:
            print(f"❌ Error extracting text: {e}")
            return ""
    
    def add_document(self, filepath: str, filename: str) -> bool:
        """Add a document to the RAG system"""
        try:
            print(f"📄 Processing document: {filename}")
            
            # Extract text
            text = self.extract_text_from_file(filepath)
            if not text or len(text.strip()) == 0:
                print("❌ No text extracted from document")
                return False
            
            print(f"📝 Extracted {len(text)} characters")
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            print(f"✂️  Split into {len(chunks)} chunks")
            
            if len(chunks) == 0:
                print("❌ No chunks created from document")
                return False
            
            # Create metadata for each chunk
            metadatas = [{'source': filename, 'chunk': i} for i in range(len(chunks))]
            
            # Add to vector store
            print("🔄 Creating embeddings...")
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_texts(
                    texts=chunks,
                    embedding=self.embeddings,
                    metadatas=metadatas
                )
            else:
                self.vectorstore.add_texts(
                    texts=chunks,
                    metadatas=metadatas
                )
            
            # Store document info
            self.documents.append(text)
            self.document_metadata.append({
                'filename': filename,
                'chunks': len(chunks),
                'filepath': filepath
            })
            
            print(f"✅ Document added successfully!")
            return True
        
        except Exception as e:
            print(f"❌ Error adding document: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def query(self, question: str, history: List = None) -> Tuple[str, List[dict]]:
        """Query the RAG system"""
        try:
            if self.vectorstore is None:
                return "Please upload a document first before asking questions.", []
            
            print(f"🔍 Querying: {question}")
            
            # Retrieve relevant documents
            docs = self.vectorstore.similarity_search(question, k=3)
            
            if not docs:
                return "No relevant information found in the uploaded documents.", []
            
            # Prepare context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Prepare sources
            sources = []
            for doc in docs:
                sources.append({
                    'source': doc.metadata.get('source', 'Unknown'),
                    'chunk': doc.metadata.get('chunk', 0),
                    'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })
            
            # Create prompt with context
            prompt = self._create_prompt(question, context, history)
            
            # Generate answer using Hugging Face model
            answer = None
            if self.client:
                try:
                    print("🤖 Generating answer with LLM...")
                    response = self.client.text_generation(
                        prompt,
                        model=self.llm_model,
                        max_new_tokens=512,
                        temperature=0.7,
                        top_p=0.95,
                        return_full_text=False
                    )
                    answer = response.strip()
                    print("✅ LLM answer generated!")
                except Exception as e:
                    print(f"⚠️  LLM API error: {e}")
                    answer = None
            
            # Fallback to extractive answer if LLM fails
            if not answer:
                print("📝 Using extractive answer...")
                answer = self._extractive_answer(question, context)
            
            return answer, sources
        
        except Exception as e:
            print(f"❌ Error during query: {e}")
            import traceback
            traceback.print_exc()
            return f"Error processing query: {str(e)}", []
    
    def _create_prompt(self, question: str, context: str, history: List = None) -> str:
        """Create a prompt for the LLM"""
        prompt = f"""<s>[INST] You are a helpful AI assistant that answers questions based on the provided context.

Context information:
{context}

Question: {question}

Please provide a clear and concise answer based on the context above. If the answer is not in the context, say so. [/INST]</s>"""
        
        return prompt
    
    def _extractive_answer(self, question: str, context: str) -> str:
        """Fallback extractive answer when LLM fails"""
        try:
            # Simple extractive approach: return relevant context
            sentences = context.replace('\n', '. ').split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            
            question_words = set(question.lower().split())
            
            # Score sentences by relevance
            scored = []
            for sent in sentences:
                if len(sent) < 10:  # Skip very short sentences
                    continue
                sent_words = set(sent.lower().split())
                score = len(question_words & sent_words)
                if score > 0:
                    scored.append((score, sent))
            
            scored.sort(reverse=True)
            
            if scored:
                # Return top 2-3 most relevant sentences
                answer_sentences = [s[1] for s in scored[:3]]
                return ". ".join(answer_sentences) + "."
            else:
                # If no match, return beginning of context
                return " ".join(sentences[:2]) + "..."
                
        except Exception as e:
            print(f"⚠️  Error in extractive answer: {e}")
            return "I found relevant context but couldn't generate a specific answer. Please rephrase your question."
    
    def clear_documents(self):
        """Clear all documents from the system"""
        self.vectorstore = None
        self.documents = []
        self.document_metadata = []
        print("🗑️  All documents cleared")
    
    def get_document_list(self) -> List[dict]:
        """Get list of uploaded documents"""
        return self.document_metadata
