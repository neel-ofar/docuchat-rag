# 🤖 DocuChat - RAG-Powered Documentation Assistant

An intelligent document assistant that uses **Retrieval-Augmented Generation (RAG)**, **Large Language Models (LLMs)**, and **Hugging Face APIs** to answer questions about your documents with accurate, context-aware responses.

## 🌐 Live Demo

🔗 **[Try it live on Render](https://docuchat-rag.onrender.com/)** 

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![HuggingFace](https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)
![License](https://img.shields.io/badge/License-MIT-red.svg)

## 🌟 Features

- **📄 Multi-Format Support**: Upload PDF, DOCX, and TXT documents
- **🔍 Intelligent Search**: Uses FAISS vector database for semantic search
- **🤖 AI-Powered Answers**: Leverages Hugging Face LLMs (Mistral-7B by default)
- **📚 Source Citations**: Shows exact sources for each answer
- **💬 Conversation History**: Maintains context across questions
- **🎨 Modern UI**: Clean, responsive interface with drag-and-drop
- **⚡ Fast Processing**: Efficient document chunking and embedding

## 🏗️ Architecture

```
┌─────────────┐
│   User      │
│  Interface  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│      Flask Backend              │
│  ┌──────────┐  ┌──────────┐   │
│  │   API    │  │   RAG    │   │
│  │ Endpoints│◄─┤  Engine  │   │
│  └──────────┘  └─────┬────┘   │
└────────────────────────┼────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
┌──────────────┐ ┌─────────────┐ ┌────────────┐
│  Embeddings  │ │    FAISS    │ │ Hugging    │
│   Model      │ │   Vector    │ │  Face      │
│ (all-MiniLM) │ │  Database   │ │   LLM      │
└──────────────┘ └─────────────┘ └────────────┘
```

## 🛠️ Technology Stack

- **Backend**: Flask (Python web framework)
- **RAG Framework**: LangChain
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **LLM**: Mistral-7B-Instruct-v0.2 (via Hugging Face Inference API)
- **Document Processing**: PyPDF2, python-docx
- **Frontend**: HTML, CSS, JavaScript

## 📋 Prerequisites

- Python 3.8 or higher
- Hugging Face account (free)
- 4GB+ RAM recommended

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/neel-ofar/docuchat-rag.git
cd docuchat-rag
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Get your Hugging Face token from: https://huggingface.co/settings/tokens

Edit `.env` and add your token:
```
HUGGINGFACE_TOKEN=hf_your_actual_token_here
```

### 5. Create Required Directories

```bash
mkdir uploads templates
```

## 🎯 Usage

### Starting the Application

```bash
python app.py
```

The application will start at: http://localhost:5000

### Using DocuChat

1. **Upload Documents**
   - Click or drag-and-drop PDF, DOCX, or TXT files
   - Wait for processing confirmation

2. **Ask Questions**
   - Type your question in the chat input
   - Receive AI-generated answers with source citations

3. **View Sources**
   - Each answer shows which document chunks were used
   - Click to see relevant excerpts

4. **Clear Session**
   - Reset all documents and conversation history

## 📁 Project Structure

```
docuchat-rag/
├── app.py                 # Flask application & API endpoints
├── rag_engine.py          # RAG implementation with Hugging Face
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── README.md             # This file
├── templates/
│   └── index.html        # Frontend interface
└── uploads/              # Uploaded documents (auto-created)
```

## 🔧 Configuration

### Change LLM Model

Edit `rag_engine.py`:

```python
self.llm_model = "meta-llama/Llama-2-7b-chat-hf"  # Or any Hugging Face model
```

Popular alternatives:
- `mistralai/Mixtral-8x7B-Instruct-v0.1`
- `meta-llama/Llama-2-13b-chat-hf`
- `google/flan-t5-xl`

### Adjust Chunk Size

In `rag_engine.py`:

```python
self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Increase for more context
    chunk_overlap=100,    # Overlap between chunks
)
```

### Modify Retrieval Count

Change number of retrieved chunks:

```python
docs = self.vectorstore.similarity_search(question, k=5)  # Default: 3
```

## 🧪 Testing

### Sample Questions to Try

After uploading a document:

- "What is the main topic of this document?"
- "Summarize the key points"
- "What does it say about [specific topic]?"
- "Can you explain [concept] mentioned in the document?"

## 🎓 Learning Outcomes

This project demonstrates:

1. **RAG Architecture**: Combining retrieval and generation
2. **Vector Databases**: Using FAISS for semantic search
3. **LLM Integration**: Working with Hugging Face APIs
4. **Document Processing**: Handling multiple file formats
5. **Full-Stack Development**: Flask backend + responsive frontend
6. **AI/ML Pipeline**: End-to-end AI application workflow

## 🚧 Future Enhancements

- [ ] Add user authentication
- [ ] Support more document formats (Excel, PPT)
- [ ] Implement conversation memory with Redis
- [ ] Add document comparison features
- [ ] Deploy to cloud (AWS, GCP, or Heroku)
- [ ] Add multi-language support
- [ ] Implement advanced analytics dashboard
- [ ] Add export conversation as PDF

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Shaik Neelofar**

- LinkedIn: [linkedin.com/in/shaikneelofar-cse](https://www.linkedin.com/in/shaikneelofar-cse/)
- GitHub: [@neel-ofar](https://github.com/neel-ofar)

## 🙏 Acknowledgments

- Hugging Face for providing amazing models and APIs
- LangChain for the RAG framework
- FAISS team for the vector database
- Open-source community

## 📧 Contact

For questions or feedback, please reach out via:
- GitHub Issues
- LinkedIn

---

**⭐ If you find this project helpful, please give it a star!**
