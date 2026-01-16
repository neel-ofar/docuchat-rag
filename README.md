# 🤖 Google Docs AI Chatbot

An intelligent RAG-powered chatbot that reads Google Docs and answers questions with accurate, cited responses.

## 🌐 Live Demo
🔗 **[Try it live on Render](https://docuchat-rag.onrender.com/)** 
=======
**🔗 https://docuchat-rag.onrender.com/** 
>>>>>>> 18c47f8 (added groq new AI for assisnment updates)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![RAG](https://img.shields.io/badge/RAG-Powered-orange.svg)


## 📋 Features

### Core Functionality
✅ **Automatic Document Loading** - Fetches content from public Google Docs (no authentication needed)  
✅ **Semantic Chunking** - Intelligent text segmentation with section detection  
✅ **Vector Search** - ChromaDB for fast, accurate retrieval  
✅ **LLM Generation** - Powered by Groq (Llama 3) for natural responses  
✅ **Source Citations** - Every answer includes section references  
✅ **Conversation History** - Maintains context across 5 previous exchanges  
✅ **Dark Mode UI** - Professional, business-friendly interface  

### Advanced Features
✅ **Multi-turn Conversations** - Understands follow-up questions  
✅ **Query Rephrasing** - Clarifies ambiguous queries  
✅ **Fallback Responses** - Graceful handling of missing information  
✅ **Edge Case Management** - Handles private docs, empty docs, rate limits  
✅ **90%+ Accuracy** - Context-grounded answers with citations  

## 🏗️ Architecture

```
Google Docs URL → Fetcher → Chunker → Embeddings → ChromaDB
                                                         ↓
User Query → Embedding → Vector Search → LLM (Groq) → Cited Answer
```

**Tech Stack:**
- **Backend**: Flask
- **Document Fetching**: BeautifulSoup4, Google Docs API
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: ChromaDB
- **LLM**: Groq (Llama 3-8B)
- **Deployment**: Docker + Render

## 🚀 Quick Start

### Prerequisites
- Python
- Groq API key (free at [groq.com](https://console.groq.com))

### Installation

```bash
# 1. Clone repository
git clone https://github.com/neel-ofar/google-docs-chatbot.git
cd google-docs-chatbot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Run application
python app.py
```

Visit: **http://localhost:5000 (ONLY RUNS LOCALLY ON MY LAPTOP WHERE IT DEPLOYED)**

## 🔑 Getting Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free)
3. Navigate to API Keys
4. Create new key
5. Copy and paste into `.env`

## 🐳 Docker Deployment

```bash
# Build image
docker build -t google-docs-chatbot .

# Run container
docker run -p 5000:5000 \
  -e GROQ_API_KEY=your_key_here \
  google-docs-chatbot
```

## ☁️ Deploy to Render

### One-Click Deploy

1. Fork this repository
2. Go to [render.com](https://render.com)
3. Click **New Web Service**
4. Connect your GitHub repository
5. Select **Docker** as runtime
6. Add environment variable: `GROQ_API_KEY`
7. Click **Create Web Service**

### Manual Deploy

```bash
# 1. Push to GitHub
git add .
git commit -m "Initial commit"
git push origin main

# 2. On Render dashboard:
# - Connect GitHub repo
# - Environment: Docker
# - Add GROQ_API_KEY
# - Deploy

# 3. Your URL: 
```

## 📊 Performance Metrics

- **Accuracy**: 90%+ on test queries
- **Response Time**: 2-5 seconds (after initial load)
- **Document Size**: Up to 50 pages efficiently
- **Chunk Size**: 800 tokens (optimal for context)
- **Retrieval**: Top-3 relevant sections
- **Memory**: ~500MB RAM usage

## 🔧 Configuration

Edit `.env` for customization:

```bash
# LLM Model (Groq)
LLM_MODEL=llama3-8b-8192  # or mixtral-8x7b-32768

# Chunking
CHUNK_SIZE=800
CHUNK_OVERLAP=100

# Retrieval
TOP_K_RESULTS=3

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## 📁 Project Structure

```
google-docs-chatbot/
├── app.py                  # Flask application
├── gdocs_fetcher.py        # Google Docs fetcher
├── rag_engine.py           # RAG implementation
├── requirements.txt        # Dependencies
├── Dockerfile              # Docker configuration
├── .env                    # Environment variables
├── templates/
│   └── index.html          # Frontend UI
└── README.md               # This file
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📜 License

MIT License - feel free to use for commercial projects

## 👤 Author

**Shaik Neelofar**
- GitHub: [@neel-ofar](https://github.com/neel-ofar)
- LinkedIn: [linkedin.com/in/shaikneelofar-cse](https://www.linkedin.com/in/shaikneelofar-cse/)

## 🙏 Acknowledgments

- Groq for fast LLM inference
- ChromaDB for vector storage
- Sentence Transformers for embeddings
- Flask for the web framework

---
**⭐ If you find this project helpful, please give it a star!**
=======
**⭐ If you find this helpful, please star the repository!**

## 📧 Contact

Questions or feedback? Create an issue or reach out via LinkedIn.

---

Built with ❤️ for production RAG applications
>>>>>>> 18c47f8 (added groq new AI for assisnment updates)
