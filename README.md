# 🤖 3D Floating RAG Chatbot

A beautiful, interactive AI chatbot with RAG (Retrieval-Augmented Generation) capabilities. Upload your documents and chat with them using a stunning 3D animated interface!

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- 🎨 **3D Animated UI** - Smooth floating animations and interactive effects
- 📄 **Multi-Format Support** - Upload PDF, DOCX, TXT, and MD files
- 🧠 **RAG Technology** - Intelligent retrieval-augmented generation
- 🔍 **FAISS Vector Search** - Fast and efficient similarity search
- 💬 **Real-time Chat** - Interactive conversation with your documents
- 📱 **Responsive Design** - Works perfectly on desktop and mobile
- 🎭 **Modern UI/UX** - Glassmorphic design with smooth animations

## 🎥 Demo

The chatbot features:
- Floating button with pulsing glow effect
- Smooth slide-in chat window animation
- 3D rotating bot avatar
- Typing indicators
- Real-time message animations
- Document upload with visual feedback

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Ollama installed locally
- 4GB+ RAM recommended

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install and setup Ollama**

Download Ollama from [ollama.ai](https://ollama.ai)

Pull the Phi model:
```bash
ollama pull phi
```

5. **Run the backend**
```bash
python finalapptry.py
```

The backend will start on `http://localhost:5000`

6. **Open the frontend**

Simply open `test_upload_page.html` in your browser, or serve it using:
```bash
python -m http.server 8000
```

Then visit `http://localhost:8000/test_upload_page.html`

## 📦 Project Structure

```
rag-chatbot/
│
├── finalapptry.py              # Flask backend server
├── test_upload_page.html       # 3D Floating chatbot UI
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── documents/                  # Uploaded documents storage
│   └── (uploaded files)
│
└── data/                       # FAISS index and metadata
    ├── faiss_index.bin        # Vector database
    └── metadata.pkl           # Document metadata
```

## 🔧 Configuration

### Backend Configuration (`finalapptry.py`)

```python
UPLOAD_FOLDER = 'documents'      # Document storage path
DATA_PATH = './data'              # FAISS index path
MODEL_NAME = 'phi'                # Ollama model name
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # Sentence transformer model
EMBEDDING_DIM = 384               # Embedding dimensions
```

### Frontend Configuration (`test_upload_page.html`)

```javascript
const API_URL = 'http://localhost:5000';  // Backend API URL
```

## 📚 API Endpoints

### Health Check
```http
GET /api/health
```
Returns backend status and document count.

### Upload Document
```http
POST /api/upload
Content-Type: multipart/form-data

file: <document file>
```

### Chat Query
```http
POST /api/chat
Content-Type: application/json

{
  "query": "Your question here"
}
```

### List Documents
```http
GET /api/documents
```

### Delete Document
```http
DELETE /api/delete/<filename>
```

## 🛠️ Technologies Used

### Backend
- **Flask** - Web framework
- **Ollama** - Local LLM inference (Phi model)
- **Sentence Transformers** - Text embeddings
- **FAISS** - Vector similarity search
- **PyPDF2** - PDF processing
- **python-docx** - DOCX processing

### Frontend
- **HTML5/CSS3** - Modern web standards
- **Vanilla JavaScript** - No framework dependencies
- **CSS Animations** - Smooth 3D effects
- **Fetch API** - Async HTTP requests

## 🎨 UI Features

### Floating Button
- Smooth floating animation
- Hover effects with scaling and rotation
- Pulsing glow effect
- Notification badge

### Chat Window
- Glassmorphic design with backdrop blur
- Slide-in animation with spring effect
- 3D rotating bot avatar
- Real-time typing indicators
- Message animations

### Document Upload
- Drag-and-drop support
- Visual upload feedback
- Document list with delete option
- File type validation

## 🔒 Security Notes

- CORS is enabled for development (configure for production)
- File type validation on both frontend and backend
- Size limits should be configured for production use
- Consider adding authentication for production deployment

## 🐛 Troubleshooting

### Backend Won't Start
- Ensure Python 3.8+ is installed
- Check if port 5000 is available
- Verify all dependencies are installed

### Ollama Connection Error
- Make sure Ollama is running: `ollama serve`
- Verify Phi model is downloaded: `ollama pull phi`
- Check Ollama is accessible at default port

### Frontend Connection Error
- Verify backend is running on port 5000
- Check browser console for CORS errors
- Ensure API_URL in frontend matches backend URL

### No Documents Showing
- Check `documents/` folder permissions
- Verify FAISS index is being created in `data/` folder
- Check browser console for upload errors

## 📈 Performance Tips

1. **Chunk Size**: Adjust `chunk_size` in `chunk_text()` for optimal performance
2. **Top K Results**: Modify `k = min(5, index.ntotal)` to retrieve more/fewer documents
3. **Model Selection**: Switch to smaller/larger Ollama models based on needs
4. **Embedding Model**: Choose different sentence-transformers models for speed vs accuracy

## 🚀 Deployment

### Production Checklist

- [ ] Set `debug=False` in Flask
- [ ] Configure proper CORS origins
- [ ] Add authentication/authorization
- [ ] Set up proper file size limits
- [ ] Use production WSGI server (Gunicorn/uWSGI)
- [ ] Add rate limiting
- [ ] Set up HTTPS
- [ ] Configure proper logging
- [ ] Add error monitoring (Sentry)
- [ ] Set up backup for FAISS index

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai) - Local LLM inference
- [Sentence Transformers](https://www.sbert.net/) - Embedding models
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [Flask](https://flask.palletsprojects.com/) - Web framework



⭐ Star this repo if you find it helpful!

## 🗺️ Roadmap

- [ ] Support for more file formats (CSV, JSON)
- [ ] Multi-language support
- [ ] Chat history persistence
- [ ] Voice input/output
- [ ] Document preview
- [ ] Advanced search filters
- [ ] User authentication
- [ ] Cloud deployment guide
- [ ] Docker containerization
- [ ] API documentation with Swagger

## 📸 Screenshots

*Add your screenshots here*

### Main Interface
<img width="1736" height="897" alt="image" src="https://github.com/user-attachments/assets/04a15eec-df6d-4f9b-95e6-10904c063654" />

### Document Upload
<img width="1502" height="928" alt="image" src="https://github.com/user-attachments/assets/3e15e4b2-a52b-4e33-a9e8-1d0057d92cee" />


