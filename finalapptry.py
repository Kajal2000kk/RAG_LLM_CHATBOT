from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import ollama
import os
import PyPDF2
import docx
from pathlib import Path
import hashlib
import json
import numpy as np
import faiss
import pickle

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'documents'
DATA_PATH = './data'
MODEL_NAME = 'phi'  # Changed from mistral to phi
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

# Initialize components
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# FAISS index and metadata storage
index_path = os.path.join(DATA_PATH, 'faiss_index.bin')
metadata_path = os.path.join(DATA_PATH, 'metadata.pkl')

# Load or create FAISS index
if os.path.exists(index_path) and os.path.exists(metadata_path):
    try:
        index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        print(f"✅ Loaded existing index with {index.ntotal} vectors")
    except Exception as e:
        print(f"⚠️ Error loading index, creating new one: {e}")
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        metadata = {'documents': [], 'filenames': [], 'chunk_ids': []}
else:
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    metadata = {'documents': [], 'filenames': [], 'chunk_ids': []}
    print("✅ Created new FAISS index")

def save_index():
    """Save FAISS index and metadata"""
    try:
        # Ensure data directory exists
        os.makedirs(DATA_PATH, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(index, index_path)
        
        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"💾 Saved index with {index.ntotal} vectors")
    except Exception as e:
        print(f"❌ Error saving index: {e}")
        raise

# Document processing functions
def extract_text_from_pdf(file_path):
    """Extract text from PDF files"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def extract_text_from_docx(file_path):
    """Extract text from DOCX files"""
    doc = docx.Document(file_path)
    return '\n'.join([paragraph.text for paragraph in doc.paragraphs])

def extract_text_from_txt(file_path):
    """Extract text from TXT files"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
    return chunks

def process_document(file_path, filename):
    """Process document and return chunks"""
    ext = Path(filename).suffix.lower()
    
    if ext == '.pdf':
        text = extract_text_from_pdf(file_path)
    elif ext == '.docx':
        text = extract_text_from_docx(file_path)
    elif ext in ['.txt', '.md']:
        text = extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    if not text.strip():
        raise ValueError("No text could be extracted from the document")
    
    return chunk_text(text)

@app.route('/api/upload', methods=['POST'])
def upload_document():
    """Upload and process documents"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save file
        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Process document
        chunks = process_document(file_path, filename)
        
        if not chunks:
            return jsonify({'error': 'No content could be extracted'}), 400
        
        # Generate embeddings
        embeddings = embedding_model.encode(chunks)
        embeddings = np.array(embeddings).astype('float32')
        
        # Add to FAISS index
        index.add(embeddings)
        
        # Store metadata
        for i, chunk in enumerate(chunks):
            metadata['documents'].append(chunk)
            metadata['filenames'].append(filename)
            metadata['chunk_ids'].append(i)
        
        # Save index and metadata
        save_index()
        
        return jsonify({
            'message': 'Document uploaded successfully',
            'filename': filename,
            'chunks': len(chunks)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat queries with RAG"""
    try:
        data = request.json
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Check if we have any documents
        if index.ntotal == 0:
            return jsonify({
                'answer': 'I don\'t have any documents uploaded yet. Please upload some documents first so I can answer your questions based on them.',
                'sources': []
            }), 200
        
        # Generate query embedding
        query_embedding = embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search in FAISS index
        k = min(5, index.ntotal)  # Get top 5 results or less if we have fewer documents
        distances, indices = index.search(query_embedding, k)
        
        # Build context from retrieved documents
        retrieved_docs = []
        sources = []
        
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(metadata['documents']):
                doc = metadata['documents'][idx]
                filename = metadata['filenames'][idx]
                chunk_id = metadata['chunk_ids'][idx]
                
                retrieved_docs.append(doc)
                sources.append({
                    'filename': filename,
                    'chunk_id': chunk_id,
                    'text': doc[:200] + '...' if len(doc) > 200 else doc,
                    'relevance': float(1 / (1 + distance))  # Convert distance to similarity score
                })
        
        context = "\n\n".join(retrieved_docs)
        
        # Create prompt for Phi
        prompt = f"""You are a helpful assistant. Answer the user's question based on the following context. If the context doesn't contain relevant information, say so politely and offer to help in other ways.

Context:
{context}

Question: {query}

Answer:"""
        
        # Generate response using Ollama with Phi model
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt,
            stream=False
        )
        
        answer = response['response']
        
        return jsonify({
            'answer': answer,
            'sources': sources
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents', methods=['GET'])
def list_documents():
    """List all uploaded documents"""
    try:
        # Get unique filenames
        filenames = list(set(metadata['filenames']))
        
        return jsonify({
            'documents': filenames,
            'total_chunks': len(metadata['documents'])
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete/<filename>', methods=['DELETE'])
def delete_document(filename):
    """Delete a document and its chunks"""
    try:
        # Find indices to keep
        indices_to_keep = []
        new_documents = []
        new_filenames = []
        new_chunk_ids = []
        
        for i, fname in enumerate(metadata['filenames']):
            if fname != filename:
                indices_to_keep.append(i)
                new_documents.append(metadata['documents'][i])
                new_filenames.append(metadata['filenames'][i])
                new_chunk_ids.append(metadata['chunk_ids'][i])
        
        chunks_deleted = len(metadata['filenames']) - len(new_filenames)
        
        # Rebuild FAISS index
        global index
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        
        if new_documents:
            # Re-embed and add remaining documents
            embeddings = embedding_model.encode(new_documents)
            embeddings = np.array(embeddings).astype('float32')
            index.add(embeddings)
        
        # Update metadata
        metadata['documents'] = new_documents
        metadata['filenames'] = new_filenames
        metadata['chunk_ids'] = new_chunk_ids
        
        # Save index and metadata
        save_index()
        
        # Delete physical file
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return jsonify({
            'message': f'Document {filename} deleted successfully',
            'chunks_deleted': chunks_deleted
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check if Ollama is running
        ollama.list()
        return jsonify({
            'status': 'healthy',
            'ollama': 'connected',
            'model': MODEL_NAME,
            'documents_indexed': len(set(metadata['filenames'])),
            'total_chunks': len(metadata['documents'])
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("=" * 50)
    print("🚀 RAG Chatbot Backend Starting...")
    print("=" * 50)
    print(f"📊 Indexed documents: {len(set(metadata['filenames']))}")
    print(f"📄 Total chunks: {len(metadata['documents'])}")
    print(f"🤖 Model: {MODEL_NAME}")
    print(f"🔍 Embedding model: {EMBEDDING_MODEL}")
    print("=" * 50)
    print("📡 Endpoints:")
    print("   POST /api/upload - Upload documents")
    print("   POST /api/chat - Chat with your documents")
    print("   GET  /api/documents - List documents")
    print("   DELETE /api/delete/<filename> - Delete document")
    print("   GET  /api/health - Health check")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)