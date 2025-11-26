# PDF Chatbot - Offline RAG System

A completely offline PDF chatbot built with Streamlit that uses Ollama for local LLM inference. Upload PDFs and ask questions - everything runs locally on your machine!

## Features

- 📄 **Document Upload**: Upload PDF, TXT, DOCX, and Excel files
- 💬 **Question Answering**: Ask questions about your documents using RAG (Retrieval-Augmented Generation)
- 🔒 **100% Offline**: Uses Ollama for local LLM inference - no internet required after setup
- 🚀 **Multi-Model Support**: Automatically selects optimal model based on query complexity
- 🎯 **Smart Retrieval**: Uses FAISS vector search with sentence transformers
- 📊 **Persistent Knowledge Base**: Uploaded documents are saved and persist across sessions

## How It Works

1. **Document Processing**: Extracts text from documents and splits into chunks
2. **Embedding**: Creates vector embeddings using sentence transformers (all-MiniLM-L6-v2)
3. **Vector Store**: Stores embeddings in FAISS for fast similarity search
4. **RAG Query**: 
   - Searches for relevant chunks using FAISS
   - Sends context + question to Ollama (local LLM)
   - Generates answer based on retrieved context

## Installation

### Prerequisites

- Python 3.8 or higher
- Ollama installed (download from https://ollama.ai)
- At least one Ollama model pulled (e.g., `ollama pull llama3.2:1b`)

### Setup Steps

1. **Install Ollama**:
   - Download from: https://ollama.ai
   - Install and start Ollama
   - Pull a model: `ollama pull llama3.2:1b` (or `llama3.2:3b`, `mistral`, etc.)

2. **Navigate to the chatbot directory**:
   ```bash
   cd chatbot
   ```

3. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

4. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

5. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

6. **First Run**:
   - The embedding model will download on first run (requires internet once)
   - After that, everything works offline
   - Models are cached locally

## Usage

1. **Start Ollama** (if not auto-started):
   ```bash
   ollama serve
   ```

2. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```
   Or double-click `run.bat` on Windows.

3. **Upload Documents**:
   - Use the sidebar to upload PDF, TXT, DOCX, or Excel files
   - Or load a folder of documents using "Load Databank"

4. **Ask Questions**:
   - Type your questions in the chat interface
   - Get answers based on your uploaded documents

## Models Used

- **Embedding Model**: `all-MiniLM-L6-v2` (sentence-transformers)
  - Creates 384-dimensional embeddings
  - Fast and efficient
  - Works offline after first download

- **LLM Models** (via Ollama):
  - **llama3.2:1b**: Very fast, good quality (~1.3GB)
  - **llama3.2:3b**: Fast, very good quality (~2.0GB)
  - **mistral**: High quality (~4.1GB)
  - **phi**: Fast and efficient (~1.6GB)
  - Or any other Ollama model you prefer

## Offline Operation

- ✅ **Ollama**: Works offline after installation and model download
- ✅ **Embedding Model**: Works offline after first download
- ✅ **No internet required** after initial setup
- ✅ **All processing happens on your machine**
- ✅ **Your data never leaves your computer**

## Features

### KB Modes
- **Base Mode**: Only searches in original knowledge base documents
- **Full Mode**: Searches in all documents (original + uploaded)

### Model Selection
- **Auto**: Automatically selects best model based on query complexity and system resources
- **Manual**: Choose specific model for consistent results

### Query Options
- **Detailed Answer**: Get comprehensive, structured responses
- **Target File**: Query specific document instead of all files
- **Numerical Queries**: Special handling for questions with numbers/statistics

## Troubleshooting

### Ollama not detected
- Make sure Ollama is installed and running
- Check: `ollama list` (should show your models)
- Start Ollama: `ollama serve` (if not auto-started)
- Restart the Streamlit app

### Model not found
- Make sure you've pulled the model: `ollama pull llama3.2:1b`
- Check available models: `ollama list`
- The app will show available models in the sidebar

### Embedding model download issues
- First run requires internet to download the embedding model (~80MB)
- After download, model is cached locally and works offline
- Check your internet connection for first-time setup

### Slow performance
- Use smaller models (llama3.2:1b) for faster responses
- Large PDFs may take longer to process initially
- Search and answer generation are fast after indexing

### PDF not processing
- Make sure the PDF contains extractable text (not just images)
- Try a different PDF file
- Check that PyPDF2/pypdf can read the PDF format

## File Structure

```
chatbot/
├── app.py              # Streamlit GUI application
├── chatbot.py          # Backend RAG system
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── run.bat            # Windows startup script
├── knowledge_base/    # Original KB (if exists)
├── persistent_kb/     # Uploaded documents storage
└── embedding_models/  # Cached embedding models
```

## Technical Details

- **PDF Processing**: PyPDF2/pypdf for text extraction
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS for efficient similarity search
- **LLM**: Ollama API (supports Llama, Mistral, Phi, etc.)
- **UI Framework**: Streamlit

## Notes

- **Ollama Setup**: Required for LLM inference. Install from https://ollama.ai
- **Model Size**: 
  - llama3.2:1b: ~1.3GB
  - llama3.2:3b: ~2.0GB
  - mistral: ~4.1GB
- **Works best with PDFs** that contain extractable text (not scanned images)
- **First run**: Requires internet only for embedding model download
- **After setup**: Works completely offline

## License

This project uses open-source models and libraries. Please refer to their respective licenses:
- Sentence Transformers: Apache 2.0
- FAISS: MIT
- Streamlit: Apache 2.0
- Ollama: MIT
