# PDF QA Chatbot System

A complete PDF Question-Answering system built with sentence transformers, LSTM models, and Streamlit. This system loads PDFs, creates embeddings using pre-trained sentence transformers, optionally trains an LSTM model, and provides a user-friendly interface for asking questions about your documents.

## Features

- 📄 **PDF Loading**: Load PDFs from a folder and automatically create text chunks
- 🔢 **Pre-trained Embeddings**: Uses sentence transformers (all-MiniLM-L6-v2) - no training needed!
- 🧠 **Optional LSTM Training**: Train an LSTM model for enhanced question-answer matching (optional)
- 🔍 **Hybrid Search**: Combines semantic search (sentence transformers) with keyword search (BM25)
- 💬 **Streamlit UI**: Beautiful, interactive UI for querying your documents
- 💾 **Persistent Storage**: All models and embeddings are saved locally
- ✅ **Easy PDF Updates**: Add new PDFs without retraining the embedding model

## Architecture

1. **Data Loader** (`data_loader.py`): Loads PDFs and creates overlapping text chunks
2. **Embeddings** (`embeddings.py`): Uses pre-trained sentence transformers for embeddings
3. **LSTM Model** (`lstm_model.py`): Optional bidirectional LSTM for question-answer matching
4. **Similarity Search** (`similarity_search.py`): Performs cosine similarity search on embeddings
5. **QA System** (`qa_system.py`): Integrates all components with hybrid search
6. **Streamlit UI** (`app_new.py`): User interface for interaction
7. **Training Script** (`train_model.py`): Interactive script for processing PDFs

## Installation

### Prerequisites

- Python 3.8 or higher
- Windows/Linux/Mac

### Setup Steps

1. **Navigate to the pdf_chatbot_lstm directory**:
   ```bash
   cd pdf_chatbot_lstm
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements_new.txt
   ```
   
   This will install all required packages including PyMuPDF for fast PDF text extraction.

## Usage

### Workflow Overview

The system follows a two-step workflow:
1. **Training (VS Code/Terminal)**: Use `train_model.py` to upload PDFs and create embeddings
2. **Querying (Streamlit UI)**: Use `app_new.py` to ask questions about your documents

### Step 1: Train/Process PDFs

#### Run the Training Script

1. Open `train_model.py` in VS Code or your terminal
2. Run the script:
   ```bash
   python train_model.py
   ```

#### Training Process

The script will guide you through:

1. **Configuration** (Optional):
   - Chunk size (default: 1000, recommended 500-2000)
   - Overlap size (default: chunk_size/5)

2. **Load PDFs**:
   - **Option 1**: Enter path to folder containing PDFs
     ```
     Enter path to folder containing PDFs: C:\Users\Documents\my_pdfs
     ```
   - **Option 2**: Enter path to a single PDF file
     ```
     Enter path to a single PDF file: C:\Users\Documents\my_file.pdf
     ```
   - The script will create text chunks from your PDFs

3. **Create Embeddings**:
   - When asked "Retrain embedding model? (y/n, default=n)": Answer **`n`**
   - The pre-trained sentence transformer will encode all chunks
   - This is fast - no training needed!

4. **Train LSTM Model** (Optional):
   - When asked "Do you want to train LSTM model? (y/n, default=n)": Answer **`n`**
   - LSTM is optional - the system works perfectly with similarity search alone
   - If you choose to train, you can specify:
     - Number of epochs (default: 20)
     - Batch size (default: 32)

#### Example Training Session

```
Configuration:
Chunk size (default=1000): [Enter]
Overlap size (default=200): [Enter]

STEP 1: Load PDFs
Enter path to folder containing PDFs: C:\Users\Documents\my_pdfs
Loading PDFs from: C:\Users\Documents\my_pdfs
✅ Loaded 150 chunks from folder

STEP 2: Create Embeddings
Retrain embedding model? (y/n, default=n): n
Creating embeddings...
✅ Embeddings created successfully

STEP 3: Optional - Train LSTM Model
Do you want to train LSTM model? (y/n, default=n): n
⏭️ Skipping LSTM training. Using similarity search only.

PROCESSING COMPLETE!
✅ Chunks: 150
✅ Embeddings: (150, 384)
✅ Using: Similarity Search (no training needed)
```

### Step 2: Query Your Documents

#### Start the Query Interface

1. Run the Streamlit app:
   ```bash
   streamlit run app_new.py
   ```
   Or double-click `run_new.bat` on Windows.

2. The app will automatically:
   - Load trained chunks
   - Load embeddings
   - Load LSTM model (if trained)
   - Initialize similarity search

3. **Ask Questions**:
   - Type your question in the chat interface
   - Get answers based on retrieved chunks
   - View similarity scores and source files

## Adding New PDFs

### Quick Answer

**❌ NO, you don't need to retrain the embedding model!**

The sentence transformer model is **pre-trained** and doesn't need retraining. You just need to process the new PDFs and the system will automatically merge them with existing data.

### Step-by-Step Instructions

1. **Navigate to the pdf_chatbot_lstm directory**:
   ```bash
   cd pdf_chatbot_lstm
   ```

2. **Run the training script**:
   ```bash
   python train_model.py
   ```

3. **Follow the interactive prompts**:
   - **Configuration**: Press Enter for defaults
   - **Load New PDFs**: Enter path to your new PDFs (folder or single file)
   - **Retrain embedding model?**: Answer **`n`** (just press Enter)
   - **Train LSTM model?**: Answer **`n`** (just press Enter)

4. **Done!** The system will:
   - Process new PDFs and create chunks
   - **Append** new chunks to existing ones (not replace!)
   - Create embeddings for new chunks only (fast!)
   - Refit BM25 on all chunks (old + new)

### What Happens Behind the Scenes

1. ✅ **New PDFs are processed** → Text extracted and chunked
2. ✅ **Chunks are appended** → Added to existing `chunks.pkl` (not replaced!)
3. ✅ **Embeddings created** → Only for new chunks (fast, uses pre-trained model)
4. ✅ **BM25 refitted** → Re-indexes all chunks (old + new) for keyword search
5. ✅ **No model training** → Sentence transformer is pre-trained, just encodes text

### Example Workflow

```bash
# First time: Load initial PDFs
python train_model.py
# Enter: C:\Users\Documents\initial_pdfs
# Result: 1000 chunks created

# Later: Add new PDFs
python train_model.py
# Enter: C:\Users\Documents\new_pdfs
# Result: 500 new chunks added (total: 1500 chunks)

# Even later: Add more PDFs
python train_model.py
# Enter: C:\Users\Documents\more_pdfs
# Result: 300 new chunks added (total: 1800 chunks)
```

### Important Notes

✅ **Existing PDFs are preserved** - Your old PDFs remain in the system  
✅ **No retraining needed** - Sentence transformers are pre-trained  
✅ **Fast process** - Only new chunks are processed  
✅ **Automatic merging** - Old and new chunks work together seamlessly  
✅ **Can add PDFs incrementally** - Run the script multiple times  

## How It Works

### 1. PDF Loading and Chunking
- Extracts text from PDF files using **PyMuPDF** (fast direct text extraction)
- **Fast Processing**: PyMuPDF extracts text directly from PDFs
- **Fallback Support**: Falls back to pypdf/PyPDF2 if PyMuPDF is not available
- Uses **paragraph-based chunking** for better context preservation
- Respects paragraph boundaries and natural text structure
- Splits text into chunks based on paragraphs (default: 1000 chars with 200 char overlap)
- Stores chunks with metadata (source file, position, chunk_id, etc.)

### 2. Embedding Creation
- Uses pre-trained sentence transformer model (`all-MiniLM-L6-v2`)
- Creates 384-dimensional embeddings for all chunks
- No training required - model is pre-trained!
- Also initializes BM25 for hybrid keyword search

### 3. LSTM Training (Optional)
- Builds a bidirectional LSTM model
- Creates training pairs (positive: chunk with itself, negative: random chunks)
- Trains the model to learn question-answer similarity
- Saves the trained model for future use
- **Note**: This is optional - similarity search works great without it!

### 4. Query Processing
- Encodes user question using the same sentence transformer
- Performs hybrid search:
  - **Semantic search**: Cosine similarity on sentence transformer embeddings (70% weight)
  - **Keyword search**: BM25 scores for exact keyword matches (30% weight)
- Retrieves top-k most similar chunks
- Generates answer from retrieved context

## File Structure

```
pdf_chatbot_lstm/
├── app_new.py              # Streamlit UI application
├── qa_system.py            # Main QA system integration
├── train_model.py          # Training/processing script
├── data_loader.py          # PDF loading and chunking
├── embeddings.py           # Sentence transformer embeddings
├── lstm_model.py           # LSTM model training (optional)
├── similarity_search.py    # Similarity search module
├── requirements_new.txt    # Python dependencies
├── run_new.bat            # Windows startup script
├── README.md              # This file
├── data/                  # Stored chunks and embeddings
│   ├── chunks.pkl         # Text chunks from PDFs
│   └── chunk_embeddings.npy # Embeddings for chunks
├── models/                # Trained models
│   ├── embedder/          # Embedding model config (if using custom)
│   └── lstm/              # LSTM model (if trained)
└── pdfs/                  # Sample PDFs (optional)
```

## Configuration

### Data Loader
- `chunk_size`: Maximum characters per chunk (default: 1000)
- `overlap`: Overlapping characters between chunks (default: 200)
- **Uses LangChain**: Chunks are created based on context (paragraphs, sentences) rather than fixed character boundaries
- **Separators Priority**: Paragraphs (`\n\n`) → Lines (`\n`) → Sentences (`.`, `!`, `?`) → Commas → Words

### Embeddings
- **Model**: `all-MiniLM-L6-v2` (sentence-transformers)
- **Dimension**: 384 (fixed by model)
- **No configuration needed** - pre-trained model!

### LSTM Model (Optional)
- `lstm_units`: Number of LSTM units (default: 64)
- `epochs`: Training epochs (default: 20, adjustable in script)
- `batch_size`: Batch size for training (default: 32)

### Similarity Search
- `top_k`: Number of chunks to retrieve (default: 5)
- Hybrid search weights: 70% semantic, 30% keyword

## Technical Details

- **Embedding Model**: Pre-trained sentence transformer (`all-MiniLM-L6-v2`)
  - 384-dimensional embeddings
  - Fast and efficient
  - Works offline after first download
- **LSTM Model**: Bidirectional LSTM with dropout (optional)
- **Similarity Metric**: Cosine similarity between embeddings
- **Hybrid Search**: Combines semantic (sentence transformers) + keyword (BM25)
- **Storage**: Pickle for chunks, NumPy for embeddings, H5 for models

## Advantages

✅ **No API Backend**: Everything runs locally  
✅ **Pre-trained Embeddings**: No training needed for embeddings  
✅ **Fast Processing**: Only new chunks are processed when adding PDFs  
✅ **Hybrid Search**: Combines semantic and keyword search for better results  
✅ **Persistent**: All models and data are saved locally  
✅ **User-Friendly**: Simple Streamlit interface  
✅ **Easy Updates**: Add new PDFs without retraining  

## Troubleshooting

### PDFs not loading
- **Text-based PDFs**: Should work automatically with PyMuPDF (primary method) or PyPDF2/pypdf (fallback)
- Check that the folder path is correct
- Verify PyMuPDF is installed: `pip install PyMuPDF`
- The system uses PyMuPDF first (fastest), then falls back to pypdf/PyPDF2 if needed
- **Note**: Scanned PDFs (image-based) may not work well as they require OCR which is not included

### Embedding creation fails
- First run requires internet to download the sentence transformer model (~80MB)
- After download, model is cached locally and works offline
- Check your internet connection for first-time setup

### LSTM training is slow
- LSTM training is optional - you can skip it!
- If training, reduce the number of epochs
- Use smaller batch size if memory is limited
- Training time depends on number of chunks

### Query interface shows "Not Ready"
- Make sure training/processing completed successfully
- Check that `data/` folder exists
- Verify `chunks.pkl` and `chunk_embeddings.npy` exist
- Run `train_model.py` first to process PDFs

### Out of memory errors
- Reduce `chunk_size` in data loader
- Process fewer PDFs at once
- Close other applications to free up memory

### Adding new PDFs - old PDFs removed?
- **No!** Chunks are appended, not replaced
- Your old PDFs remain in the system
- Check `data/chunks.pkl` - it should contain all chunks

### How long does processing take?
- Small PDFs (few pages): 1-2 minutes
- Medium PDFs (100 pages): 5-10 minutes
- Large PDFs (1000+ pages): 15-30 minutes
- Adding new PDFs to existing system: Only processes new PDFs (faster!)

### Want to start fresh?
- Delete the `data/` folder
- Delete the `models/` folder (optional)
- Run `train_model.py` again

## Tips

- **First Run**: Takes longer as the sentence transformer model downloads (~80MB)
- **Subsequent Runs**: Much faster - models are cached locally
- **Adding PDFs**: Very fast - only new chunks are processed
- **LSTM Training**: Optional but can improve results for specific use cases
- **Chunk Size**: Larger chunks = more context but fewer chunks. Smaller chunks = more chunks but less context per chunk
- **Hybrid Search**: Works best when you have both semantic similarity and keyword matches

## Notes

- First run will download the sentence transformer model (requires internet once)
- After setup, everything works offline
- Training the LSTM model is optional - similarity search works great without it
- All data is stored locally in `data/` and `models/` directories
- You can add PDFs incrementally - run `train_model.py` multiple times
- The system automatically merges old and new PDFs
- **Note**: This system works best with text-based PDFs. Scanned PDFs (image-based) may not extract text properly

## License

This project uses open-source libraries. Please refer to their respective licenses:
- Sentence Transformers: Apache 2.0
- Streamlit: Apache 2.0
- NumPy: BSD
- scikit-learn: BSD
- TensorFlow: Apache 2.0 (if using LSTM)

---

## Quick Reference

### Initial Setup
```bash
cd pdf_chatbot_lstm
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements_new.txt
python train_model.py
```

### Add New PDFs
```bash
python train_model.py
# Enter path to new PDFs
# Answer 'n' to retrain embedding model
# Answer 'n' to LSTM training
```

### Query Documents
```bash
streamlit run app_new.py
```

### File Locations
- Chunks: `data/chunks.pkl`
- Embeddings: `data/chunk_embeddings.npy`
- Models: `models/embedder/` and `models/lstm/`

---

**That's it!** You now have a complete PDF QA system that works offline and can easily be updated with new documents. 🎉

