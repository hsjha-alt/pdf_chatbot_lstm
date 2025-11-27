"""
PDF Processing Script for QA System
Run this script in VS Code to load PDFs and create embeddings
No training required - uses similarity search for queries
After processing, use app_new.py for querying
"""

import os
import sys
from qa_system import QASystem

def main():
    print("="*70)
    print("PDF QA System - Training Script")
    print("="*70)
    print()
    
    # Configuration options
    print("Configuration:")
    chunk_size_input = input("Chunk size (default=1000, recommended 500-2000): ").strip()
    chunk_size = int(chunk_size_input) if chunk_size_input.isdigit() else 1000
    
    overlap_input = input(f"Overlap size (default={chunk_size//5}, recommended {chunk_size//5}-{chunk_size//3}): ").strip()
    overlap = int(overlap_input) if overlap_input.isdigit() else chunk_size // 5
    
    print(f"\nUsing chunk_size={chunk_size}, overlap={overlap}")
    print()
    
    # Initialize QA system with custom chunking
    from data_loader import PDFDataLoader
    qa = QASystem()
    qa.data_loader = PDFDataLoader(chunk_size=chunk_size, overlap=overlap)
    
    # Step 1: Load PDFs
    print("STEP 1: Load PDFs")
    print("-" * 70)
    
    # Option 1: Upload PDFs from a folder
    pdf_folder = input("Enter path to folder containing PDFs (or press Enter to skip): ").strip()
    
    if pdf_folder and os.path.exists(pdf_folder):
        print(f"\nLoading PDFs from: {pdf_folder}")
        chunks = qa.load_pdfs(pdf_folder)
        print(f"✅ Loaded {len(chunks)} chunks from folder")
    else:
        # Option 2: Load single PDF
        pdf_file = input("Enter path to a single PDF file (or press Enter to skip): ").strip()
        if pdf_file and os.path.exists(pdf_file):
            print(f"\nLoading PDF: {os.path.basename(pdf_file)}")
            chunks = qa.load_single_pdf(pdf_file)
            print(f"✅ Loaded {len(chunks)} chunks from file")
        else:
            # Try to load existing chunks
            chunks_path = os.path.join(qa.data_dir, 'chunks.pkl')
            if os.path.exists(chunks_path):
                qa.chunks = qa.data_loader.load_chunks(chunks_path)
                chunks = qa.chunks
                print(f"✅ Loaded {len(chunks)} existing chunks")
            else:
                print("❌ No PDFs provided and no existing chunks found.")
                print("Please provide a PDF folder or file path.")
                return
    
    if not chunks:
        print("❌ No chunks available. Exiting.")
        return
    
    print(f"\nTotal chunks: {len(chunks)}")
    
    # Step 2: Create Embeddings
    print("\n" + "="*70)
    print("STEP 2: Create Embeddings")
    print("-" * 70)
    
    retrain = input("Retrain embedding model? (y/n, default=n): ").strip().lower() == 'y'
    
    print("\nCreating embeddings...")
    qa.create_embeddings(retrain=retrain)
    print("✅ Embeddings created successfully")
    
    # Step 3: Optional - Train LSTM Model (not required)
    print("\n" + "="*70)
    print("STEP 3: Optional - Train LSTM Model")
    print("-" * 70)
    print("Note: LSTM training is optional. The system works with similarity search alone.")
    
    train_lstm = input("Do you want to train LSTM model? (y/n, default=n): ").strip().lower() == 'y'
    
    if train_lstm:
        epochs_input = input("Number of training epochs (default=20): ").strip()
        epochs = int(epochs_input) if epochs_input.isdigit() else 20
        
        batch_size_input = input("Batch size (default=32): ").strip()
        batch_size = int(batch_size_input) if batch_size_input.isdigit() else 32
        
        print(f"\nTraining LSTM model with {epochs} epochs, batch size {batch_size}...")
        print("This may take several minutes...")
        print("="*70)
        
        try:
            history = qa.train_lstm(epochs=epochs, batch_size=batch_size)
            print("\n✅ LSTM model training completed")
        except Exception as e:
            print(f"\n⚠️ Warning: LSTM training failed: {e}")
            print("You can still use the system with similarity search only.")
    else:
        print("\n⏭️ Skipping LSTM training. Using similarity search only.")
    
    # Summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE!")
    print("="*70)
    print(f"✅ Chunks: {len(qa.chunks)}")
    print(f"✅ Embeddings: {qa.chunk_embeddings.shape if qa.chunk_embeddings is not None else 'None'}")
    if qa.is_trained:
        print(f"✅ LSTM Model: Trained (optional)")
    else:
        print(f"✅ Using: Similarity Search (no training needed)")
    print("\nYou can now run 'streamlit run app_new.py' to start querying!")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

