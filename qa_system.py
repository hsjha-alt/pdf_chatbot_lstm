"""
Complete QA System integrating all components
Handles PDF loading, embedding, LSTM training, and query answering
"""

import os
import numpy as np
import pickle
from typing import List, Dict, Optional
from pathlib import Path

from data_loader import PDFDataLoader
from embeddings import SentenceTransformerEmbedder, BM25Embedder
from lstm_model import LSTMModel
from similarity_search import SimilaritySearch


class QASystem:
    """Complete QA system with LSTM-based similarity search"""
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models"):
        """
        Initialize QA system
        
        Args:
            data_dir: Directory to store chunks and embeddings
            models_dir: Directory to store trained models
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        
        self.data_loader = PDFDataLoader()
        # Use sentence transformers for better accuracy
        self.embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
        # Optional: BM25 for hybrid search
        self.bm25 = None
        self.lstm_model = LSTMModel()
        self.similarity_search = None
        self.chunks = []
        self.chunk_embeddings = None
        self.is_trained = False
        self.use_hybrid_search = True  # Combine semantic + keyword search
    
    def load_pdfs(self, pdf_folder: str) -> List[Dict]:
        """
        Load PDFs from folder and create chunks
        
        Args:
            pdf_folder: Path to folder containing PDFs
            
        Returns:
            List of chunks
        """
        print(f"\n{'='*70}")
        print(f"Loading PDFs from: {pdf_folder}")
        print(f"{'='*70}")
        
        chunks = self.data_loader.load_folder(pdf_folder)
        
        # Append to existing chunks or replace
        if self.chunks:
            self.chunks.extend(chunks)
        else:
            self.chunks = chunks
        
        # Save chunks
        chunks_path = os.path.join(self.data_dir, 'chunks.pkl')
        self.data_loader.chunks = self.chunks
        self.data_loader.save_chunks(chunks_path)
        
        return chunks
    
    def load_single_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Load a single PDF file and create chunks
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of chunks
        """
        print(f"\n{'='*70}")
        print(f"Loading PDF: {os.path.basename(pdf_path)}")
        print(f"{'='*70}")
        
        # Extract text from PDF
        text = self.data_loader.load_pdf(pdf_path)
        if not text:
            print("Could not extract text from PDF")
            return []
        
        # Create chunks
        filename = os.path.basename(pdf_path)
        chunks = self.data_loader.create_chunks(text, filename)
        
        # Append to existing chunks or replace
        if self.chunks:
            self.chunks.extend(chunks)
        else:
            self.chunks = chunks
        
        # Save chunks
        chunks_path = os.path.join(self.data_dir, 'chunks.pkl')
        self.data_loader.chunks = self.chunks
        self.data_loader.save_chunks(chunks_path)
        
        print(f"Created {len(chunks)} chunks from {filename}")
        return chunks
    
    def create_embeddings(self, retrain: bool = False):
        """
        Create embeddings for chunks
        
        Args:
            retrain: Whether to retrain the embedding model
        """
        if not self.chunks:
            print("No chunks available. Please load PDFs first.")
            return
        
        print(f"\n{'='*70}")
        print("Creating embeddings...")
        print(f"{'='*70}")
        
        texts = [chunk['text'] for chunk in self.chunks]
        
        # Check if embedding model exists
        embedder_path = os.path.join(self.models_dir, 'embedder')
        
        if os.path.exists(embedder_path) and not retrain:
            print("Loading existing embedding model...")
            self.embedder.load(embedder_path)
        else:
            print("Using pre-trained sentence transformer model...")
            self.embedder.load_model()  # Load sentence transformer
            self.embedder.save(embedder_path)
        
        # Create embeddings using sentence transformers
        print("Encoding chunks with sentence transformers...")
        self.chunk_embeddings = self.embedder.encode(texts, batch_size=32, show_progress=True)
        
        # Also initialize BM25 for hybrid search
        if self.use_hybrid_search:
            print("Initializing BM25 for keyword search...")
            self.bm25 = BM25Embedder()
            self.bm25.fit(texts)
            print("✅ BM25 initialized")
        
        # Save embeddings
        embeddings_path = os.path.join(self.data_dir, 'chunk_embeddings.npy')
        np.save(embeddings_path, self.chunk_embeddings)
        print(f"Saved embeddings to {embeddings_path}")
        print(f"Embedding shape: {self.chunk_embeddings.shape}")
    
    def train_lstm(self, epochs: int = 10, batch_size: int = 32):
        """
        Train LSTM model on chunks
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        if not self.chunks or self.chunk_embeddings is None:
            print("No chunks or embeddings available. Please load PDFs and create embeddings first.")
            return
        
        print(f"\n{'='*70}")
        print("Training LSTM model...")
        print(f"{'='*70}")
        
        # Check if model exists
        lstm_path = os.path.join(self.models_dir, 'lstm')
        
        if os.path.exists(os.path.join(lstm_path, 'lstm_model.h5')):
            print("Loading existing LSTM model...")
            vocab_size = self.embedder.vocab_size
            self.lstm_model.load(lstm_path, vocab_size)
            self.is_trained = True
            history = None  # No training history when loading existing model
        else:
            print("Building new LSTM model...")
            vocab_size = self.embedder.vocab_size
            self.lstm_model.build_model(vocab_size)
            
            # Create training data
            print("Creating training data...")
            X_q, X_a, y = self.lstm_model.create_training_data(self.chunks, self.embedder)
            print(f"Training data shape: X_q={X_q.shape}, X_a={X_a.shape}, y={y.shape}")
            
            # Train model
            print("Training...")
            model_save_path = os.path.join(lstm_path, 'lstm_model.h5')
            history = self.lstm_model.train(X_q, X_a, y, epochs=epochs, batch_size=batch_size, 
                                           save_path=model_save_path)
            
            # Save model
            self.lstm_model.save(lstm_path)
            self.is_trained = True
        
        # Initialize similarity search
        self.similarity_search = SimilaritySearch(self.chunks, self.chunk_embeddings)
        print("LSTM model ready!")
        
        return history
        
        # Initialize similarity search
        self.similarity_search = SimilaritySearch(self.chunks, self.chunk_embeddings)
        print("LSTM model ready!")
    
    def query(self, question: str, top_k: int = 5, use_lstm: bool = False) -> Dict:
        """
        Answer a question using hybrid similarity search (semantic + keyword)
        
        Args:
            question: User's question
            top_k: Number of top chunks to retrieve
            use_lstm: Not used anymore - kept for compatibility
            
        Returns:
            Dictionary with answer, relevant chunks, and scores
        """
        if not self.chunks or self.chunk_embeddings is None:
            return {
                'answer': "No documents loaded. Please load PDFs first.",
                'chunks': [],
                'scores': []
            }
        
        if self.similarity_search is None:
            self.similarity_search = SimilaritySearch(self.chunks, self.chunk_embeddings)
        
        # 1. Semantic search using sentence transformers
        question_emb = self.embedder.encode([question])[0]
        semantic_results = self.similarity_search.search(question_emb, top_k=top_k * 2)
        
        # 2. Keyword search using BM25 (if available)
        if self.use_hybrid_search and self.bm25 is not None:
            bm25_scores = []
            for i in range(len(self.chunks)):
                score = self.bm25.score(question, i)
                bm25_scores.append((i, score))
            
            # Sort by BM25 score
            bm25_scores.sort(key=lambda x: x[1], reverse=True)
            bm25_results = [(self.chunks[idx], score) for idx, score in bm25_scores[:top_k * 2]]
            
            # Combine semantic and keyword results
            combined_scores = {}
            
            # Normalize and combine scores
            for chunk, score in semantic_results:
                chunk_id = id(chunk)
                combined_scores[chunk_id] = {
                    'chunk': chunk,
                    'semantic': score,
                    'keyword': 0.0
                }
            
            # Add BM25 scores
            max_bm25 = max([score for _, score in bm25_results]) if bm25_results else 1.0
            for chunk, score in bm25_results:
                chunk_id = id(chunk)
                normalized_bm25 = score / max_bm25 if max_bm25 > 0 else 0.0
                
                if chunk_id in combined_scores:
                    combined_scores[chunk_id]['keyword'] = normalized_bm25
                else:
                    combined_scores[chunk_id] = {
                        'chunk': chunk,
                        'semantic': 0.0,
                        'keyword': normalized_bm25
                    }
            
            # Calculate final combined score (weighted average)
            # Semantic: 70%, Keyword: 30%
            final_results = []
            for chunk_id, scores in combined_scores.items():
                final_score = 0.7 * scores['semantic'] + 0.3 * scores['keyword']
                final_results.append((scores['chunk'], final_score))
            
            # Sort by combined score
            final_results.sort(key=lambda x: x[1], reverse=True)
            results = final_results[:top_k]
        else:
            # Use only semantic search
            results = semantic_results[:top_k]
        
        # Format answer - get chunks and scores
        chunks = [chunk for chunk, score in results]
        scores = [score for chunk, score in results]
        
        # Sort chunks by their original position to maintain sequence within files
        chunks_with_pos = [(chunk, chunk.get('chunk_id', chunk.get('start_pos', 0))) for chunk in chunks]
        chunks_with_pos.sort(key=lambda x: (x[0].get('source_file', ''), x[1]))
        chunks = [chunk for chunk, _ in chunks_with_pos]
        
        # Re-sort scores to match sorted chunks
        chunk_to_score = {id(chunk): score for chunk, score in results}
        scores = [chunk_to_score[id(chunk)] for chunk in chunks]
        
        # Create answer from top chunks (returns actual PDF text)
        answer = self._generate_answer(question, chunks)
        
        return {
            'answer': answer,
            'chunks': chunks,
            'scores': scores,
            'sources': list(set([chunk.get('source_file', 'Unknown') for chunk in chunks]))
        }
    
    def _generate_answer(self, question: str, chunks: List[Dict]) -> str:
        """
        Generate answer from retrieved chunks - returns actual PDF text in sequence
        
        Args:
            question: User's question
            chunks: Retrieved relevant chunks (sorted by similarity and position)
            
        Returns:
            Formatted answer string with actual PDF text from documents
        """
        if not chunks:
            return "I couldn't find relevant information to answer your question."
        
        # Group chunks by source file to maintain sequence
        chunks_by_file = {}
        for chunk in chunks:
            file_name = chunk.get('source_file', 'Unknown')
            if file_name not in chunks_by_file:
                chunks_by_file[file_name] = []
            chunks_by_file[file_name].append(chunk)
        
        # Sort chunks within each file by their original position (chunk_id or start_pos)
        for file_name in chunks_by_file:
            chunks_by_file[file_name].sort(key=lambda x: x.get('chunk_id', x.get('start_pos', 0)))
        
        # Build answer with actual PDF text in sequence
        answer_parts = []
        answer_parts.append(f"**Relevant Content for: {question}**\n")
        answer_parts.append("=" * 70)
        answer_parts.append("")
        
        # Return all retrieved chunks, maintaining sequence
        for file_name, file_chunks in chunks_by_file.items():
            answer_parts.append(f"**Source: {file_name}**")
            answer_parts.append("-" * 70)
            
            for chunk in file_chunks:
                answer_parts.append(chunk['text'])
                answer_parts.append("")  # Empty line between chunks
            
            answer_parts.append("")  # Empty line between files
        
        answer = "\n".join(answer_parts)
        
        return answer
    
    def load_saved_data(self):
        """Load previously saved chunks and embeddings"""
        chunks_path = os.path.join(self.data_dir, 'chunks.pkl')
        embeddings_path = os.path.join(self.data_dir, 'chunk_embeddings.npy')
        embedder_path = os.path.join(self.models_dir, 'embedder')
        lstm_path = os.path.join(self.models_dir, 'lstm')
        
        # Load chunks
        if os.path.exists(chunks_path):
            self.chunks = self.data_loader.load_chunks(chunks_path)
            print(f"Loaded {len(self.chunks)} chunks")
        
        # Load embeddings
        if os.path.exists(embeddings_path):
            self.chunk_embeddings = np.load(embeddings_path)
            print(f"Loaded embeddings: {self.chunk_embeddings.shape}")
        
        # Load embedder
        if os.path.exists(embedder_path):
            self.embedder.load(embedder_path)
        else:
            # Load model if config doesn't exist
            self.embedder.load_model()
        
        # Load BM25 if chunks exist
        if self.chunks and self.chunk_embeddings is not None:
            texts = [chunk['text'] for chunk in self.chunks]
            if self.use_hybrid_search:
                self.bm25 = BM25Embedder()
                self.bm25.fit(texts)
        
        # Load LSTM model (optional)
        if os.path.exists(os.path.join(lstm_path, 'lstm_model.h5')):
            try:
                # For sentence transformers, we don't have vocab_size
                # Skip LSTM loading if using sentence transformers
                if hasattr(self.embedder, 'vocab_size') and self.embedder.vocab_size > 0:
                    vocab_size = self.embedder.vocab_size
                    self.lstm_model.load(lstm_path, vocab_size)
                    self.is_trained = True
            except:
                pass  # LSTM is optional
        
        # Initialize similarity search
        if self.chunks is not None and self.chunk_embeddings is not None:
            self.similarity_search = SimilaritySearch(self.chunks, self.chunk_embeddings)

