"""
Embedding Module using Sentence Transformers
Uses pre-trained models for much better accuracy
"""

import os
import numpy as np
import pickle
from typing import List, Dict, Optional


class SentenceTransformerEmbedder:
    """Creates embeddings using pre-trained sentence transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder with a pre-trained model
        
        Args:
            model_name: Name of the sentence transformer model
                       Options:
                       - "all-MiniLM-L6-v2" (fast, 384 dim, good quality)
                       - "all-mpnet-base-v2" (slower, 768 dim, better quality)
                       - "paraphrase-multilingual-MiniLM-L12-v2" (multilingual)
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        
    def load_model(self):
        """Load the sentence transformer model"""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"Loading sentence transformer model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name, device='cpu')
                
                # Get embedding dimension
                test_embedding = self.model.encode(["test"])
                self.embedding_dim = test_embedding.shape[1]
                print(f"✅ Model loaded. Embedding dimension: {self.embedding_dim}")
            except ImportError:
                print("❌ sentence-transformers not installed.")
                print("Install with: pip install sentence-transformers")
                raise
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                raise
    
    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Create embeddings for texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        if self.model is None:
            self.load_model()
        
        # Encode texts
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for better cosine similarity
        )
        
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """Create embedding for a single text"""
        return self.encode([text])[0]
    
    def save(self, directory: str):
        """Save model configuration"""
        os.makedirs(directory, exist_ok=True)
        
        config = {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim
        }
        config_path = os.path.join(directory, 'config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
        
        print(f"Saved embedding config to {directory}")
    
    def load(self, directory: str):
        """Load model configuration"""
        config_path = os.path.join(directory, 'config.pkl')
        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
            self.model_name = config.get('model_name', self.model_name)
            self.embedding_dim = config.get('embedding_dim')
        
        # Load the model
        self.load_model()


class BM25Embedder:
    """BM25 algorithm for better keyword-based retrieval"""
    
    def __init__(self):
        """Initialize BM25 embedder"""
        self.vocab = {}
        self.idf = {}
        self.doc_freqs = []
        self.avg_doc_length = 0
        self.k1 = 1.5  # BM25 parameter
        self.b = 0.75   # BM25 parameter
        
    def fit(self, texts: List[str]):
        """
        Fit BM25 on texts
        
        Args:
            texts: List of text strings
        """
        from collections import Counter
        import re
        
        # Tokenize and build vocabulary
        tokenized_docs = []
        doc_lengths = []
        
        for text in texts:
            # Simple tokenization
            tokens = re.findall(r'\b\w+\b', text.lower())
            tokenized_docs.append(tokens)
            doc_lengths.append(len(tokens))
            
            # Count term frequencies
            term_freq = Counter(tokens)
            self.doc_freqs.append(term_freq)
        
        # Calculate average document length
        self.avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
        
        # Build vocabulary and IDF
        all_terms = set()
        for term_freq in self.doc_freqs:
            all_terms.update(term_freq.keys())
        
        # Calculate IDF
        N = len(texts)
        for term in all_terms:
            df = sum(1 for term_freq in self.doc_freqs if term in term_freq)
            self.idf[term] = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
    
    def encode_query(self, query: str) -> Dict[str, float]:
        """
        Encode query for BM25 search
        
        Args:
            query: Query string
            
        Returns:
            Dictionary of term scores
        """
        import re
        tokens = re.findall(r'\b\w+\b', query.lower())
        query_terms = {}
        for token in tokens:
            if token in self.idf:
                query_terms[token] = self.idf[token]
        return query_terms
    
    def score(self, query: str, doc_index: int) -> float:
        """
        Calculate BM25 score for a document
        
        Args:
            query: Query string
            doc_index: Document index
            
        Returns:
            BM25 score
        """
        if doc_index >= len(self.doc_freqs):
            return 0.0
        
        query_terms = self.encode_query(query)
        doc_freq = self.doc_freqs[doc_index]
        doc_length = sum(doc_freq.values())
        
        score = 0.0
        for term, idf in query_terms.items():
            if term in doc_freq:
                tf = doc_freq[term]
                # BM25 formula
                numerator = idf * tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                score += numerator / denominator
        
        return score


# For backward compatibility
TensorFlowEmbedder = SentenceTransformerEmbedder
