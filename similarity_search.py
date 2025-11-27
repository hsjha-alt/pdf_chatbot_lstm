"""
Similarity Search Module using LSTM embeddings
Performs similarity search on chunks using LSTM-encoded embeddings
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity


class SimilaritySearch:
    """Similarity search using LSTM embeddings"""
    
    def __init__(self, chunks: List[Dict], chunk_embeddings: np.ndarray):
        """
        Initialize similarity search
        
        Args:
            chunks: List of chunk dictionaries
            chunk_embeddings: Pre-computed embeddings for chunks
        """
        self.chunks = chunks
        self.chunk_embeddings = chunk_embeddings
        print(f"Initialized similarity search with {len(chunks)} chunks")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search for most similar chunks
        
        Args:
            query_embedding: Embedding of the query/question
            top_k: Number of top results to return
            
        Returns:
            List of tuples (chunk_dict, similarity_score)
        """
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return chunks with similarity scores
        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(similarities[idx])))
        
        return results
    
    def search_with_threshold(self, query_embedding: np.ndarray, 
                             top_k: int = 5, 
                             threshold: float = 0.3) -> List[Tuple[Dict, float]]:
        """
        Search with minimum similarity threshold
        
        Args:
            query_embedding: Embedding of the query/question
            top_k: Number of top results to return
            threshold: Minimum similarity score threshold
            
        Returns:
            List of tuples (chunk_dict, similarity_score) above threshold
        """
        results = self.search(query_embedding, top_k * 2)  # Get more candidates
        filtered = [(chunk, score) for chunk, score in results if score >= threshold]
        return filtered[:top_k]  # Return top k after filtering

