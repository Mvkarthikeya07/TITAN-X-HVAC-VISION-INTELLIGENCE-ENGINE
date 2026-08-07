"""
Cosine Similarity Module for InTakeoff Pipeline.

Provides highly optimized matrix operations to compute similarity between 
detected symbol embeddings and a database of known legend embeddings.
"""

import numpy as np
from typing import List, Tuple
from constants import LegendEntry

class CosineSimilarityEngine:
    """
    Computes cosine similarity between query vectors and a gallery of vectors.
    Assumes vectors are already L2 normalized.
    """
    
    def __init__(self, legend_entries: List[LegendEntry]):
        """
        Args:
            legend_entries (List[LegendEntry]): The parsed legend database.
        """
        self.legend_entries = legend_entries
        self.valid_entries = [e for e in legend_entries if e.embedding is not None]
        
        if self.valid_entries:
            # Shape: (num_legends, 512)
            self.gallery_matrix = np.vstack([e.embedding for e in self.valid_entries])
        else:
            self.gallery_matrix = np.empty((0, 0))

    def match(self, query_embedding: np.ndarray, top_k: int = 1) -> List[Tuple[LegendEntry, float]]:
        """
        Matches a single query embedding against the legend database.
        
        Args:
            query_embedding (np.ndarray): Shape (512,) - already L2 normalized.
            top_k (int): Number of top matches to return.
            
        Returns:
            List[Tuple[LegendEntry, float]]: Top K matches with their similarity score (0-1).
        """
        if self.gallery_matrix.size == 0 or query_embedding is None:
            return []
            
        # Ensure 1D
        query = query_embedding.flatten()
        
        # Dot product (since both are L2 normalized, this is cosine similarity)
        similarities = np.dot(self.gallery_matrix, query)
        
        # Get top K indices
        # If top_k > len, limit it
        k = min(top_k, len(self.valid_entries))
        # Use argpartition for O(N) top-k instead of O(N log N) sort
        top_k_indices = np.argpartition(similarities, -k)[-k:]
        
        # Sort the top K indices by similarity score descending
        top_k_indices = top_k_indices[np.argsort(-similarities[top_k_indices])]
        
        results = []
        for idx in top_k_indices:
            score = float(similarities[idx])
            results.append((self.valid_entries[idx], score))
            
        return results
