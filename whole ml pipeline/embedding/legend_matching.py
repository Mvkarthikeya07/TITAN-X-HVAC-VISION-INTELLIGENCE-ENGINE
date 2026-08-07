"""
Legend Matching Module for InTakeoff Pipeline.

Orchestrates the matching of detected YOLO symbols against the parsed legend database
using the Cosine Similarity engine.
"""

import numpy as np
from typing import List
from constants import DetectedSymbol, LegendEntry
from embedding.cosine_similarity import CosineSimilarityEngine
from utils.logger import get_logger

logger = get_logger(__name__)

class LegendMatcher:
    """
    Matches detected symbols to legend entries to refine or validate classifications.
    """
    
    def __init__(self, legend_entries: List[LegendEntry], similarity_threshold: float = 0.85):
        self.engine = CosineSimilarityEngine(legend_entries)
        self.similarity_threshold = similarity_threshold
        
    def match_symbols(self, symbols: List[DetectedSymbol], embeddings: np.ndarray) -> List[DetectedSymbol]:
        """
        Updates detected symbols with matches from the legend.
        
        Args:
            symbols (List[DetectedSymbol]): YOLO detections.
            embeddings (np.ndarray): Corresponding embeddings for the symbols (N, 512).
            
        Returns:
            List[DetectedSymbol]: Symbols updated with legend matching data.
        """
        if len(symbols) != embeddings.shape[0]:
            logger.error("Mismatch between number of symbols and embeddings.")
            return symbols
            
        for i, sym in enumerate(symbols):
            matches = self.engine.match(embeddings[i], top_k=1)
            if matches:
                best_match, score = matches[0]
                if score >= self.similarity_threshold:
                    # Store the matched score and keep class_name unchanged
                    sym.cosine_score = score
                    logger.debug(f'Matched {sym.class_name} to legend {best_match.label} with score {score:.3f}')
                    
        return symbols
