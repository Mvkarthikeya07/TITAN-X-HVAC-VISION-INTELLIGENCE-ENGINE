"""
Utility functions for embedding extraction and processing.
"""
import numpy as np
from typing import List, Tuple, Any
from utils.logger import get_logger
from numpy.linalg import norm

logger = get_logger(__name__)

def crop_symbol_from_image(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop a symbol from a larger image using a bounding box.

    Args:
        image: Source image as a numpy array.
        bbox: Bounding box in format (x1, y1, x2, y2).

    Returns:
        Cropped image array.
    """
    try:
        x1, y1, x2, y2 = map(int, bbox)
        
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"Invalid crop dimensions: w={x2-x1}, h={y2-y1}")
            
        cropped = image[y1:y2, x1:x2]
        return cropped
    except Exception as e:
        logger.error(f"Failed to crop symbol from image: {e}")
        raise

def batch_generate_embeddings(images: List[np.ndarray], model: Any) -> np.ndarray:
    """
    Generate embeddings for a batch of images.

    Args:
        images: List of cropped symbol images.
        model: Embedding model with an encode() or predict() method.

    Returns:
        Numpy array of embeddings of shape (N, embedding_dim).
    """
    try:
        if not images:
            logger.warning("Empty image list provided for embedding generation.")
            return np.array([])
            
        logger.info(f"Generating embeddings for {len(images)} images.")
        
        if hasattr(model, 'get_embeddings'):
            embeddings = model.get_embeddings(images)
        elif hasattr(model, 'encode'):
            embeddings = model.encode(images)
        elif hasattr(model, 'predict'):
            embeddings = model.predict(np.array(images))
        else:
            raise AttributeError("Model must have an 'encode', 'predict', or 'get_embeddings' method.")
            
        return np.array(embeddings)
    except Exception as e:
        logger.error(f"Failed to generate batch embeddings: {e}")
        raise

def compute_pairwise_similarity(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between two sets of embeddings.

    Args:
        emb1: Embeddings array of shape (N, D).
        emb2: Embeddings array of shape (M, D).

    Returns:
        Similarity matrix of shape (N, M).
    """
    try:
        if len(emb1.shape) == 1:
            emb1 = emb1.reshape(1, -1)
        if len(emb2.shape) == 1:
            emb2 = emb2.reshape(1, -1)
            
        norm1 = norm(emb1, axis=1, keepdims=True)
        norm2 = norm(emb2, axis=1, keepdims=True)
        
        norm1[norm1 == 0] = 1e-10
        norm2[norm2 == 0] = 1e-10
        
        similarity = np.dot(emb1, emb2.T) / np.dot(norm1, norm2.T)
        return similarity
    except Exception as e:
        logger.error(f"Failed to compute pairwise similarity: {e}")
        raise
