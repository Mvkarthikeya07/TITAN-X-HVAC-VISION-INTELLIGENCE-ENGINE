"""
Preprocessing utilities for object detection.
"""
import cv2
import numpy as np
from typing import Tuple, List
from utils.logger import get_logger

logger = get_logger(__name__)

def letterbox_image(image: np.ndarray, target_size: Tuple[int, int], color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image to a target size with padding to preserve aspect ratio.

    Args:
        image: Original image as a numpy array.
        target_size: Desired output size (width, height).
        color: Padding color.

    Returns:
        Tuple containing the padded image, the scaling ratio used, and the (dw, dh) padding.
    """
    try:
        shape = image.shape[:2]
        new_w, new_h = target_size
        
        r = min(new_w / shape[1], new_h / shape[0])
        
        pad_w = int(round(shape[1] * r))
        pad_h = int(round(shape[0] * r))
        dw = (new_w - pad_w) / 2
        dh = (new_h - pad_h) / 2
        
        if shape[::-1] != (pad_w, pad_h):
            image = cv2.resize(image, (pad_w, pad_h), interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return image, r, (left, top)
    except Exception as e:
        logger.error(f"Failed to letterbox image: {e}")
        raise

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image pixel values to [0, 1].

    Args:
        image: Input image as numpy array.

    Returns:
        Normalized image.
    """
    try:
        norm_image = image.astype(np.float32) / 255.0
        return norm_image
    except Exception as e:
        logger.error(f"Failed to normalize image: {e}")
        raise

def prepare_batch(images: List[np.ndarray], target_size: Tuple[int, int]) -> np.ndarray:
    """
    Prepare a batch of images for neural network input.

    Args:
        images: List of original images.
        target_size: Target (width, height) for each image.

    Returns:
        Batch array of shape (N, C, H, W).
    """
    try:
        batch = []
        for img in images:
            letterboxed, _, _ = letterbox_image(img, target_size)
            normalized = normalize_image(letterboxed)
            transposed = normalized.transpose((2, 0, 1))
            batch.append(transposed)
            
        return np.array(batch, dtype=np.float32)
    except Exception as e:
        logger.error(f"Failed to prepare batch: {e}")
        raise
