"""
ResNet18 Embedder Module for InTakeoff Pipeline.

This module generates 512-dimensional feature embeddings for detected HVAC symbols
and legend entries using a ResNet18 model exported to ONNX. 
Supports batched inference and L2 normalization.
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List

from utils.logger import get_logger
from utils.exceptions import ModelInferenceError
from settings import settings
from config import EMBEDDING_DIM

logger = get_logger(__name__)

class ResNetEmbedder:
    """
    ONNX Runtime based ResNet18 Embedder.
    Generates L2-normalized 512-dim vectors for input image crops.
    """
    
    def __init__(
        self, 
        model_path: str,
        input_size: int = 224,
        use_cuda: bool = settings.USE_CUDA
    ):
        self.model_path = model_path
        self.input_size = input_size
        self.session = self._init_session(use_cuda)
        self.input_name = self.session.get_inputs()[0].name
        
    def _init_session(self, use_cuda: bool) -> ort.InferenceSession:
        providers = []
        if use_cuda:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        try:
            logger.info(f"Loading ResNet18 ONNX model from {self.model_path}")
            return ort.InferenceSession(self.model_path, providers=providers)
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise ModelInferenceError("Could not initialize ResNet18 ONNX session") from e

    def _preprocess(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Resizes, normalizes, and batches a list of image crops.
        Uses standard ImageNet normalization.
        """
        batch = []
        
        # ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        for img in images:
            if img is None or img.size == 0:
                # Fallback to zero tensor if invalid image crop
                tensor = np.zeros((3, self.input_size, self.input_size), dtype=np.float32)
            else:
                resized = cv2.resize(img, (self.input_size, self.input_size))
                # Ensure RGB
                if len(resized.shape) == 2:
                    resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
                elif resized.shape[2] == 4:
                    resized = cv2.cvtColor(resized, cv2.COLOR_RGBA2RGB)
                    
                resized = resized.astype(np.float32) / 255.0
                resized = (resized - mean) / std
                
                # HWC to CHW
                tensor = resized.transpose(2, 0, 1)
                
            batch.append(tensor)
            
        return np.stack(batch)

    def get_embeddings(self, image_crops: List[np.ndarray]) -> np.ndarray:
        """
        Runs batched inference to get L2 normalized embeddings.
        
        Args:
            image_crops (List[np.ndarray]): List of image crops (H, W, C).
            
        Returns:
            np.ndarray: Array of shape (N, 512).
        """
        if not image_crops:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
            
        logger.info(f"Generating embeddings for batch of {len(image_crops)} crops.")
        
        try:
            tensor = self._preprocess(image_crops)
            outputs = self.session.run(None, {self.input_name: tensor})[0]
            
            # outputs shape: (N, 512)
            # L2 Normalize
            norms = np.linalg.norm(outputs, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0] = 1e-9
            normalized = outputs / norms
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error during ResNet embedding: {e}")
            raise ModelInferenceError("ResNet embedding failed") from e
