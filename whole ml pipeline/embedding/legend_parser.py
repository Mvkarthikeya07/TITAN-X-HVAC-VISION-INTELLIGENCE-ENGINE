"""
Legend Parser Module for InTakeoff Pipeline.

Parses a legend page to extract bounding boxes for symbols and their associated text labels.
Uses OpenCV heuristics and dependency injection for OCR and Embedder.
"""

import cv2
import numpy as np
from typing import List, Any
import uuid

from constants import LegendEntry, BoundingBox
from embedding.resnet_embedder import ResNetEmbedder
from utils.logger import get_logger

logger = get_logger(__name__)

class LegendParser:
    """
    Parses an image of a legend page to extract Symbol-Text pairs.
    """
    
    def __init__(self, embedder: ResNetEmbedder, ocr_engine: Any = None):
        """
        Args:
            embedder (ResNetEmbedder): Engine to generate embeddings for extracted symbols.
            ocr_engine (Any): Engine to extract text. If None, it will be mocked.
        """
        self.embedder = embedder
        self.ocr_engine = ocr_engine
        
    def parse(self, image: np.ndarray) -> List[LegendEntry]:
        """
        Extracts legend entries from the page image.
        Uses a heuristic OpenCV contour approach to find symbols, and assumes 
        text is immediately to the right of the symbol.
        """
        logger.info("Parsing legend image for symbols and text...")
        
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        entries = []
        crops = []
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Filter out very small or very large contours
            if w < 20 or h < 20 or w > 300 or h > 300:
                continue
                
            # Crop symbol
            crop = image[y:y+h, x:x+w]
            crops.append(crop)
            
            # Heuristic: Text is to the right
            text_roi = image[max(0, y-10):min(image.shape[0], y+h+10), x+w:min(image.shape[1], x+w+400)]
            
            label = "Unknown Symbol"
            if self.ocr_engine is not None and text_roi.size > 0:
                # OCR extraction (mocked interface expectation)
                label = self.ocr_engine.extract_text(text_roi)
                
            entry = LegendEntry(
                symbol_id=str(uuid.uuid4()),
                label=label.strip() if label else "Unknown Symbol",
                bbox=BoundingBox(x1=x, y1=y, x2=x+w, y2=y+h)
            )
            entries.append(entry)
            
        # Batch generate embeddings
        if crops:
            embeddings = self.embedder.get_embeddings(crops)
            for i, entry in enumerate(entries):
                entry.embedding = embeddings[i].tolist()
                
        logger.info(f"Successfully extracted {len(entries)} legend entries.")
        return entries
