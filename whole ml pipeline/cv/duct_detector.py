import cv2
import numpy as np
from typing import List, Dict, Any

from utils.logger import get_logger
from cv.geometry import ScaleConverter

logger = get_logger(__name__)

class DuctDetector:
    """Detector for finding rectangular duct runs using morphological operations."""
    
    def __init__(self, scale_converter: ScaleConverter):
        """Initialize the DuctDetector.
        
        Args:
            scale_converter (ScaleConverter): Converter for real-world scaling.
        """
        self.scale_converter = scale_converter
        
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect duct runs in an image.
        
        Args:
            image (np.ndarray): Input image.
            
        Returns:
            List[Dict[str, Any]]: Detected duct runs with contours and measurements.
        """
        try:
            logger.info("Starting duct detection.")
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
                
            # Basic thresholding to isolate elements
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            
            # Morphological operations to group rectangular shapes typical of ducts
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            results = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 100:  # Minimum area filter for noise removal
                    continue
                    
                rect = cv2.minAreaRect(cnt)
                width, height = rect[1]
                
                if min(width, height) == 0:
                    continue
                    
                # Evaluate aspect ratio for typical duct structures
                aspect_ratio = max(width, height) / min(width, height)
                if 1.5 < aspect_ratio < 15.0:
                    real_width = self.scale_converter.pixels_to_units(width)
                    real_height = self.scale_converter.pixels_to_units(height)
                    
                    results.append({
                        "contour": cnt,
                        "bounding_rect": rect,
                        "real_width": real_width,
                        "real_height": real_height,
                        "area_pixels": area
                    })
                    
            logger.info(f"Detected {len(results)} potential duct runs.")
            return results
        except Exception as e:
            logger.error(f"Error in duct detection: {e}")
            raise
