"""
Line & Duct Detector Module for InTakeoff Pipeline.

Uses OpenCV heuristics (HoughLinesP, Connected Components, Douglas-Peucker) 
to extract linear MEP elements like pipes and ducts from rasterized plans.
"""

import cv2
import numpy as np
from typing import List, Tuple

from cv.geometry import ScaleConverter
from utils.logger import get_logger

logger = get_logger(__name__)

class LineDetector:
    """
    Extracts pipes and ducts from an image.
    """
    
    def __init__(self, scale_converter: ScaleConverter):
        self.scale = scale_converter

    def detect_pipes(self, image: np.ndarray) -> Tuple[List[np.ndarray], float]:
        """
        Detects thin lines representing pipes using HoughLinesP.
        
        Args:
            image (np.ndarray): Image tile.
            
        Returns:
            lines: List of [x1, y1, x2, y2]
            total_feet: Total length in real-world feet.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Probabilistic Hough Transform
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=100, 
            minLineLength=100, 
            maxLineGap=10
        )
        
        extracted_lines = []
        total_pixels = 0.0
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                extracted_lines.append(np.array([x1, y1, x2, y2]))
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                total_pixels += length
                
        total_feet = self.scale.pixels_to_feet(total_pixels)
        logger.debug(f"Detected {len(extracted_lines)} pipe segments totalling {total_feet:.2f} feet.")
        return extracted_lines, total_feet

    def detect_ducts(self, image: np.ndarray) -> Tuple[List[np.ndarray], float]:
        """
        Detects thicker ductwork using Connected Components and the 
        Douglas-Peucker algorithm to simplify contours into polylines.
        
        Args:
            image (np.ndarray): Image tile.
            
        Returns:
            polygons: List of simplified contours (points).
            total_sqft: Total area in real-world square feet.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Thresholding (assuming dark lines on white background)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological closing to merge duct lines
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        total_pixel_area = 0.0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000: # Filter small noise
                continue
                
            # Douglas-Peucker simplification
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            polygons.append(approx)
            total_pixel_area += area
            
        total_sqft = self.scale.pixel_area_to_sqft(total_pixel_area)
        logger.debug(f"Detected {len(polygons)} duct polygons totalling {total_sqft:.2f} sqft.")
        return polygons, total_sqft
