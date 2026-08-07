import cv2
import numpy as np
from typing import List, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)

def filter_contours_by_area(
    contours: List[np.ndarray], 
    min_area: float = 0.0, 
    max_area: float = float('inf')
) -> List[np.ndarray]:
    """Filter contours based on their area.
    
    Args:
        contours (List[np.ndarray]): List of contours.
        min_area (float): Minimum allowed area.
        max_area (float): Maximum allowed area.
        
    Returns:
        List[np.ndarray]: Filtered list of contours.
    """
    try:
        filtered = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                filtered.append(cnt)
        return filtered
    except Exception as e:
        logger.error(f"Error filtering contours by area: {e}")
        raise

def get_contour_centroid(contour: np.ndarray) -> Tuple[int, int]:
    """Calculate the centroid of a contour.
    
    Args:
        contour (np.ndarray): The contour.
        
    Returns:
        Tuple[int, int]: (x, y) coordinates of the centroid.
    """
    try:
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return (0, 0)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)
    except Exception as e:
        logger.error(f"Error calculating contour centroid: {e}")
        raise

def merge_nearby_contours(contours: List[np.ndarray], distance_threshold: float) -> List[np.ndarray]:
    """Merge contours that are close to each other.
    
    Args:
        contours (List[np.ndarray]): List of contours to merge.
        distance_threshold (float): Maximum distance between centroids/rects to merge.
        
    Returns:
        List[np.ndarray]: List of merged contours.
    """
    try:
        if not contours:
            return []
            
        merged_contours = []
        bboxes = [cv2.boundingRect(cnt) for cnt in contours]
        
        class DisjointSet:
            def __init__(self, n: int):
                self.parent = list(range(n))
            def find(self, i: int) -> int:
                if self.parent[i] == i:
                    return i
                self.parent[i] = self.find(self.parent[i])
                return self.parent[i]
            def union(self, i: int, j: int) -> None:
                root_i = self.find(i)
                root_j = self.find(j)
                if root_i != root_j:
                    self.parent[root_i] = root_j
                    
        n = len(contours)
        ds = DisjointSet(n)
        
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1, w1, h1 = bboxes[i]
                x2, y2, w2, h2 = bboxes[j]
                
                c1 = (x1 + w1/2, y1 + h1/2)
                c2 = (x2 + w2/2, y2 + h2/2)
                
                dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                if dist <= distance_threshold:
                    ds.union(i, j)
                    
        groups = {}
        for i in range(n):
            root = ds.find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(contours[i])
            
        for root, group in groups.items():
            if len(group) == 1:
                merged_contours.append(group[0])
            else:
                points = np.vstack(group)
                hull = cv2.convexHull(points)
                merged_contours.append(hull)
                
        return merged_contours
    except Exception as e:
        logger.error(f"Error merging nearby contours: {e}")
        raise

def simplify_contour(contour: np.ndarray, epsilon_factor: float = 0.01) -> np.ndarray:
    """Simplify a contour using the Douglas-Peucker algorithm.
    
    Args:
        contour (np.ndarray): The input contour.
        epsilon_factor (float): Factor to determine approximation accuracy.
        
    Returns:
        np.ndarray: The simplified contour.
    """
    try:
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return approx
    except Exception as e:
        logger.error(f"Error simplifying contour: {e}")
        raise
