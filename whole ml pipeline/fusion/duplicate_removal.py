import math
from typing import List

from utils.logger import get_logger
from constants import DetectedSymbol, BoundingBox

logger = get_logger(__name__)


class DuplicateRemover:
    """
    Removes exact and near-duplicate detections based on spatial proximity
    and class matching. Useful for post-processing multiple overlapping 
    detection outputs.
    """

    def __init__(self, distance_threshold: float = 5.0):
        """
        Initialize the DuplicateRemover.

        Args:
            distance_threshold (float): Maximum Euclidean distance between centers 
                                        to be considered near-duplicates.
        """
        self.distance_threshold = distance_threshold
        logger.info(f"Initialized DuplicateRemover with distance threshold {distance_threshold}")

    def _get_center(self, bbox: BoundingBox) -> tuple:
        """Calculates the center point of a bounding box."""
        x_center = (bbox.x1 + bbox.x2) / 2.0
        y_center = (bbox.y1 + bbox.y2) / 2.0
        return x_center, y_center

    def _calculate_distance(self, center1: tuple, center2: tuple) -> float:
        """Calculates Euclidean distance between two points."""
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def remove_duplicates(self, detections: List[DetectedSymbol]) -> List[DetectedSymbol]:
        """
        Removes duplicates from a list of detections.

        Args:
            detections (List[DetectedSymbol]): List of detection dictionaries.

        Returns:
            List[DetectedSymbol]: Cleaned list of detections.
        """
        if not detections:
            return []

        # Sort by confidence so that we keep the most confident one among near-duplicates
        sorted_detections = sorted(detections, key=lambda x: x.confidence if x.confidence is not None else 0.0, reverse=True)
        unique_detections = []

        for current_det in sorted_detections:
            is_duplicate = False
            current_bbox = current_det.bbox
            
            if not current_bbox:
                # If no bbox, we cannot process spatial duplicates
                unique_detections.append(current_det)
                continue
                
            current_center = self._get_center(current_bbox)

            for unique_det in unique_detections:
                unique_bbox = unique_det.bbox
                if not unique_bbox:
                    continue
                
                # Check if classes match
                if current_det.class_name and current_det.class_name == unique_det.class_name:
                    
                    unique_center = self._get_center(unique_bbox)
                    distance = self._calculate_distance(current_center, unique_center)
                    
                    if distance <= self.distance_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_detections.append(current_det)

        logger.info(f"Removed duplicates: {len(detections)} -> {len(unique_detections)} detections.")
        return unique_detections
