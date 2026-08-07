from typing import List, Dict, Any, Optional

from utils.logger import get_logger
from constants import DetectedSymbol, BoundingBox

logger = get_logger(__name__)


class DetectionFilter:
    """
    Filters detections by confidence threshold, class whitelist, and spatial region.
    """

    def __init__(self, 
                 confidence_threshold: float = 0.5, 
                 class_whitelist: Optional[List[str]] = None,
                 spatial_bounds: Optional[Dict[str, float]] = None):
        """
        Initializes the DetectionFilter.

        Args:
            confidence_threshold (float): Minimum confidence required.
            class_whitelist (Optional[List[str]]): List of allowed class labels. 
                                                   If None, all classes allowed.
            spatial_bounds (Optional[Dict[str, float]]): Dictionary with keys 'x1', 'y1', 'x2', 'y2'.
                                                         Detections outside these bounds will be filtered.
        """
        self.confidence_threshold = confidence_threshold
        self.class_whitelist = class_whitelist
        self.spatial_bounds = spatial_bounds
        logger.info("Initialized DetectionFilter.")
        logger.debug(f"Confidence Threshold: {self.confidence_threshold}")
        logger.debug(f"Class Whitelist: {self.class_whitelist}")
        logger.debug(f"Spatial Bounds: {self.spatial_bounds}")

    def filter_detections(self, detections: List[DetectedSymbol]) -> List[DetectedSymbol]:
        """
        Filters a list of detections based on configured criteria.

        Args:
            detections (List[DetectedSymbol]): List of detections.

        Returns:
            List[DetectedSymbol]: Filtered list of detections.
        """
        filtered_results = []

        for det in detections:
            # 1. Check confidence
            conf = det.confidence if det.confidence is not None else 0.0
            if conf < self.confidence_threshold:
                continue

            # 2. Check class whitelist
            label = det.class_name
            if self.class_whitelist is not None and label not in self.class_whitelist:
                continue

            # 3. Check spatial bounds
            if self.spatial_bounds:
                bbox = det.bbox
                if not bbox:
                    continue
                
                # Check if box is completely outside bounds
                if (bbox.x2 < self.spatial_bounds.get("x1", 0) or
                    bbox.x1 > self.spatial_bounds.get("x2", float('inf')) or
                    bbox.y2 < self.spatial_bounds.get("y1", 0) or
                    bbox.y1 > self.spatial_bounds.get("y2", float('inf'))):
                    continue

            filtered_results.append(det)

        logger.info(f"DetectionFilter: {len(detections)} input -> {len(filtered_results)} passed.")
        return filtered_results
