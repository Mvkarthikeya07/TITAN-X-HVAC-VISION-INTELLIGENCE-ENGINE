import logging
from typing import List, Dict, Any

from utils.logger import get_logger
from constants import DetectedSymbol, BoundingBox

logger = get_logger(__name__)


class ResultMerger:
    """
    Merges detections from multiple pages/tiles into a unified list,
    deduplicating them based on an Intersection over Union (IoU) threshold.
    """

    def __init__(self, iou_threshold: float = 0.5):
        """
        Initializes the ResultMerger.

        Args:
            iou_threshold (float): Threshold above which detections are considered duplicates.
        """
        self.iou_threshold = iou_threshold
        logger.info(f"Initialized ResultMerger with IoU threshold: {iou_threshold}")

    def _calculate_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculates Intersection over Union between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = box1.x1, box1.y1, box1.x2, box1.y2
        x2_min, y2_min, x2_max, y2_max = box2.x1, box2.y1, box2.x2, box2.y2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_width = max(0.0, inter_x_max - inter_x_min)
        inter_height = max(0.0, inter_y_max - inter_y_min)
        inter_area = inter_width * inter_height

        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)

        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def merge_and_deduplicate(self, detections: List[DetectedSymbol]) -> List[DetectedSymbol]:
        """
        Merges a list of detections and removes duplicates based on IoU.
        Prioritizes detections with higher confidence scores.

        Args:
            detections (List[DetectedSymbol]): List of raw detections. 

        Returns:
            List[DetectedSymbol]: Deduplicated list of detections.
        """
        if not detections:
            return []

        # Sort detections by confidence, descending
        sorted_detections = sorted(detections, key=lambda x: x.confidence if x.confidence is not None else 0.0, reverse=True)
        merged_results = []

        for det in sorted_detections:
            is_duplicate = False
            for existing_det in merged_results:
                # Only check IoU if classes match
                if det.class_name and det.class_name == existing_det.class_name:
                    if det.bbox and existing_det.bbox:
                        iou = self._calculate_iou(det.bbox, existing_det.bbox)
                        if iou > self.iou_threshold:
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                merged_results.append(det)

        logger.info(f"Merged {len(detections)} detections down to {len(merged_results)} unique items.")
        return merged_results
