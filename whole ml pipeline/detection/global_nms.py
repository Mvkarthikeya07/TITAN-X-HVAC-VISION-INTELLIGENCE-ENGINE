"""
Global NMS Module for InTakeoff Pipeline.

This module resolves duplicate bounding box detections that occur at the 
overlapping boundaries of adjacent tiles. It merges all detections from a page 
and applies a global Non-Maximum Suppression (NMS).
"""

import cv2
import numpy as np
from typing import List

from constants import DetectedSymbol
from settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

class GlobalNMS:
    """
    Applies Non-Maximum Suppression across an entire page's detected symbols.
    """
    
    def __init__(self, iou_threshold: float = settings.YOLO_IOU_THRESHOLD):
        """
        Args:
            iou_threshold (float): The Intersection over Union threshold for NMS.
        """
        self.iou_threshold = iou_threshold

    def apply(self, symbols: List[DetectedSymbol]) -> List[DetectedSymbol]:
        """
        Filters duplicate detections using NMS.
        Since NMS should be applied per-class, we separate them by class_id.
        
        Args:
            symbols (List[DetectedSymbol]): All detected symbols on a page.
            
        Returns:
            List[DetectedSymbol]: The filtered list of symbols.
        """
        if not symbols:
            return []

        logger.info(f"Applying Global NMS to {len(symbols)} symbols with IoU {self.iou_threshold}")
        
        # Group by class_id
        class_map = {}
        for idx, sym in enumerate(symbols):
            if sym.class_id not in class_map:
                class_map[sym.class_id] = []
            class_map[sym.class_id].append((idx, sym))
            
        final_indices = []
        
        for class_id, items in class_map.items():
            bboxes = []
            scores = []
            indices_map = []
            
            for idx, sym in items:
                # cv2.dnn.NMSBoxes expects [x, y, w, h]
                x = sym.bbox.x1
                y = sym.bbox.y1
                w = sym.bbox.x2 - sym.bbox.x1
                h = sym.bbox.y2 - sym.bbox.y1
                
                bboxes.append([x, y, w, h])
                scores.append(sym.confidence)
                indices_map.append(idx)
                
            # Apply NMS
            # score_threshold=0.0 because filtering was already done by YOLOv8Detector
            kept_idx = cv2.dnn.NMSBoxes(
                bboxes=bboxes,
                scores=scores,
                score_threshold=0.0,
                nms_threshold=self.iou_threshold
            )
            
            if len(kept_idx) > 0:
                kept_idx = np.array(kept_idx).flatten()
                for i in kept_idx:
                    final_indices.append(indices_map[i])
                    
        # Reconstruct the list preserving original order
        final_indices_set = set(final_indices)
        filtered_symbols = [sym for idx, sym in enumerate(symbols) if idx in final_indices_set]
        
        logger.info(f"Global NMS retained {len(filtered_symbols)} symbols out of {len(symbols)}")
        return filtered_symbols
