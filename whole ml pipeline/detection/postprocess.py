"""
Postprocessing utilities for YOLO object detection.
"""
import numpy as np
from typing import List, Dict, Any
from utils.logger import get_logger

logger = get_logger(__name__)

def filter_by_confidence(predictions: np.ndarray, conf_thresh: float) -> np.ndarray:
    """
    Filter predictions below a certain confidence threshold.

    Args:
        predictions: Array of shape (N, 5 + num_classes) where 5 is (x, y, w, h, objectness).
        conf_thresh: Confidence threshold.

    Returns:
        Filtered array of predictions.
    """
    try:
        if len(predictions) == 0:
            return predictions
            
        scores = predictions[:, 4]
        mask = scores > conf_thresh
        return predictions[mask]
    except Exception as e:
        logger.error(f"Error filtering by confidence: {e}")
        raise

def apply_nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    """
    Apply Non-Maximum Suppression to bounding boxes.

    Args:
        boxes: Array of shape (N, 4) in (x1, y1, x2, y2) format.
        scores: Array of shape (N,) containing confidence scores.
        iou_thresh: IoU threshold for NMS.

    Returns:
        List of indices to keep.
    """
    try:
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]

        return keep
    except Exception as e:
        logger.error(f"Error applying NMS: {e}")
        raise

def decode_yolo_output(
    predictions: np.ndarray, 
    conf_thresh: float = 0.25, 
    iou_thresh: float = 0.45
) -> List[Dict[str, Any]]:
    """
    Decode raw YOLO outputs into structured bounding box dictionaries.

    Args:
        predictions: Raw output from YOLO model.
        conf_thresh: Confidence threshold.
        iou_thresh: NMS IoU threshold.

    Returns:
        List of dictionaries with keys 'bbox', 'confidence', and 'class_id'.
    """
    try:
        filtered = filter_by_confidence(predictions, conf_thresh)
        if len(filtered) == 0:
            return []

        x = filtered[:, 0]
        y = filtered[:, 1]
        w = filtered[:, 2]
        h = filtered[:, 3]

        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        scores = filtered[:, 4]
        
        if filtered.shape[1] > 5:
            class_probs = filtered[:, 5:]
            class_ids = np.argmax(class_probs, axis=1)
            class_max_probs = np.max(class_probs, axis=1)
            scores = scores * class_max_probs
        else:
            class_ids = np.zeros(len(filtered), dtype=int)

        keep_indices = apply_nms(boxes, scores, iou_thresh)
        
        results = []
        for i in keep_indices:
            results.append({
                'bbox': [float(boxes[i, 0]), float(boxes[i, 1]), float(boxes[i, 2]), float(boxes[i, 3])],
                'confidence': float(scores[i]),
                'class_id': int(class_ids[i])
            })
            
        return results
    except Exception as e:
        logger.error(f"Error decoding YOLO output: {e}")
        raise
