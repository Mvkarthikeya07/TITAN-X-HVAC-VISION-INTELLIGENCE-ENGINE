"""
YOLOv8 Detector Module for InTakeoff Pipeline.

Implements batched ONNX Runtime inference for YOLOv8 models. Supports CUDA with
CPU fallback, letterboxing, batch processing, and output coordinate remapping.
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple

from constants import BoundingBox, DetectedSymbol
from config import YOLO_CLASSES
from settings import settings
from utils.logger import get_logger
from utils.exceptions import ModelInferenceError
from preprocessing.tile_generator import Tile

logger = get_logger(__name__)

class YOLOv8Detector:
    """
    ONNX Runtime based YOLOv8 Detector.
    Supports batched inference and coordinate mapping.
    """
    
    def __init__(
        self, 
        model_path: str,
        input_size: int = 640,
        conf_thres: float = settings.YOLO_CONFIDENCE_THRESHOLD,
        iou_thres: float = settings.YOLO_IOU_THRESHOLD,
        use_cuda: bool = settings.USE_CUDA
    ):
        self.model_path = model_path
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        self.session = self._init_session(use_cuda)
        self.input_name = self.session.get_inputs()[0].name
        
    def _init_session(self, use_cuda: bool) -> ort.InferenceSession:
        """Initializes the ONNX inference session."""
        providers = []
        if use_cuda:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        try:
            logger.info(f"Loading YOLOv8 ONNX model from {self.model_path}")
            return ort.InferenceSession(self.model_path, providers=providers)
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise ModelInferenceError("Could not initialize YOLOv8 ONNX session") from e

    def _letterbox(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Resize image and pad to square for YOLO inference.
        Returns:
            padded_img: the letterboxed image
            ratio: resize ratio
            dw, dh: padding amounts
        """
        shape = img.shape[:2]
        r = min(self.input_size / shape[0], self.input_size / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        
        dw, dh = self.input_size - new_unpad[0], self.input_size - new_unpad[1]
        dw /= 2
        dh /= 2
        
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, r, (dw, dh)

    def _preprocess(self, tiles: List[Tile]) -> Tuple[np.ndarray, List[float], List[Tuple[float, float]]]:
        """Prepares a batch of tiles for inference."""
        batch_images = []
        ratios = []
        dwdhs = []
        
        for tile in tiles:
            img, r, dwdh = self._letterbox(tile.image)
            # HWC to CHW, keep RGB ordering (YOLOv8 expects RGB)
            img = img.transpose((2, 0, 1))
            img = np.ascontiguousarray(img)
            batch_images.append(img)
            ratios.append(r)
            dwdhs.append(dwdh)
            
        # NCHW
        batch_tensor = np.stack(batch_images).astype(np.float32) / 255.0
        return batch_tensor, ratios, dwdhs

    def _postprocess(self, outputs: np.ndarray, ratios: List[float], dwdhs: List[Tuple[float, float]], tiles: List[Tile]) -> List[DetectedSymbol]:
        """
        Processes YOLOv8 batched outputs and remaps coordinates.
        Outputs shape: (batch_size, num_classes + 4, num_anchors)
        """
        all_symbols = []
        batch_size = outputs.shape[0]
        
        for b in range(batch_size):
            preds = outputs[b].transpose(1, 0) # (num_anchors, 4 + num_classes)
            
            # Extract boxes and scores
            boxes = preds[:, :4]
            scores = preds[:, 4:]
            
            # Max score per anchor
            class_ids = np.argmax(scores, axis=1)
            confidences = np.max(scores, axis=1)
            
            # Confidence filtering
            mask = confidences > self.conf_thres
            boxes = boxes[mask]
            class_ids = class_ids[mask]
            confidences = confidences[mask]
            
            if len(boxes) == 0:
                continue
                
            # YOLOv8 format is center_x, center_y, width, height
            # Convert to x1, y1, x2, y2
            x1 = boxes[:, 0] - boxes[:, 2] / 2
            y1 = boxes[:, 1] - boxes[:, 3] / 2
            x2 = boxes[:, 0] + boxes[:, 2] / 2
            y2 = boxes[:, 1] + boxes[:, 3] / 2
            
            # Apply NMS per image (OpenCV NMS)
            indices = cv2.dnn.NMSBoxes(
                bboxes=np.column_stack((x1, y1, x2 - x1, y2 - y1)).tolist(),
                scores=confidences.tolist(),
                score_threshold=self.conf_thres,
                nms_threshold=self.iou_thres
            )
            
            if len(indices) == 0:
                continue
                
            r = ratios[b]
            dw, dh = dwdhs[b]
            tile = tiles[b]
            
            # Ensure indices is a numpy array (cv2.dnn.NMSBoxes may return tuple)
            indices = np.array(indices).flatten()
            for i in indices:
                # Revert letterbox padding
                bx1 = (x1[i] - dw) / r
                by1 = (y1[i] - dh) / r
                bx2 = (x2[i] - dw) / r
                by2 = (y2[i] - dh) / r
                
                # Clip to tile boundaries
                bx1 = max(0, min(bx1, tile.width))
                by1 = max(0, min(by1, tile.height))
                bx2 = max(0, min(bx2, tile.width))
                by2 = max(0, min(by2, tile.height))
                
                if bx2 <= bx1 or by2 <= by1:
                    continue
                
                local_bbox = BoundingBox(x1=bx1, y1=by1, x2=bx2, y2=by2)
                global_bbox = tile.local_to_global_bbox(local_bbox)
                
                cid = int(class_ids[i])
                cname = YOLO_CLASSES.get(cid, "Unknown")
                
                symbol = DetectedSymbol(
                    class_id=cid,
                    class_name=cname,
                    confidence=float(confidences[i]),
                    bbox=global_bbox,
                    tile_id=tile.tile_id
                )
                all_symbols.append(symbol)
                
        return all_symbols

    def detect(self, tiles: List[Tile]) -> List[DetectedSymbol]:
        """
        Runs batched inference on a list of Tiles.
        
        Args:
            tiles (List[Tile]): List of tile images to process.
            
        Returns:
            List[DetectedSymbol]: Detected symbols with global coordinates.
        """
        if not tiles:
            return []
            
        logger.info(f"Running YOLOv8 inference on batch of {len(tiles)} tiles.")
        try:
            tensor, ratios, dwdhs = self._preprocess(tiles)
            # ONNX Run
            outputs = self.session.run(None, {self.input_name: tensor})[0]
            # Postprocess
            return self._postprocess(outputs, ratios, dwdhs, tiles)
        except Exception as e:
            logger.error(f"Error during YOLOv8 inference: {e}")
            raise ModelInferenceError("YOLOv8 inference failed") from e
