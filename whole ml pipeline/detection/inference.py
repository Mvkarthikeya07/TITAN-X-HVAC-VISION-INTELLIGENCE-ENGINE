"""
Inference wrapper for YOLOv8 object detection on tiles.
"""
from typing import List, Any
from utils.logger import get_logger
from detection.yolo_detector import YOLOv8Detector

logger = get_logger(__name__)

def run_inference(tiles: List[Any], model_path: str) -> List[Any]:
    """
    Run YOLOv8 inference on a list of image tiles.

    Args:
        tiles: A list of Tile objects to process.
        model_path: Path to the YOLOv8 model weights.

    Returns:
        A list of detection results corresponding to each tile.
    """
    logger.info(f"Loading YOLOv8Detector from {model_path}")
    try:
        detector = YOLOv8Detector(model_path)
    except Exception as e:
        logger.error(f"Failed to load YOLOv8Detector from {model_path}: {e}")
        raise

    results = []
    logger.info(f"Running inference on {len(tiles)} tiles.")
    for i, tile in enumerate(tiles):
        try:
            if not hasattr(tile, 'image'):
                logger.warning(f"Tile at index {i} does not have an 'image' attribute. Skipping.")
                results.append([])
                continue
            
            predictions = detector.detect(tile.image)
            results.append(predictions)
        except Exception as e:
            logger.error(f"Error during inference on tile index {i}: {e}")
            results.append([])
            
    logger.info("Inference completed.")
    return results
