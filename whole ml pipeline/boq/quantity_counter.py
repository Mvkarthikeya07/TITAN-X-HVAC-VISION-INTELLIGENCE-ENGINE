from typing import List, Dict, Any
from collections import defaultdict

from constants import DetectedSymbol

from utils.logger import get_logger

logger = get_logger(__name__)


class QuantityCounter:
    """
    Counts equipment by type and aggregates pipe/duct lengths per floor or area.
    """

    def __init__(self):
        """Initializes the QuantityCounter."""
        logger.info("Initialized QuantityCounter.")

    def count_equipment(self, detections: List[DetectedSymbol]) -> Dict[str, int]:
        """
        Counts the occurrences of each equipment type.

        Args:
            detections (List[DetectedSymbol]): List of detection objects.

        Returns:
            Dict[str, int]: Dictionary mapping equipment labels to their count.
        """
        counts = defaultdict(int)
        for sym in detections:
            label = sym.equipment_tag if sym.equipment_tag else sym.class_name
            counts[str(label)] += 1
            
        logger.info(f"Counted {sum(counts.values())} total equipment across {len(counts)} types.")
        return dict(counts)

    def aggregate_lengths(self, line_detections: List[Dict[str, Any]], group_key: str = "floor") -> Dict[str, Dict[str, float]]:
        """
        Aggregates lengths (e.g. pipes, ducts) grouped by a specific key (like floor).

        Args:
            line_detections (List[Dict[str, Any]]): List of line-type detections containing 'length' and 'label'.
            group_key (str): The key in the detection dict to group by (e.g., 'floor', 'system').

        Returns:
            Dict[str, Dict[str, float]]: Nested dictionary mapping group_value -> label -> total length.
        """
        aggregation = defaultdict(lambda: defaultdict(float))
        
        for item in line_detections:
            group_val = str(item.get(group_key, "Unknown"))
            label = str(item.get("label", "Unknown_Line"))
            length = float(item.get("length", 0.0))
            
            aggregation[group_val][label] += length
            
        # Convert defaultdicts to standard dicts
        result = {k: dict(v) for k, v in aggregation.items()}
        logger.info(f"Aggregated lengths across {len(result)} groups.")
        return result
