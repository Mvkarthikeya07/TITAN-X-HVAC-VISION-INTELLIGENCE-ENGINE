from typing import List, Dict, Any, Set

from constants import DetectedSymbol
from schedule.schedule_parser import EquipmentRecord

from utils.logger import get_logger

logger = get_logger(__name__)


class MissingSymbolDetector:
    """
    Checks if expected symbols from the schedule are present in the final detections.
    """

    def __init__(self):
        """Initializes the MissingSymbolDetector."""
        logger.info("Initialized MissingSymbolDetector.")

    def detect_missing_symbols(self, 
                               expected_symbols: List[EquipmentRecord], 
                               detected_symbols: List[DetectedSymbol]) -> List[EquipmentRecord]:
        """
        Compares expected schedule symbols against actual detected symbols to find missing ones.

        Args:
            expected_symbols (List[EquipmentRecord]): List of items expected from schedule.
            detected_symbols (List[DetectedSymbol]): List of actually detected items.

        Returns:
            List[EquipmentRecord]: List of expected symbols that were NOT found.
        """
        # Extract unique identifiers for detected symbols
        detected_tags: Set[str] = set()
        for det in detected_symbols:
            if det.equipment_tag:
                detected_tags.add(det.equipment_tag)

        missing_symbols = []

        for expected in expected_symbols:
            if expected.tag not in detected_tags:
                missing_symbols.append(expected)

        logger.info(f"Found {len(missing_symbols)} missing symbols out of {len(expected_symbols)} expected.")
        return missing_symbols
