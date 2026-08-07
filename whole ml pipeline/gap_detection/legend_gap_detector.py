"""
Legend Gap Detector Module for InTakeoff Pipeline.

Compares the reference legend dictionary against the detected symbols to flag
items that appear on the drawing but lack a definition in the legend, or vice versa.
"""

from typing import List
from pydantic import BaseModel

from constants import DetectedSymbol, LegendEntry, Severity
from utils.logger import get_logger

logger = get_logger(__name__)

class LegendGap(BaseModel):
    missing_component: str
    severity: Severity
    description: str

class LegendGapDetector:
    """
    Analyzes gaps between defined legend entries and detected plan symbols.
    """
    
    def analyze(self, legend_entries: List[LegendEntry], detected_symbols: List[DetectedSymbol]) -> List[LegendGap]:
        gaps = []
        
        legend_labels = {entry.label.lower() for entry in legend_entries}
        
        # Track which classes were detected
        detected_classes = set()
        for sym in detected_symbols:
            # Our YOLO classes are well defined, but Cosine Similarity might have appended 
            # the legend name. Let's simplify and just use the YOLO base class name if needed,
            # or the matched name.
            name = sym.class_name.lower()
            detected_classes.add(name)
            
        # 1. Used on Plan, but missing from Legend
        for cls_name in detected_classes:
            # Simple heuristic: see if any legend label mentions this class
            if not any(cls_name in lbl or lbl in cls_name for lbl in legend_labels):
                gaps.append(LegendGap(
                    missing_component=cls_name.title(),
                    severity=Severity.CRITICAL,
                    description=f"Symbol '{cls_name}' detected on plan but not defined in the Legend."
                ))
                
        # 2. In Legend, but never used on Plan
        # (This is a WARNING, as sometimes standard legends include unused items)
        for label in legend_labels:
            if not any(label in dcls or dcls in label for dcls in detected_classes):
                gaps.append(LegendGap(
                    missing_component=label.title(),
                    severity=Severity.WARNING,
                    description=f"Legend defines '{label}' but it is not used anywhere on the plan."
                ))
                
        logger.info(f"Legend Gap Analysis complete. Found {len(gaps)} gaps.")
        return gaps
