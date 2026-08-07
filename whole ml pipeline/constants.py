"""
Core enumerations and basic Pydantic structures for the pipeline.
"""
from enum import Enum
from pydantic import BaseModel
from typing import Optional, List

class PageType(str, Enum):
    HVAC_PLAN = "HVAC_PLAN"
    LEGEND = "LEGEND"
    MECHANICAL_SCHEDULE = "MECHANICAL_SCHEDULE"
    SPECIFICATION = "SPECIFICATION"
    DETAIL = "DETAIL"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"

class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def iou(self, other: "BoundingBox") -> float:
        """Compute Intersection over Union with another BoundingBox."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area() + other.area() - inter
        return inter / union if union > 0 else 0.0

class Point(BaseModel):
    x: float
    y: float

class DetectedSymbol(BaseModel):
    """
    Represents a single detected HVAC symbol on a drawing page.
    Fields like equipment_tag, cosine_score, ocr_score are populated
    by downstream pipeline stages (OCR, embedding matcher, fusion).
    """
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    tile_id: Optional[str] = None
    page_num: Optional[int] = None
    equipment_tag: Optional[str] = None
    cosine_score: Optional[float] = None
    ocr_score: Optional[float] = None
    legend_label: Optional[str] = None

class LegendEntry(BaseModel):
    symbol_id: str
    label: str
    embedding: Optional[List[float]] = None
    bbox: Optional[BoundingBox] = None
