"""
Geometry Module for InTakeoff Pipeline.

Provides mathematical utilities and conversions for architectural scales.
Handles conversion of pixel lengths into real-world Imperial units (feet, inches).
"""

from utils.logger import get_logger
from settings import settings

logger = get_logger(__name__)

class ScaleConverter:
    """
    Converts pixel measurements to real-world feet based on PDF DPI and architectural scale.
    """
    
    def __init__(self, dpi: int = settings.PDF_DPI, scale_str: str = "1/8"):
        """
        Args:
            dpi (int): Rendering DPI.
            scale_str (str): Architectural scale (e.g., "1/8" meaning 1/8 inch = 1 foot).
        """
        self.dpi = dpi
        self.scale_fraction = self._parse_scale(scale_str)
        # Calculate how many pixels represent 1 foot.
        # If 1/8 inch = 1 foot, and 1 inch = DPI pixels,
        # then 1/8 inch = DPI / 8 pixels.
        self.pixels_per_foot = self.dpi * self.scale_fraction
        
    def _parse_scale(self, scale_str: str) -> float:
        """Parses a scale string like '1/8' into 0.125"""
        try:
            if "/" in scale_str:
                num, den = scale_str.split("/")
                return float(num) / float(den)
            return float(scale_str)
        except Exception as e:
            logger.warning(f"Failed to parse scale '{scale_str}'. Defaulting to 1/8. Error: {e}")
            return 0.125
            
    def pixels_to_feet(self, pixels: float) -> float:
        """Converts a pixel length into real world feet."""
        if self.pixels_per_foot <= 0:
            return 0.0
        return pixels / self.pixels_per_foot
        
    def pixel_area_to_sqft(self, pixel_area: float) -> float:
        """Converts a pixel area into square feet."""
        if self.pixels_per_foot <= 0:
            return 0.0
        return pixel_area / (self.pixels_per_foot ** 2)
