"""
BOQ (Bill of Quantities) Generator Module for InTakeoff Pipeline.

Aggregates all ML extractions (YOLO symbols, Pipe/Duct lengths) into a final
structured JSON-ready dictionary.
"""

import json
from typing import List, Dict, Any

from constants import DetectedSymbol
from boq.excel_formatter import ExcelFormatter
from boq.quantity_counter import QuantityCounter
from utils.logger import get_logger

logger = get_logger(__name__)

class BOQGenerator:
    """
    Generates the final Bill of Quantities.
    """
    
    def generate(self, symbols: List[DetectedSymbol], total_pipe_feet: float, total_duct_sqft: float) -> Dict[str, Any]:
        """
        Aggregates data into a dictionary payload.
        
        Args:
            symbols (List[DetectedSymbol]): Detections.
            total_pipe_feet (float): Total length of pipes.
            total_duct_sqft (float): Total area of ducts.
            
        Returns:
            Dict[str, Any]: Dictionary of the BOQ.
        """
        logger.info("Generating Final BOQ...")
        
        # 1. Aggregate Equipment by tag (if available) or class name
        # We will build a list of unique equipment instances and their counts
        # e.g., {"class": "AHU", "tag": "AHU-1", "quantity": 1}
        equipment_map: Dict[str, Dict[str, Any]] = {}
        
        for sym in symbols:
            eq_class = sym.class_name
            eq_tag = sym.equipment_tag
            
            # Use tag as unique identifier if exists, else the class itself
            key = eq_tag if eq_tag else f"{eq_class}_untagged"
            
            if key not in equipment_map:
                equipment_map[key] = {
                    "class": eq_class,
                    "tag": eq_tag,
                    "quantity": 0
                }
            equipment_map[key]["quantity"] += 1
            
        equipment_list = list(equipment_map.values())
            
        # 2. Build Payload
        boq = {
            "materials": {
                "pipe_linear_feet": round(total_pipe_feet, 2),
                "duct_square_feet": round(total_duct_sqft, 2)
            },
            "equipment": equipment_list,
            "metadata": {
                "total_items": sum(item["quantity"] for item in equipment_list)
            }
        }
        
        # 3. Return Dictionary
        return boq

    def save_excel(self, boq: Dict[str, Any], path: str) -> None:
        """Saves BOQ dict to Excel using ExcelFormatter."""
        formatter = ExcelFormatter()
        # Create dummy aggregations for length metrics
        length_aggs = {
            "Piping": {"Total Pipe Length": boq.get("materials", {}).get("pipe_linear_feet", 0.0)},
            "Ducting": {"Total Duct Area": boq.get("materials", {}).get("duct_square_feet", 0.0)}
        }
        rows = formatter.format_boq_data(boq.get("equipment", []), length_aggs)
        formatter.save_to_excel(rows, path)

    def save_csv(self, boq: Dict[str, Any], path: str) -> None:
        """Saves BOQ dict to CSV using ExcelFormatter."""
        formatter = ExcelFormatter()
        length_aggs = {
            "Piping": {"Total Pipe Length": boq.get("materials", {}).get("pipe_linear_feet", 0.0)},
            "Ducting": {"Total Duct Area": boq.get("materials", {}).get("duct_square_feet", 0.0)}
        }
        rows = formatter.format_boq_data(boq.get("equipment", []), length_aggs)
        formatter.save_to_csv(rows, path)
