import pandas as pd
from typing import List, Dict, Any

from utils.logger import get_logger

logger = get_logger(__name__)


class ExcelFormatter:
    """
    Formats BOQ (Bill of Quantities) data into an Excel-ready structure.
    """

    def __init__(self, default_sheet_name: str = "BOQ"):
        """
        Initializes the ExcelFormatter.

        Args:
            default_sheet_name (str): Default name for the output Excel sheet.
        """
        self.default_sheet_name = default_sheet_name
        logger.info("Initialized ExcelFormatter.")

    def format_boq_data(self, equipment_list: List[Dict[str, Any]], 
                        length_aggregations: Dict[str, Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Converts raw counts and aggregations into a flat list of dictionaries suitable for Excel rows.

        Args:
            equipment_list (List[Dict[str, Any]]): List of dicts with 'class', 'tag', 'quantity'.
            length_aggregations (Dict[str, Dict[str, float]], optional): Nested map of group -> label -> length.

        Returns:
            List[Dict[str, Any]]: Flattened list of rows with keys like 'Type', 'Item', 'Quantity', 'Unit', 'Group'.
        """
        rows = []
        
        # Format equipment counts
        for item in equipment_list:
            eq_class = item.get("class", "Unknown")
            eq_tag = item.get("tag")
            qty = item.get("quantity", 0)
            
            rows.append({
                "Type": "Equipment",
                "Equipment Class": eq_class,
                "Equipment Tag": eq_tag if eq_tag else "N/A",
                "Group": "General",
                "Quantity": qty,
                "Unit": "ea"
            })
            
        # Format length aggregations if provided
        if length_aggregations:
            for group, items in length_aggregations.items():
                for label, length in items.items():
                    rows.append({
                        "Type": "Linear",
                        "Equipment Class": label,
                        "Equipment Tag": "N/A",
                        "Group": group,
                        "Quantity": round(length, 2),
                        "Unit": "ft" # Assuming feet, could be configurable
                    })
                    
        logger.info(f"Formatted {len(rows)} BOQ rows.")
        return rows

    def save_to_excel(self, formatted_data: List[Dict[str, Any]], filepath: str, sheet_name: str = None) -> None:
        """
        Saves the formatted data directly to an Excel file.

        Args:
            formatted_data (List[Dict[str, Any]]): The list of row dictionaries.
            filepath (str): Output file path (.xlsx).
            sheet_name (str, optional): Overrides default sheet name.
        """
        if not formatted_data:
            logger.warning("No data provided to save_to_excel. Creating empty file.")
            
        df = pd.DataFrame(formatted_data)
        sheet = sheet_name or self.default_sheet_name
        
        try:
            df.to_excel(filepath, index=False, sheet_name=sheet)
            logger.info(f"Successfully saved BOQ to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save BOQ to Excel: {str(e)}")
            raise

    def save_to_csv(self, formatted_data: List[Dict[str, Any]], filepath: str) -> None:
        """
        Saves the formatted data directly to a CSV file.

        Args:
            formatted_data (List[Dict[str, Any]]): The list of row dictionaries.
            filepath (str): Output file path (.csv).
        """
        if not formatted_data:
            logger.warning("No data provided to save_to_csv. Creating empty file.")
            
        df = pd.DataFrame(formatted_data)
        
        try:
            df.to_csv(filepath, index=False)
            logger.info(f"Successfully saved BOQ to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save BOQ to CSV: {str(e)}")
            raise
