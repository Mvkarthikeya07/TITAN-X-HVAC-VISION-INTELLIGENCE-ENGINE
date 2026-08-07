import json
from datetime import datetime
from typing import List, Dict, Any

from utils.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """
    Creates structured JSON/dict reports from gap analysis results.
    """

    def __init__(self, project_name: str = "Unknown Project"):
        """
        Initializes the ReportGenerator.

        Args:
            project_name (str): Name of the project for the report header.
        """
        self.project_name = project_name
        logger.info(f"Initialized ReportGenerator for project: {project_name}")

    def generate_report(self, 
                        total_expected: int, 
                        total_detected: int, 
                        missing_items: List[Dict[str, Any]],
                        extra_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generates a structured report dictionary.

        Args:
            total_expected (int): Number of items expected from the schedule.
            total_detected (int): Number of items actually detected.
            missing_items (List[Dict[str, Any]]): List of missing schedule items.
            extra_metadata (Dict[str, Any], optional): Any additional information to include.

        Returns:
            Dict[str, Any]: The structured report.
        """
        report = {
            "project_name": self.project_name,
            "report_timestamp": datetime.utcnow().isoformat() + "Z",
            "summary": {
                "total_expected": total_expected,
                "total_detected": total_detected,
                "total_missing": len(missing_items),
                "completeness_percentage": 0.0 if total_expected == 0 else round((total_expected - len(missing_items)) / total_expected * 100, 2)
            },
            "missing_items": missing_items,
            "metadata": extra_metadata or {}
        }

        logger.info(f"Generated report with {len(missing_items)} missing items.")
        return report

    def save_report_to_file(self, report: Dict[str, Any], filepath: str) -> None:
        """
        Saves the generated report to a JSON file.

        Args:
            report (Dict[str, Any]): The report dictionary.
            filepath (str): Destination file path.
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=4)
            logger.info(f"Successfully saved report to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save report to {filepath}: {str(e)}")
            raise
