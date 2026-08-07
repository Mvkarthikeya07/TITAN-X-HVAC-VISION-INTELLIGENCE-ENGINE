"""
Confidence Fusion Module for InTakeoff Pipeline.

Fuses YOLO object detection confidence, Cosine similarity score, and OCR confidence
into a single, robust confidence metric using evidence-quality-aware weighted heuristics.

Key design principles:
- Confidence is only fused from sources that actually contributed evidence.
- A class-prefix match between YOLO detection and OCR tag boosts confidence.
- A class-prefix mismatch between YOLO detection and OCR tag flags a contradiction
  and REDUCES confidence rather than blindly averaging.
"""

import re
from typing import List

from constants import DetectedSymbol
from utils.logger import get_logger

logger = get_logger(__name__)


class ConfidenceFusion:
    """
    Merges different ML confidence metrics into one unified score,
    with evidence-quality awareness and contradiction detection.
    """

    def __init__(
        self,
        yolo_weight: float = 0.5,
        sim_weight: float = 0.3,
        ocr_weight: float = 0.2,
    ):
        self.yolo_weight = yolo_weight
        self.sim_weight = sim_weight
        self.ocr_weight = ocr_weight

        # Normalize weights so they sum to 1.0
        total = self.yolo_weight + self.sim_weight + self.ocr_weight
        self.yolo_weight /= total
        self.sim_weight /= total
        self.ocr_weight /= total

    @staticmethod
    def _extract_prefix(tag: str) -> str:
        """Extract the alphabetic prefix from an equipment tag (e.g. 'RTU-5' -> 'RTU')."""
        m = re.match(r'^([A-Za-z]+)', tag)
        return m.group(1).upper() if m else ""

    def _check_class_tag_agreement(self, sym: DetectedSymbol) -> str:
        """
        Compare the YOLO class_name against the OCR tag prefix.

        Returns:
            'match'       — prefix agrees with class (e.g. YOLO=RTU, tag=RTU-5)
            'mismatch'    — prefix contradicts class (e.g. YOLO=AHU, tag=VB-1)
            'no_evidence' — no OCR tag available to compare
        """
        if not sym.equipment_tag:
            return "no_evidence"

        tag_prefix = self._extract_prefix(sym.equipment_tag)
        class_upper = sym.class_name.upper()

        if not tag_prefix:
            return "no_evidence"

        # Direct match or known aliases
        if tag_prefix == class_upper:
            return "match"

        # Handle known aliases (VB = VAV_Box, etc.)
        aliases = {
            "VB": ["VAV_BOX", "VAV"],
            "VAV": ["VAV_BOX", "VB"],
            "EF": ["EXHAUST_FAN"],
            "SF": ["SUPPLY_FAN"],
            "CU": ["CONDENSING_UNIT"],
            "HP": ["HEAT_PUMP"],
        }
        alias_list = aliases.get(tag_prefix, [])
        if class_upper in alias_list or class_upper.replace("_", "") == tag_prefix:
            return "match"

        return "mismatch"

    def fuse(self, symbols: List[DetectedSymbol]) -> List[DetectedSymbol]:
        """
        Updates the confidence attribute of each symbol by fusing multiple scores
        with evidence-quality awareness.

        Fusion logic:
        - YOLO confidence is always included.
        - ResNet cosine score is included only if available.
        - OCR score is included only if a tag was found.
        - If OCR tag prefix MATCHES the YOLO class, OCR weight gets a 50% boost.
        - If OCR tag prefix CONTRADICTS the YOLO class, overall confidence is penalized.
        """
        logger.info(f"Applying Confidence Fusion to {len(symbols)} detected symbols...")

        contradictions = 0
        agreements = 0

        for sym in symbols:
            # Start with YOLO confidence
            yolo_w = self.yolo_weight
            total_weight = yolo_w
            final_conf = sym.confidence * yolo_w

            # Incorporate ResNet Cosine Score if matched
            if sym.cosine_score is not None:
                final_conf += sym.cosine_score * self.sim_weight
                total_weight += self.sim_weight

            # Incorporate OCR Score with evidence-quality adjustment
            if sym.equipment_tag is not None and sym.ocr_score is not None:
                agreement = self._check_class_tag_agreement(sym)

                if agreement == "match":
                    # OCR confirms YOLO — boost OCR weight by 50%
                    boosted_ocr_w = self.ocr_weight * 1.5
                    final_conf += sym.ocr_score * boosted_ocr_w
                    total_weight += boosted_ocr_w
                    agreements += 1

                elif agreement == "mismatch":
                    # OCR contradicts YOLO — include OCR at reduced weight
                    # and apply a 15% penalty to overall confidence
                    reduced_ocr_w = self.ocr_weight * 0.5
                    final_conf += sym.ocr_score * reduced_ocr_w
                    total_weight += reduced_ocr_w
                    contradictions += 1
                    logger.warning(
                        f"Contradiction: YOLO={sym.class_name}, "
                        f"OCR tag={sym.equipment_tag}. "
                        f"Confidence penalized."
                    )

                else:
                    # Normal OCR inclusion
                    final_conf += sym.ocr_score * self.ocr_weight
                    total_weight += self.ocr_weight

            # Normalize final confidence
            fused = final_conf / total_weight if total_weight > 0 else sym.confidence

            # Apply mismatch penalty after normalization
            agreement = self._check_class_tag_agreement(sym)
            if agreement == "mismatch":
                fused *= 0.85  # 15% penalty for contradicting evidence

            sym.confidence = round(min(1.0, max(0.0, fused)), 4)

        logger.info(
            f"Confidence Fusion complete. "
            f"{agreements} class-tag agreements, {contradictions} contradictions."
        )
        return symbols
