"""
Visual Debug Overlay Script for InTakeoff Pipeline.

Renders the rasterized drawing pages with:
  1. YOLO bounding boxes + predicted class labels (GREEN)
  2. OCR text boxes + recognized text (CYAN)
  3. Assignment lines connecting each OCR tag to its assigned symbol (YELLOW)
  4. Contradiction highlights where YOLO class != OCR tag prefix (RED)

Output: one PNG per page saved to output_debug/

Usage:
    python debug_overlay.py
"""

import os
import sys
import cv2
import math
import re
import numpy as np
import fitz
import time

sys.stdout.reconfigure(encoding='utf-8')

from preprocessing.rasterizer import PyMuPDFRasterizer
from preprocessing.page_classifier import PageClassifier
from preprocessing.tile_generator import TileGenerator
from constants import DetectedSymbol, BoundingBox
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Colour palette ──────────────────────────────────────────────────
YOLO_COLOR   = (0, 200, 0)      # Green – YOLO boxes
OCR_COLOR    = (255, 200, 0)    # Cyan  – OCR text boxes
LINE_COLOR   = (0, 255, 255)    # Yellow – assignment lines
CONTRA_COLOR = (0, 0, 255)      # Red   – contradiction highlights
MATCH_COLOR  = (0, 255, 0)      # Green – agreement highlights


def extract_prefix(tag: str) -> str:
    m = re.match(r'^([A-Za-z]+)', tag)
    return m.group(1).upper() if m else ""


def draw_text_bg(img, text, org, font_scale=0.5, thickness=1, color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Draw text with a filled background rectangle for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + baseline + 2), bg_color, -1)
    cv2.putText(img, text, (x + 2, y), font, font_scale, color, thickness, cv2.LINE_AA)


def run_debug_overlay():
    pdf_path = "sample 1.pdf"
    output_dir = "output_debug"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  VISUAL DEBUG OVERLAY")
    print("=" * 70)

    # ── Stage 1: Rasterize ──────────────────────────────────────────
    print("\n[1] Rasterizing PDF...")
    rasterizer = PyMuPDFRasterizer(dpi=150)
    pages = []
    for page_num, img in rasterizer.stream_pages(pdf_path):
        pages.append((page_num, img))
        print(f"  -> Page {page_num}: {img.shape}")

    # ── Stage 2: YOLO Detection ─────────────────────────────────────
    print("\n[2] Running YOLO detection...")
    yolo_path = "models/yolo/best.onnx"
    if not os.path.exists(yolo_path) or os.path.getsize(yolo_path) == 0:
        print("  -> YOLO model not found. Cannot generate overlay.")
        return

    from detection.yolo_detector import YOLOv8Detector
    from detection.global_nms import GlobalNMS
    tile_gen = TileGenerator()
    yolo = YOLOv8Detector(model_path=yolo_path)
    nms = GlobalNMS()

    page_symbols = {}  # page_num -> list of DetectedSymbol
    for page_num, img in pages:
        tiles = tile_gen.generate_tiles(img)
        symbols = yolo.detect(tiles)
        symbols = nms.apply(symbols)
        for s in symbols:
            s.page_num = page_num
        page_symbols[page_num] = symbols
        print(f"  -> Page {page_num}: {len(symbols)} symbols")

    # ── Stage 3: OCR + Tag Association ──────────────────────────────
    print("\n[3] Running OCR tag reading...")
    from ocr.paddle_ocr import PaddleOCREngine
    from ocr.equipment_tag_reader import EquipmentTagReader

    ocr = PaddleOCREngine()
    tag_reader = EquipmentTagReader(ocr)

    # Also collect raw OCR boxes for overlay
    page_ocr_boxes = {}  # page_num -> list of (BoundingBox, text, conf)

    for page_num, img in pages:
        symbols = page_symbols.get(page_num, [])
        if not symbols:
            continue

        # Run tag reader (assigns tags to symbols)
        tag_reader.read_tags(img, symbols)

        # Run full-page OCR to get ALL text boxes for overlay
        print(f"  -> Running full-page OCR on page {page_num} for overlay...")
        try:
            all_boxes = ocr.extract_boxes(img)
            page_ocr_boxes[page_num] = all_boxes
            print(f"  -> Page {page_num}: {len(all_boxes)} OCR text boxes found")
        except Exception as e:
            print(f"  -> Full-page OCR failed on page {page_num}: {e}")
            page_ocr_boxes[page_num] = []

    # ── Stage 4: Render Overlays ────────────────────────────────────
    print("\n[4] Rendering debug overlays...")

    for page_num, img in pages:
        # Work on a copy
        canvas = img.copy()
        if len(canvas.shape) == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        symbols = page_symbols.get(page_num, [])
        ocr_boxes = page_ocr_boxes.get(page_num, [])

        agreements = 0
        contradictions = 0

        # ── Draw OCR text boxes (cyan) ──────────────────────────────
        for bbox, text, conf in ocr_boxes:
            x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), OCR_COLOR, 1)
            label = f"{text} ({conf:.2f})"
            draw_text_bg(canvas, label, (x1, y1 - 2), font_scale=0.35, color=OCR_COLOR, bg_color=(40, 40, 40))

        # ── Draw YOLO boxes + assignment lines ──────────────────────
        for sym in symbols:
            x1, y1 = int(sym.bbox.x1), int(sym.bbox.y1)
            x2, y2 = int(sym.bbox.x2), int(sym.bbox.y2)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Determine box color based on agreement
            box_color = YOLO_COLOR
            tag_info = ""

            if sym.equipment_tag:
                tag_prefix = extract_prefix(sym.equipment_tag)
                class_upper = sym.class_name.upper()

                # Check alias matches
                aliases = {
                    "VB": ["VAV_BOX", "VAV"],
                    "VAV": ["VAV_BOX", "VB"],
                    "EF": ["EXHAUST_FAN"],
                    "SF": ["SUPPLY_FAN"],
                    "CU": ["CONDENSING_UNIT"],
                    "HP": ["HEAT_PUMP"],
                }
                alias_list = aliases.get(tag_prefix, [])
                is_match = (tag_prefix == class_upper or class_upper in alias_list)

                if is_match:
                    box_color = MATCH_COLOR
                    tag_info = f" -> {sym.equipment_tag} ✓"
                    agreements += 1
                else:
                    box_color = CONTRA_COLOR
                    tag_info = f" -> {sym.equipment_tag} ✗ MISMATCH"
                    contradictions += 1

                    # Draw thick red border for contradictions
                    cv2.rectangle(canvas, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), CONTRA_COLOR, 3)

            # Draw YOLO box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)

            # Draw class label
            label = f"{sym.class_name} ({sym.confidence:.2f}){tag_info}"
            draw_text_bg(canvas, label, (x1, y1 - 14), font_scale=0.4, color=box_color, bg_color=(0, 0, 0))

            # Draw assignment line from symbol center to the OCR tag location
            if sym.equipment_tag and sym.ocr_score is not None:
                # Find the OCR box that contains this tag text
                for bbox, text, conf in ocr_boxes:
                    if sym.equipment_tag.replace("-", "") in text.upper().replace("-", "").replace(" ", ""):
                        ocr_cx = int((bbox.x1 + bbox.x2) / 2)
                        ocr_cy = int((bbox.y1 + bbox.y2) / 2)
                        line_col = MATCH_COLOR if (tag_prefix == class_upper or class_upper in alias_list) else CONTRA_COLOR
                        cv2.arrowedLine(canvas, (cx, cy), (ocr_cx, ocr_cy), line_col, 2, tipLength=0.03)
                        break

        # ── Save output ─────────────────────────────────────────────
        out_path = os.path.join(output_dir, f"page_{page_num}_debug.png")
        cv2.imwrite(out_path, canvas)
        print(f"  -> Saved: {out_path}")
        print(f"     YOLO detections: {len(symbols)}")
        print(f"     OCR text boxes: {len(ocr_boxes)}")
        print(f"     Agreements: {agreements}, Contradictions: {contradictions}")
        tagged = sum(1 for s in symbols if s.equipment_tag)
        print(f"     Tagged: {tagged}/{len(symbols)}")

    print("\n" + "=" * 70)
    print(f"  Debug overlays saved to: {output_dir}/")
    print("  Open the PNGs to visually inspect:")
    print("    GREEN boxes  = YOLO detection + class (agreement)")
    print("    RED boxes    = YOLO detection with contradicting OCR tag")
    print("    CYAN boxes   = All OCR text detections")
    print("    ARROWS       = Assignment lines (OCR tag -> symbol)")
    print("=" * 70)


if __name__ == "__main__":
    run_debug_overlay()
