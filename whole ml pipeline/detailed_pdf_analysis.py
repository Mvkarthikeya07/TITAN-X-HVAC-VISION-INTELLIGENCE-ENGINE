"""
Detailed step-by-step execution of EVERY pipeline module on the sample PDF.
Tests all 14 stages individually and then runs the full pipeline.
"""
import os
import sys
import fitz
import json
import numpy as np
import time

sys.stdout.reconfigure(encoding='utf-8')

from preprocessing.rasterizer import PyMuPDFRasterizer
from preprocessing.page_classifier import PageClassifier
from preprocessing.tile_generator import TileGenerator
from cv.geometry import ScaleConverter
from cv.line_detector import LineDetector
from schedule.table_extractor import TableExtractor
from schedule.schedule_parser import ScheduleParser
from schedule.cross_reference import CrossReferenceEngine
from schedule.schedule_matcher import ScheduleMatcher
from fusion.confidence_fusion import ConfidenceFusion
from fusion.result_merger import ResultMerger
from fusion.duplicate_removal import DuplicateRemover
from fusion.detection_filter import DetectionFilter
from gap_detection.legend_gap_detector import LegendGapDetector
from gap_detection.missing_symbol_detector import MissingSymbolDetector
from gap_detection.report_generator import ReportGenerator
from boq.boq_generator import BOQGenerator
from boq.quantity_counter import QuantityCounter
from boq.excel_formatter import ExcelFormatter
from constants import DetectedSymbol, BoundingBox, LegendEntry
from pipelines.full_pipeline import FullPipeline


def run_detailed_analysis():
    pdf_path = "sample 1.pdf"
    print("=" * 70)
    print(f"  DETAILED 14-STAGE PIPELINE EXECUTION ON: {pdf_path}")
    print("=" * 70)

    # ================================================================
    # STAGE 1: PDF Loading & Metadata
    # ================================================================
    print("\n[STAGE 1] PDF Loading & Metadata")
    doc = fitz.open(pdf_path)
    print(f"  -> Total Pages: {len(doc)}")
    for i in range(len(doc)):
        page = doc[i]
        print(f"  -> Page {i} Size: {page.rect.width:.1f} x {page.rect.height:.1f} points")

    # ================================================================
    # STAGE 2: Rasterization
    # ================================================================
    print("\n[STAGE 2] Rasterization (PDF → Images)")
    rasterizer = PyMuPDFRasterizer(dpi=150)
    pages = []
    for page_num, img in rasterizer.stream_pages(pdf_path):
        print(f"  -> Rasterized Page {page_num}: Shape {img.shape}, dtype {img.dtype}")
        pages.append((page_num, img))

    # ================================================================
    # STAGE 3: Page Classification
    # ================================================================
    print("\n[STAGE 3] Page Classification")
    classifier = PageClassifier()
    page_types = {}
    for i in range(len(doc)):
        ptype = classifier.classify(doc, i)
        page_types[i] = ptype
        print(f"  -> Page {i}: {ptype.name}")

    # ================================================================
    # STAGE 4: Tiling
    # ================================================================
    print("\n[STAGE 4] Image Tiling (for YOLO)")
    tile_gen = TileGenerator()
    if pages:
        first_page_img = pages[0][1]
        tiles = tile_gen.generate_tiles(first_page_img)
        print(f"  -> Generated {len(tiles)} tiles from Page 0")
        print(f"  -> Tile 0: {tiles[0].image.shape} at offset ({tiles[0].x_offset}, {tiles[0].y_offset})")
        print(f"  -> Tile {len(tiles)-1}: {tiles[-1].image.shape} at offset ({tiles[-1].x_offset}, {tiles[-1].y_offset})")

    # ================================================================
    # STAGE 5: YOLO Detection + NMS (requires model)
    # ================================================================
    print("\n[STAGE 5] YOLO Detection + Global NMS")
    yolo_path = "models/yolo/best.onnx"
    all_symbols = []
    if os.path.exists(yolo_path) and os.path.getsize(yolo_path) > 0:
        from detection.yolo_detector import YOLOv8Detector
        from detection.global_nms import GlobalNMS
        yolo = YOLOv8Detector(model_path=yolo_path)
        nms = GlobalNMS()
        for page_num, img in pages:
            tiles = tile_gen.generate_tiles(img)
            t0 = time.time()
            symbols = yolo.detect(tiles)
            symbols = nms.apply(symbols)
            # Stamp each symbol with its page number for per-page processing
            for sym in symbols:
                sym.page_num = page_num
            t1 = time.time()
            print(f"  -> Page {page_num}: {len(symbols)} symbols detected in {t1-t0:.2f}s")
            all_symbols.extend(symbols)
        print(f"  -> Total raw detections: {len(all_symbols)}")
    else:
        print("  -> YOLO model not found, skipping detection")

    # ================================================================
    # STAGE 6: OCR Tag Reading
    # ================================================================
    print("\n[STAGE 6] OCR Tag Reading (PaddleOCR)")
    try:
        from ocr.paddle_ocr import PaddleOCREngine
        from ocr.equipment_tag_reader import EquipmentTagReader
        ocr = PaddleOCREngine()
        tag_reader = EquipmentTagReader(ocr)
        if all_symbols and pages:
            tagged_count = 0
            for page_num, img in pages:
                # Filter symbols belonging to THIS page only
                page_syms = [s for s in all_symbols if s.page_num == page_num]
                if not page_syms:
                    print(f"  -> Page {page_num}: 0 symbols to tag, skipping.")
                    continue
                    
                print(f"  -> Page {page_num}: tagging {len(page_syms)} symbols...")
                try:
                    page_syms = tag_reader.read_tags(img, page_syms)
                    tagged = sum(1 for s in page_syms if s.equipment_tag)
                    tagged_count += tagged
                    print(f"  -> Page {page_num}: tagged {tagged}/{len(page_syms)} symbols")
                except Exception as e:
                    print(f"  -> OCR failed on page {page_num}: {e}")
                
            # Global deduplication: if same tag assigned to multiple symbols, keep the best
            tag_assignments = {}
            for sym in all_symbols:
                if sym.equipment_tag:
                    if sym.equipment_tag not in tag_assignments:
                        tag_assignments[sym.equipment_tag] = sym
                    else:
                        existing = tag_assignments[sym.equipment_tag]
                        existing_score = existing.ocr_score or 0.0
                        new_score = sym.ocr_score or 0.0
                        if new_score > existing_score:
                            existing.equipment_tag = None
                            existing.ocr_score = None
                            tag_assignments[sym.equipment_tag] = sym
                        else:
                            sym.equipment_tag = None
                            sym.ocr_score = None
                            
            print(f"  -> Tagged {tagged_count} symbols total (after dedup: {len(tag_assignments)} unique tags)")
            print("  -> OCR extracted tags:")
            for s in all_symbols:
                if s.equipment_tag:
                    print(f"     [OCR] {s.class_name} at ({s.bbox.x1:.0f},{s.bbox.y1:.0f}) -> {s.equipment_tag} (Score: {s.ocr_score:.2f})")
        else:
            print("  -> No symbols to tag")
    except Exception as e:
        print(f"  -> OCR initialization failed (expected on some systems): {e}")

    # ================================================================
    # STAGE 7: ResNet Embedding + Legend Matching
    # ================================================================
    print("\n[STAGE 7] ResNet Embedding Engine")
    resnet_path = "models/resnet/resnet18.onnx"
    if os.path.exists(resnet_path) and os.path.getsize(resnet_path) > 0:
        try:
            from embedding.resnet_embedder import ResNetEmbedder
            from embedding.embedding_utils import crop_symbol_from_image
            embedder = ResNetEmbedder(model_path=resnet_path)

            # Generate embeddings for detected symbols
            if all_symbols and pages:
                img = pages[0][1]
                crops = []
                for sym in all_symbols[:5]:  # First 5 for demo
                    crop = crop_symbol_from_image(
                        img, (sym.bbox.x1, sym.bbox.y1, sym.bbox.x2, sym.bbox.y2)
                    )
                    crops.append(crop)
                embeddings = embedder.get_embeddings(crops)
                print(f"  -> Generated embeddings: shape {embeddings.shape}")
                print(f"  -> Embedding vector norm (should be ~1.0): {np.linalg.norm(embeddings[0]):.4f}")
            else:
                print("  -> No symbols to embed")
        except Exception as e:
            print(f"  -> ResNet embedding failed: {e}")
            
        legend_page = next((p for p, pt in page_types.items() if pt.name == "LEGEND"), None)
        if legend_page is not None:
            print(f"  -> Legend page identified at index {legend_page}. ResNet matching will proceed.")
        else:
            print("  -> Searched all pages. No Legend page found. ResNet matching skipped.")
    else:
        print("  -> ResNet ONNX model not found, skipping embedding")

    # ================================================================
    # STAGE 8: OpenCV Pipe & Duct Measurement
    # ================================================================
    print("\n[STAGE 8] OpenCV Pipe & Duct Measurement")
    scale_conv = ScaleConverter(dpi=150, scale_str="1/8")
    line_detector = LineDetector(scale_conv)
    total_pipe = 0.0
    total_duct = 0.0
    for page_num, img in pages:
        t0 = time.time()
        lines, pipe_ft = line_detector.detect_pipes(img)
        ducts, duct_sqft = line_detector.detect_ducts(img)
        t1 = time.time()
        print(f"  -> Page {page_num} ({t1-t0:.2f}s): {len(lines)} pipes → {pipe_ft:.1f} ft, {len(ducts)} ducts → {duct_sqft:.1f} sqft")
        total_pipe += pipe_ft
        total_duct += duct_sqft

    # ================================================================
    # STAGE 9: Schedule Extraction & Parsing
    # ================================================================
    print("\n[STAGE 9] Schedule Extraction & Parsing")
    all_schedule_records = []
    parser = ScheduleParser()
    for i in range(len(doc)):
        table_results = TableExtractor().extract_tables_to_dfs(doc, i)
        if table_results:
            print(f"  -> Page {i}: {len(table_results)} tables found")
            for tr in table_results:
                title = tr.get('title', 'Untitled')
                df = tr.get('df')
                print(f"     Table: '{title}' ({len(df)} rows x {len(df.columns)} cols)")
                records = parser.parse_schedule(df)
                if records:
                    print(f"     -> Parsed {len(records)} equipment records")
                    for rec in records[:3]:
                        print(f"        [{rec.tag}] model={rec.model}, cfm={rec.cfm}, tons={rec.tons}")
                    if len(records) > 3:
                        print(f"        ... and {len(records) - 3} more")
                    all_schedule_records.extend(records)
        else:
            print(f"  -> Page {i}: 0 tables found")
    
    print(f"  -> Total schedule records: {len(all_schedule_records)}")

    doc.close()

    # ================================================================
    # STAGE 10: Fusion Chain
    # ================================================================
    print("\n[STAGE 10] Fusion Chain (Merge → Dedup → Filter → Confidence)")
    pre_fusion = len(all_symbols)
    if all_symbols:
        merger = ResultMerger(iou_threshold=0.5)
        dedup = DuplicateRemover(distance_threshold=10.0)
        filt = DetectionFilter(confidence_threshold=0.3)
        fusion = ConfidenceFusion()

        all_symbols = merger.merge_and_deduplicate(all_symbols)
        print(f"  -> After merge: {len(all_symbols)} symbols")
        all_symbols = dedup.remove_duplicates(all_symbols)
        print(f"  -> After dedup: {len(all_symbols)} symbols")
        all_symbols = filt.filter_detections(all_symbols)
        print(f"  -> After filter: {len(all_symbols)} symbols")
        
        print("\n  -> Fusion inputs:")
        # Print a before/after of confidence fusion for a few symbols
        for sym in all_symbols[:3]:
            print(f"     Symbol {sym.class_name}: Base YOLO={sym.confidence:.2f}, OCR={sym.ocr_score or 'N/A'}, ResNet={sym.cosine_score or 'N/A'}")
            
        all_symbols = fusion.fuse(all_symbols)
        print(f"  -> After confidence fusion: {len(all_symbols)} symbols")
        for sym in all_symbols[:3]:
            print(f"     -> Fused Confidence: {sym.confidence:.2f}")
            
        print("\n  -> Final validated tags:")
        for sym in all_symbols:
            if sym.equipment_tag:
                print(f"     Symbol {sym.class_name} -> {sym.equipment_tag}")
                
    print(f"  -> Fusion: {pre_fusion} → {len(all_symbols)} symbols")

    # ================================================================
    # STAGE 11: Cross Reference Engine
    # ================================================================
    print("\n[STAGE 11] Cross Reference Engine")
    cross_ref = CrossReferenceEngine()
    discrepancies = cross_ref.analyze(all_schedule_records, all_symbols)
    print(f"  -> Found {len(discrepancies)} discrepancies")
    for d in discrepancies[:5]:
        print(f"     [{d.issue_type}] {d.description}")

    # ================================================================
    # STAGE 12: Legend Gap Detection
    # ================================================================
    print("\n[STAGE 12] Legend Gap Detection")
    gap_detector = LegendGapDetector()
    all_legend_entries = []
    if all_legend_entries:
        gaps = gap_detector.analyze(all_legend_entries, all_symbols)
        print(f"  -> Found {len(gaps)} legend gaps")
        for g in gaps[:5]:
            print(f"     [{g.severity.value}] {g.description}")
    else:
        print("  -> No legend available. Skipping legend gap detection.")

    # Missing Symbol Detection
    missing_det = MissingSymbolDetector()
    missing = missing_det.detect_missing_symbols([], all_symbols)
    print(f"  -> Missing symbols: {len(missing)}")

    # ================================================================
    # STAGE 13: BOQ Generation (with Excel/CSV export)
    # ================================================================
    print("\n[STAGE 13] BOQ Generation")
    boq_gen = BOQGenerator()
    boq = boq_gen.generate(all_symbols, total_pipe, total_duct)
    print(f"  -> BOQ Equipment: {boq.get('equipment', {})}")
    print(f"  -> BOQ Materials: {boq.get('materials', {})}")

    # Test Excel/CSV export
    try:
        boq_gen.save_excel(boq, "output_boq.xlsx")
        print("  -> Saved: output_boq.xlsx")
    except Exception as e:
        print(f"  -> Excel export failed: {e}")

    try:
        boq_gen.save_csv(boq, "output_boq.csv")
        print("  -> Saved: output_boq.csv")
    except Exception as e:
        print(f"  -> CSV export failed: {e}")

    # Gap Report
    report_gen = ReportGenerator(project_name="Sample HVAC Project")
    report = report_gen.generate_report(
        total_expected=0,
        total_detected=len(all_symbols),
        missing_items=[],
        extra_metadata={"pipe_feet": total_pipe, "duct_sqft": total_duct}
    )
    print(f"  -> Gap Report: completeness={report['summary']['completeness_percentage']}%")

    # ================================================================
    # STAGE 14: Full End-to-End Pipeline
    # ================================================================
    print("\n" + "=" * 70)
    print("  [STAGE 14] FULL END-TO-END PIPELINE")
    print("=" * 70)
    pipeline = FullPipeline(
        dpi=150,
        scale_str="1/8",
        yolo_model_path="models/yolo/best.onnx" if os.path.exists("models/yolo/best.onnx") else None,
        resnet_model_path="models/resnet/resnet18.onnx" if (os.path.exists("models/resnet/resnet18.onnx") and os.path.getsize("models/resnet/resnet18.onnx") > 0) else None,
        use_ocr=True,
    )
    result = pipeline.process_pdf(pdf_path)

    print("\n" + "=" * 70)
    print("  FINAL PIPELINE JSON OUTPUT")
    print("=" * 70)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run_detailed_analysis()
