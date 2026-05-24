import os
import cv2
import numpy as np
from ultralytics import YOLO

# ==============================
# ⚙️ CONFIGURATION
# ==============================
MODEL_PATH     = "weights/best.pt"   # ← path to your trained model
INPUT_IMAGE    = "input.jpg"         # ← path to your input image
OUTPUT_IMAGE   = "output.jpg"        # ← path to save annotated result

CONF_THRESHOLD = 0.25                # lower to 0.10 or 0.01 if nothing is detected
IOU_THRESHOLD  = 0.45                # NMS overlap threshold
IMG_SIZE       = 640                 # must match your training imgsz (check data.yaml)
USE_AUGMENT    = True                # helps detection on engineering/schematic drawings


# ==============================
# 🎨 CLASS COLORS (BGR format)
# ==============================
CLASS_COLORS = {
    "pipe":       (255,   0,   0),
    "valve":      (  0, 255,   0),
    "duct":       (  0,   0, 255),
    "diffuser":   (255, 255,   0),
    "pump":       (255,   0, 255),
    "fan":        (  0, 255, 255),
    "boiler":     (128,   0, 255),
    "chiller":    (255, 128,   0),
    "coil":       (  0, 128, 255),
    "filter":     (128, 255,   0),
    "damper":     (  0, 255, 128),
    "thermostat": (255,   0, 128),
}
DEFAULT_COLOR = (0, 255, 0)          # bright green fallback for unknown classes


# ==============================
# ✅ LOAD MODEL
# ==============================
def load_model(model_path):
    print(f"\n{'='*40}")
    print(f"📂 Model path : {model_path}")
    print(f"📂 Exists     : {os.path.exists(model_path)}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found: {model_path}")

    model = YOLO(model_path)
    print(f"✅ Model loaded successfully")
    print(f"📋 Classes ({len(model.names)}): {list(model.names.values())}")

    if len(model.names) != 12:
        print(f"⚠️  Warning: expected 12 HVAC classes, got {len(model.names)}")

    print(f"{'='*40}\n")
    return model


# ==============================
# 🔍 VALIDATION HELPER
# ==============================
def validate_results(results, class_names):
    boxes  = results.boxes.xywhn
    scores = results.boxes.conf

    print("\n========== VALIDATION ==========")
    print(f"🔍 Total detections : {len(boxes)}")

    if len(boxes) == 0:
        print("⚠️  No detections — try lowering CONF_THRESHOLD or check model weights")
    elif len(boxes) > 1000:
        print("⚠️  Too many detections — consider raising CONF_THRESHOLD")
    else:
        print("✅ Detection count OK")

    invalid_boxes = sum(
        1 for box in boxes if not all(0.0 <= float(x) <= 1.0 for x in box)
    )
    if invalid_boxes:
        print(f"❌ {invalid_boxes} box(es) with invalid coordinates")
    else:
        print("✅ All bounding-box coordinates valid")

    invalid_scores = sum(1 for s in scores if not (0.0 <= float(s) <= 1.0))
    if invalid_scores:
        print(f"❌ {invalid_scores} confidence score(s) out of range")
    else:
        print("✅ All confidence scores valid")

    print("\n--- Raw detections (before threshold filter) ---")
    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        label  = class_names.get(cls_id, f"id={cls_id}")
        coords = [round(v, 1) for v in box.xyxy[0].tolist()]
        print(f"  [{i:03d}] class={label:<15} conf={conf:.3f}  xyxy={coords}")

    print("=" * 40 + "\n")


# ==============================
# 🖼️ DRAW SINGLE BOX + LABEL
# ==============================
def draw_box(img, box, label, conf):
    x1 = int(box.xyxy[0][0].item())
    y1 = int(box.xyxy[0][1].item())
    x2 = int(box.xyxy[0][2].item())
    y2 = int(box.xyxy[0][3].item())

    color     = CLASS_COLORS.get(label, DEFAULT_COLOR)
    thickness = max(2, img.shape[1] // 400)

    # Bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # Label with filled background for readability
    text       = f"{label}  {conf:.2f}"
    font_scale = max(0.5, img.shape[1] / 1800)
    font_thick = max(1, thickness - 1)
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick
    )
    label_y = max(y1 - 4, th + 4)

    cv2.rectangle(
        img,
        (x1, label_y - th - baseline - 4),
        (x1 + tw + 4, label_y + baseline),
        color, -1
    )
    cv2.putText(
        img, text,
        (x1 + 2, label_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        font_thick,
        cv2.LINE_AA
    )


# ==============================
# 📊 DRAW SUMMARY OVERLAY
# ==============================
def draw_summary(img, counts):
    y_offset = 30
    for label, count in sorted(counts.items()):
        text = f"{label}: {count}"
        # Black shadow
        cv2.putText(img, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        # White text
        cv2.putText(img, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y_offset += 30


# ==============================
# 🎯 MAIN DETECTION FUNCTION
# ==============================
def detect_objects(model, image_path, output_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"❌ Cannot read image: {image_path}")
    print(f"🖼️  Image loaded: {image_path}  shape={img.shape}")

    # Run inference
    results = model(
        image_path,
        conf    = CONF_THRESHOLD,
        iou     = IOU_THRESHOLD,
        imgsz   = IMG_SIZE,
        augment = USE_AUGMENT,
        verbose = False,
    )[0]

    # Validate and print raw detections
    validate_results(results, model.names)

    # Draw detections
    counts = {}

    if len(results.boxes) == 0:
        print("⚠️  No objects detected in image.")
    else:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])

            if conf < CONF_THRESHOLD:
                continue

            label = model.names[cls_id].lower()
            draw_box(img, box, label, conf)
            counts[label] = counts.get(label, 0) + 1

    # Draw summary on image
    draw_summary(img, counts)

    # Save result
    cv2.imwrite(output_path, img)
    print(f"\n✅ Output saved : {output_path}")
    print(f"📦 Detected     : {counts}")

    return counts


# ==============================
# 🚀 ENTRY POINT
# ==============================
if __name__ == "__main__":
    # Load model once
    model = load_model(MODEL_PATH)

    # Run detection — all 3 arguments provided
    counts = detect_objects(model, INPUT_IMAGE, OUTPUT_IMAGE)

    # Final summary in terminal
    print("\n========== FINAL COUNTS ==========")
    if counts:
        for label, count in sorted(counts.items()):
            print(f"  {label:<20} : {count}")
    else:
        print("  No detections above threshold.")
        print("\n💡 Troubleshooting tips:")
        print("  1. Lower CONF_THRESHOLD (try 0.10 or 0.01)")
        print("  2. Verify IMG_SIZE matches your training imgsz")
        print("  3. Confirm best.pt is your HVAC-trained model, not a generic one")
        print("  4. Check model.names printed above — classes should be HVAC types")
    print("=" * 35)