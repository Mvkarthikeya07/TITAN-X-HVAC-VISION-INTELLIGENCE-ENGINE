"""
HVAC Component Detector
Wraps YOLO model with consistent interface.
"""

import os
import cv2

CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.40
IMG_SIZE       = 1280
USE_AUGMENT    = True

CLASS_COLORS = {
    "pipe":            (255,   0,   0),
    "valve":           (  0, 255,   0),
    "duct":            (  0,   0, 255),
    "diffuser":        (255, 255,   0),
    "supply diffuser": (255, 255,   0),
    "supply_diffuser": (255, 255,   0),
    "pump":            (255,   0, 255),
    "fan":             (  0, 255, 255),
    "boiler":          (128,   0, 255),
    "chiller":         (255, 128,   0),
    "coil":            (  0, 128, 255),
    "filter":          (128, 255,   0),
    "damper":          (  0, 255, 128),
    "thermostat":      (255,   0, 128),
}
DEFAULT_COLOR = (0, 200, 100)


def load_model(model_path):
    from ultralytics import YOLO
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    model = YOLO(model_path)
    print(f"[Detector] Loaded model with {len(model.names)} classes: {list(model.names.values())}")
    return model


def detect_objects(model, image_path, output_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    results = model(
        image_path,
        conf    = CONF_THRESHOLD,
        iou     = IOU_THRESHOLD,
        imgsz   = IMG_SIZE,
        augment = USE_AUGMENT,
        verbose = False,
    )[0]

    counts = {}

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        label  = model.names[cls_id].lower()

        # Draw bounding box
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        color = CLASS_COLORS.get(label, DEFAULT_COLOR)
        thickness = max(2, img.shape[1] // 500)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        text = f"{label} {conf:.2f}"
        font_scale = max(0.45, img.shape[1] / 2000)
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        ly = max(y1 - 4, th + 4)
        cv2.rectangle(img, (x1, ly - th - bl - 4), (x1 + tw + 4, ly + bl), color, -1)
        cv2.putText(img, text, (x1 + 2, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

        counts[label] = counts.get(label, 0) + 1

    # Summary overlay
    y_off = 30
    for lbl, cnt in sorted(counts.items()):
        txt = f"{lbl}: {cnt}"
        cv2.putText(img, txt, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(img, txt, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
        y_off += 28

    cv2.imwrite(output_path, img)
    print(f"[Detector] Output saved: {output_path} | Counts: {counts}")
    return counts