# HVAC Inference Endpoint Validation Report

## 1. Objective

To validate the HVAC object detection inference pipeline end-to-end using a held-out (unseen) HVAC drawing and verify correctness of predictions, output format, and system robustness.

---

## 2. System Overview

* **Model:** YOLOv8 (custom trained – `best.pt`)
* **Framework:** Ultralytics YOLOv8
* **Application:** Flask-based web interface
* **Pipeline:**

  1. Upload HVAC drawing (PDF/Image)
  2. Convert PDF → image (high resolution)
  3. Run object detection
  4. Visualize bounding boxes and labels
  5. Aggregate counts and generate output

---

## 3. Test Configuration

* **Input Image:** Unseen HVAC drawing (not used in training)
* **Image Size:** 312 × 365 pixels
* **Inference Parameters:**

  * Confidence Threshold: 0.01 (for debugging)
  * IoU Threshold: 0.3
  * Image Size: 1280
  * Augmentation: Enabled

---

## 4. Validation Criteria

The following checks were performed:

| Check              | Requirement                           |
| ------------------ | ------------------------------------- |
| Detection Count    | Must not be zero or excessively large |
| Bounding Box Range | Must be within [0, 1] (normalized)    |
| Class Mapping      | Must match 12 HVAC classes            |
| Confidence Scores  | Must be within [0.0, 1.0]             |

---

## 5. Results & Validation

### 5.1 Model Verification

* Model path: ✔ Valid
* Model loaded successfully: ✔ Yes
* Total classes detected in model: **8**

⚠️ **Issue:** Expected 12 classes, but model contains only 8 classes.

---

### 5.2 Detection Output

* Total detections (raw, conf=0.01): **6**
* Detected class: **Supply Diffuser only**

| Class           | Count | Confidence Range |
| --------------- | ----- | ---------------- |
| Supply Diffuser | 6     | 0.011 – 0.764    |

✔ Detection count is non-zero
⚠️ However, only a single class is detected

---

### 5.3 Bounding Box Validation

* All bounding boxes verified within normalized range [0, 1]

✔ **Status: Passed**

---

### 5.4 Confidence Score Validation

* All confidence values fall within [0.0, 1.0]

✔ **Status: Passed**

---

### 5.5 Class Mapping Validation

* Model provides valid class indices
* However, only 8 classes are present instead of required 12

⚠️ **Status: Partially Passed**

---

## 6. Visual Verification

* Bounding boxes are correctly rendered
* Labels and confidence scores are visible
* Detection is limited to a single class
* Several visible HVAC components remain undetected

---

## 7. Debug Analysis

Detailed debugging revealed:

* Model detects only **Supply Diffuser**
* No detections for:

  * Ducts
  * Pipes
  * Valves
  * Other HVAC symbols

### Key Observations:

* Confidence scores for non-primary detections are very low (< 0.05)
* Indicates weak feature learning for most classes

---

## 8. Root Cause Analysis

The issue is **not related to the inference pipeline**, as:

* Model loads correctly
* Inference executes without error
* Output format is valid
* Bounding boxes and confidence values are correct

The limitation is due to **model training issues**, specifically:

### 8.1 Incomplete Model Classes

* Model contains only 8 classes instead of required 12
* Indicates training was done on a partial dataset

### 8.2 Dataset Imbalance

* Model predicts only one class (Supply Diffuser)
* Suggests heavy imbalance in training data

### 8.3 Insufficient Training

* Weak confidence for most detections
* Poor generalization to unseen data

---

## 9. Conclusion

The inference pipeline has been successfully validated end-to-end.

✔ All system-level validations passed:

* Bounding box normalization
* Confidence score validity
* Inference execution

⚠️ However, detection performance is limited due to model constraints.

> The system is functionally correct, but the model is not yet performance-ready.

---

## 10. Recommendations

To improve detection performance:

1. Retrain model using complete dataset (12 classes)
2. Ensure balanced representation of all HVAC components
3. Increase training epochs (≥100)
4. Use higher resolution training (≥1024)
5. Consider larger model variant (YOLOv8m or YOLOv8l)
6. Validate on multiple unseen drawings

---

## 11. Key Insight

> The inference pipeline is fully operational and validated.
> The primary limitation lies in model training quality and dataset completeness.
