# Computer Vision Module

The CV module handles the extraction of linear runs (pipes) and areal runs (ducts) that are typically too large, interconnected, or uniquely shaped for standard bounding box object detection like YOLO.

## Geometry (`geometry.py`)
- **Scale Resolution**: Automatically converts pixels to real-world imperial feet using the known PDF rendering DPI and the architectural scale parsed from the title block (e.g., `1/8" = 1'-0"`).

## Line & Duct Detector (`line_detector.py`)
- **Pipes**: Uses Canny Edge Detection and `HoughLinesP` to extract thin linear runs and calculates total linear feet.
- **Ducts**: Employs morphology, Connected Components, and the `Douglas-Peucker` algorithm (`cv2.approxPolyDP`) to simplify thick duct layouts into polygonal arrays and calculate square footage for sheet metal BOQs.

### Usage Example
See `example_cv.py` for a runnable snippet.
