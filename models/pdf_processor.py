import fitz  # PyMuPDF
import os


def pdf_to_images(pdf_path):
    """
    Convert PDF to images using PyMuPDF (fitz).
    No external dependencies like Poppler required.
    """

    # Check if file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ PDF not found: {pdf_path}")

    # Create upload folder if not exists
    output_dir = "static/uploads"
    os.makedirs(output_dir, exist_ok=True)

    # Open PDF
    doc = fitz.open(pdf_path)

    image_paths = []

    # Loop through pages
    for i, page in enumerate(doc):

        # Increase resolution (important for detection quality)
        zoom = 2  # 2 = 2x resolution
        mat = fitz.Matrix(zoom, zoom)

        pix = page.get_pixmap(matrix=mat)

        # Save image
        img_path = os.path.join(output_dir, f"page_{i}.jpg")
        pix.save(img_path)

        image_paths.append(img_path)

    doc.close()

    print(f"✅ Converted {len(image_paths)} pages to images")

    return image_paths