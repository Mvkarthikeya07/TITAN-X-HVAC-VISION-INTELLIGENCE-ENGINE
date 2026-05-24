"""
PDF to images converter.
Tries pdf2image first (needs poppler), falls back to PyMuPDF (fitz).
"""

import os


def pdf_to_images(pdf_path, output_dir=None, dpi=200):
    """
    Convert each page of a PDF into a JPEG image.
    Returns list of saved image paths.
    """
    if output_dir is None:
        output_dir = 'static/uploads'

    os.makedirs(output_dir, exist_ok=True)

    # Try pdf2image (poppler-based)
    try:
        from pdf2image import convert_from_path
        pages = convert_from_path(pdf_path, dpi=dpi)
        paths = []
        for i, page in enumerate(pages):
            out = os.path.join(output_dir, f'page_{i}.jpg')
            page.save(out, 'JPEG', quality=92)
            paths.append(out)
        print(f"[PDF] Converted {len(paths)} pages via pdf2image")
        return paths
    except Exception as e1:
        print(f"[PDF] pdf2image failed ({e1}), trying PyMuPDF...")

    # Fallback: PyMuPDF
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        paths = []
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat)
            out = os.path.join(output_dir, f'page_{i}.jpg')
            pix.save(out)
            paths.append(out)
        print(f"[PDF] Converted {len(paths)} pages via PyMuPDF")
        return paths
    except Exception as e2:
        raise RuntimeError(
            f"PDF conversion failed.\n"
            f"  pdf2image error: {e1}\n"
            f"  PyMuPDF error:   {e2}\n"
            f"Install poppler or PyMuPDF:\n"
            f"  pip install PyMuPDF\n"
            f"  OR: pip install pdf2image (+ install poppler)"
        )
