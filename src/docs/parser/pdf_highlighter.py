import fitz  # PyMuPDF
import tempfile
import os
import webbrowser
from typing import List, Tuple

from pathlib2 import Path


class PDFHighlighter:
    def __init__(self, pdf_bytes: bytes):
        """
        Initialize with PDF bytes.

        Args:
            pdf_bytes: PDF content as bytes
        """
        self.doc = fitz.open("pdf", pdf_bytes)

    def highlight_chunk(
            self,
            page_start: int,
            first_word_bbox: Tuple[float, float, float, float],
            page_end: int,
            last_word_bbox: Tuple[float, float, float, float]
    ) -> bytes:
        """
        Highlight a chunk in the PDF from first to last word.

        Args:
            page_start: Page number of the first word (0-indexed)
            first_word_bbox: Bounding box of first word (x0, y0, x1, y1)
            page_end: Page number of the last word (0-indexed)
            last_word_bbox: Bounding box of last word (x0, y0, x1, y1)

        Returns:
            Modified PDF as bytes
        """
        # Highlight the entire area from first to last word
        for page_num in range(page_start, page_end + 1):
            page = self.doc.load_page(page_num)

            if page_num == page_start and page_num == page_end:
                # Single page: highlight from first word to last word
                rect = fitz.Rect(first_word_bbox[0], first_word_bbox[1],
                                 last_word_bbox[2], last_word_bbox[3])
                page.add_highlight_annot(rect)

            elif page_num == page_start:
                # First page: highlight from first word to end of page
                page_rect = page.rect
                rect = fitz.Rect(first_word_bbox[0], first_word_bbox[1],
                                 page_rect.width, page_rect.height)
                page.add_highlight_annot(rect)

            elif page_num == page_end:
                # Last page: highlight from start to last word
                rect = fitz.Rect(0, 0, last_word_bbox[2], last_word_bbox[3])
                page.add_highlight_annot(rect)

            else:
                # Intermediate pages: highlight entire page
                page.add_highlight_annot(page.rect)

        return self.doc.tobytes()

    def display_in_browser(self, pdf_bytes: bytes):
        """
        Save PDF to temp file and open in browser.

        Args:
            pdf_bytes: PDF content as bytes
        """
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        # Open in default browser
        webbrowser.open(f"file://{tmp_path}")

        # Clean up after browser opens
        # Delay deletion to ensure browser opens the file
        print(f"Temporary file created at: {tmp_path}")
        input("Press Enter after the file is opened in the browser to delete it...")
        os.unlink(tmp_path)
        print("Temporary file deleted.")

# Example usage
if __name__ == "__main__":
    # Load your PDF bytes (replace with actual PDF loading)
    pdf_path = (
            Path(__file__).parent / "docs" / "citibank-caterpillar.pdf"
    )  # Replace with your PDF file path
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    highlighter = PDFHighlighter(pdf_bytes)

    # Example positions (replace with actual positions from your chunks)
    # Format: (x0, y0, x1, y1)
    first_word_bbox = (50, 100, 100, 110)  # Example position
    last_word_bbox = (200, 300, 250, 310)   # Example position
    page_start = 3  # First page (0-indexed)
    page_end = 3    # Same page

    # Highlight the chunk
    highlighted_pdf = highlighter.highlight_chunk(
        page_start, first_word_bbox, page_end, last_word_bbox
    )

    # Display in browser
    highlighter.display_in_browser(highlighted_pdf)