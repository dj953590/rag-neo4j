from abc import ABC

import pymupdf  # PyMuPDF
from PIL import Image
import io
from pathlib2 import Path
from src.docs.parser.base_processor import BaseProcessor
from src.docs.parser.helpers.pdf.pdf_md import to_markdown


class PDFProcessor(BaseProcessor, ABC):
    def __init__(self, file_path: str, ):
        super().__init__(file_path)
        self.doc = pymupdf.open(self.file_path)

    def markdown(self):

        md_string = to_markdown(self.doc, page_chunks=True, extract_words=True, margins=(0, 20, 0, 20))
        return md_string


def main(file_path):
    """
    Main function to extract text, images, and tables from a PDF.
    """
    extractor = PDFProcessor(file_path)
    structured_data = extractor.markdown()
    print(structured_data)

if __name__ == "__main__":
    pdf_path = (
            Path(__file__).parent / "docs" / "citibank-caterpillar.pdf"
    )  # Replace with your PDF file path
    main(pdf_path)
