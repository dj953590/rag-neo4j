from abc import ABC

import fitz  # PyMuPDF
import json
from PIL import Image
import io
import pdfplumber  # For table extraction
import cv2
import numpy as np
from onnxtr.io import DocumentFile
from onnxtr.models import ocr_predictor, EngineConfig
from pathlib2 import Path
from src.docs.parser.base_processor import BaseProcessor


class PDFProcessor(BaseProcessor, ABC):

    def text_and_images(self):
        """
        Extract text, images, and tables from the PDF.
        """
        structured_data = []

        for page_num, page in enumerate(self.doc):
            page_data = {"page_number": page_num + 1, "content": []}

            # Extract text blocks with font information
            text_blocks = page.get_text("dict")["blocks"]
            for block in text_blocks:
                if block["type"] == 0:  # Text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"]
                            font_size = span["size"]
                            is_heading = self.is_heading(span)
                            is_bullet = self.is_bullet_point(span)

                            # Add metadata for headings and bullets
                            metadata = {
                                "type": "text",
                                "text": text.strip(),
                                "bounding_box": [
                                    span["bbox"][0],
                                    span["bbox"][1],
                                    span["bbox"][2],
                                    span["bbox"][3],
                                ],
                                "page": page_num + 1,
                                "is_heading": is_heading,
                                "is_bullet": is_bullet,
                                "font_size": font_size,
                            }
                            page_data["content"].append(metadata)
            # Extract images and perform OCR
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = self.doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))

                # Preprocess the image
                # image = self.preprocess_image(image)

                # Perform OCR on the image using onnxtr
                ocr_result = self.perform_ocr_with_onnxtr(image)
                ocr_text = " ".join(
                    word.value
                    for block in ocr_result.pages[0].blocks
                    for line in block.lines
                    for word in line.words
                )

                page_data["content"].append(
                    {
                        "type": "ocr_text",
                        "text": ocr_text.strip(),
                        "bounding_box": [
                            0,
                            0,
                            image.width,
                            image.height,
                        ],  # Full image bounding box
                        "page": page_num + 1,
                        "is_heading": False,
                        "is_bullet": False,
                        "font_size": None,  # OCR does not provide font size information
                    }
                )

            structured_data.append(page_data)
        return structured_data


def main(pdf_path, output_path):
    """
    Main function to extract text, images, and tables from a PDF.
    """
    extractor = PDFProcessor(file_path=pdf_path, output_path=output_path)
    structured_data = extractor.process()
    print(f"Structured data saved to {output_path} in the json format \n {structured_data}")


if __name__ == "__main__":
    pdf_path = (
            Path(__file__).parent / "docs" / "neo4j.pdf"
    )  # Replace with your PDF file path
    output_path = Path(__file__).parent / "docs" / "neo4j.json"  # Output JSON file
    main(pdf_path, output_path)
