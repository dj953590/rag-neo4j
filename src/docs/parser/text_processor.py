from abc import ABC

from src.docs.parser.base_processor import BaseProcessor


class TXTProcessor(BaseProcessor, ABC):
    def extract_text(self):
        """
        Extract text from a TXT file.
        """
        with open(self.file_path, "r", encoding="utf-8") as file:
            text = file.read()
        return [{
            "type": "text",
            "text": text,
            "bounding_box": [0, 0, 0, 0],  # TXT does not provide bounding boxes
            "page": 1
        }]

    def extract_tables(self):
        """
        TXT files do not contain tables.
        """
        return []

    def extract_images(self, output_dir="extracted_images"):
        """
        TXT files do not contain images.
        """
        return []