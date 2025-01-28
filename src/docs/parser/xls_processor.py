from abc import ABC

from openpyxl import load_workbook

from src.docs.parser.base_processor import BaseProcessor


class XLSProcessor(BaseProcessor, ABC):
    def extract_text(self):
        """
        Extract text from an XLS file (from all sheets).
        """
        wb = load_workbook(self.file_path)
        text_blocks = []
        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            for row in sheet.iter_rows(values_only=True):
                text_blocks.append({
                    "type": "text",
                    "text": "\t".join([str(cell) for cell in row]),
                    "bounding_box": [0, 0, 0, 0],  # XLS does not provide bounding boxes
                    "page": sheet_name
                })
        return text_blocks

    def extract_tables(self):
        """
        Extract tables from an XLS file (from all sheets).
        """
        wb = load_workbook(self.file_path)
        tables = []
        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            table_data = []
            for row in sheet.iter_rows(values_only=True):
                table_data.append([str(cell) for cell in row])
            tables.append({
                "type": "table",
                "data": table_data,
                "bounding_box": [0, 0, 0, 0],  # XLS does not provide bounding boxes
                "page": sheet_name
            })
        return tables

    def extract_images(self, output_dir="extracted_images"):
        """
        Extract images from an XLS file and perform OCR.
        """
        wb = load_workbook(self.file_path)
        images = []
        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            for image in sheet._images:
                image_data = image._data()
                image_path = self.save_images(image_data, output_dir, f"{sheet_name}_image_{len(images)}.png")

                # Perform OCR on the image
                ocr_text = self.perform_ocr(image_path)
                images.append({
                    "type": "image",
                    "path": image_path,
                    "text": ocr_text,
                    "bounding_box": [0, 0, 0, 0],  # XLS does not provide bounding boxes
                    "page": sheet_name
                })
        return images
