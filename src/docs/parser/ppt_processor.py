from pptx import Presentation
from .base_processor import BaseProcessor


class PPTProcessor(BaseProcessor):
    def extract_text(self):
        """
        Extract text from a PPT file.
        """
        prs = Presentation(self.file_path)
        text_blocks = []
        for slide_num, slide in enumerate(prs.slides):
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text_blocks.append({
                        "type": "text",
                        "text": shape.text.strip(),
                        "bounding_box": [0, 0, 0, 0],  # PPT does not provide bounding boxes
                        "page": slide_num + 1
                    })
        return text_blocks

    def extract_tables(self):
        """
        Extract tables from a PPT file.
        """
        prs = Presentation(self.file_path)
        tables = []
        for slide_num, slide in enumerate(prs.slides):
            for shape in slide.shapes:
                if shape.has_table:
                    table_data = []
                    for row in shape.table.rows:
                        row_data = [cell.text for cell in row.cells]
                        table_data.append(row_data)
                    tables.append({
                        "type": "table",
                        "data": table_data,
                        "bounding_box": [0, 0, 0, 0],  # PPT does not provide bounding boxes
                        "page": slide_num + 1
                    })
        return tables

    def extract_images(self, output_dir="extracted_images"):
        """
        Extract images from a PPT file and perform OCR.
        """
        prs = Presentation(self.file_path)
        images = []
        for slide_num, slide in enumerate(prs.slides):
            for shape in slide.shapes:
                if hasattr(shape, "image"):
                    image_data = shape.image.blob
                    image_ext = shape.image.ext
                    image_path = self.save_images(image_data, output_dir,
                                                  f"slide_{slide_num}_image_{len(images)}.{image_ext}")

                    # Perform OCR on the image
                    ocr_text = self.perform_ocr(image_path)
                    images.append({
                        "type": "image",
                        "path": image_path,
                        "text": ocr_text,
                        "bounding_box": [0, 0, 0, 0],  # PPT does not provide bounding boxes
                        "page": slide_num + 1
                    })
        return images
