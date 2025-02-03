from abc import ABC, abstractmethod
import os
import json
import io

from PIL.Image import Image
from onnxtr.io import DocumentFile
from onnxtr.models import ocr_predictor, EngineConfig
import cv2
import numpy as np


class BaseProcessor(ABC):
    def __init__(self, file_path, output_path=None, output_text_path=None):
        self.file_path = file_path
        self.output_path = output_path
        self.output_text_path = output_text_path
        # Initialize the OCR model
        self.ocr_model = ocr_predictor(
            det_arch="fast_base",  # detection architecture
            reco_arch="vitstr_base",  # recognition architecture
            det_bs=2,  # detection batch size
            reco_bs=512,  # recognition batch size
            # Document related parameters
            assume_straight_pages=True,
            # set to `False` if the pages are not straight (rotation, perspective, etc.) (default: True)
            straighten_pages=False,
            # set to `True` if the pages should be straightened before final processing (default: False)
            export_as_straight_boxes=False,
            # set to `True` if the boxes should be exported as if the pages were straight (default: False)
            # Preprocessing related parameters
            preserve_aspect_ratio=True,  # set to `False` if the aspect ratio should not be preserved (default: True)
            symmetric_pad=True,  # set to `False` to disable symmetric padding (default: True)
            # Additional parameters - meta information
            detect_orientation=False,
            # set to `True` if the orientation of the pages should be detected (default: False)
            detect_language=False,  # set to `True` if the language of the pages should be detected (default: False)
            # Orientation specific parameters in combination with `assume_straight_pages=False` and/or
            # `straighten_pages=True`
            disable_crop_orientation=False,
            # set to `True` if the crop orientation classification should be disabled (default: False)
            disable_page_orientation=False,
            # set to `True` if the general page orientation classification should be disabled (default: False)
            # DocumentBuilder specific parameters
            resolve_lines=True,  # whether words should be automatically grouped into lines (default: True)
            resolve_blocks=False,  # whether lines should be automatically grouped into blocks (default: False)
            paragraph_break=0.035,  # relative length of the minimum space separating paragraphs (default: 0.035)
            # OnnxTR specific parameters NOTE: 8-Bit quantized models are not available for FAST detection models and
            # can in general lead to poorer accuracy
            load_in_8_bit=False,
            # set to `True` to load 8-bit quantized models instead of the full precision onces (default: False)
            # Advanced engine configuration options
            det_engine_cfg=EngineConfig(),
            # detection model engine configuration (default: internal predefined configuration)
            reco_engine_cfg=EngineConfig(),
            # recognition model engine configuration (default: internal predefined configuration)
            clf_engine_cfg=EngineConfig(),
            # classification (orientation) model engine configuration (default: internal predefined configuration)
        )

    def preprocess_image(self, image):
        """
        Preprocess the image for better OCR accuracy.
        """
        # Convert to grayscale
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        # Apply thresholding (binarization)
        _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # Remove noise
        image = cv2.medianBlur(image, 3)
        return Image.fromarray(image)

    def is_heading(self, text_block):
        """
        Detect if a text block is a heading based on font size or style.
        """

        # Example: Check if the font size is larger than a threshold
        font_size = text_block["size"]
        if font_size > 12:  # Adjust the threshold as needed
            return True

        # Example: Check if the text is in uppercase
        if text_block["text"].isupper():
            return True

        return False

    def is_bullet_point(self, text_block):
        """
        Detect if a text block is a bullet point.
        """
        text = text_block["text"].strip()
        # Check if the text starts with a bullet symbol
        if text.startswith(("•", "-", "*")):
            return True
        return False

    def perform_ocr(self, image):
        """
        Perform OCR on the image using the ONNX OCR engine.
        """
        # Perform OCR using onnxtr
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes = image_bytes.getvalue()

        document = DocumentFile.from_images([image_bytes])

        # Perform OCR using onnxtr
        ocr_result = self.ocr_model(document)
        return ocr_result

    @abstractmethod
    def text_and_images(self):
        """
        Extract text from the file.
        """
        pass



    def save_images(self, image_data, output_dir, image_name):
        """
        Save images to the output directory.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image_path = os.path.join(output_dir, image_name)
        with open(image_path, "wb") as img_file:
            img_file.write(image_data)
        return image_path

    def save_structured_data(self, structured_data):
        """
        Save the structured data to a JSON file.
        """
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(structured_data, f, indent=4, ensure_ascii=False)

    def extract_text_from_structured_data(self, structured_data):
        extracted_text = []
        for page in structured_data:
            for content in page["content"]:
                if content["type"] == "text" or content["type"] == "ocr_text":
                    extracted_text.append(content["text"])
        return "\n".join(extracted_text)

    def process(self):
        """
        Process the file and return structured data in JSON format.
        """
        structured_data = self.text_and_images()
        if self.output_path:
            self.save_structured_data(structured_data)

        if self.output_text_path:
            extracted_text = self.extract_text_from_structured_data(structured_data)
            with open(self.output_text_path, "w", encoding="utf-8") as f:
                f.write(extracted_text)

        return structured_data
