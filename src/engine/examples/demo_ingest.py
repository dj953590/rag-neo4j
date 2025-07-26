import os
import time
from pathlib2 import Path

from src.engine.graph import GraphEngine
from src.llm.oai import gpt_4o_mini_complete
from dynaconf import settings
from src.docs.parser.pdf_processor import PDFProcessor
from src.utils.utils import compute_mdhash_id

parent_folder = "amazon"
WORKING_DIR = "./" + parent_folder

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

pdf_dir = Path(__file__).parent / parent_folder
pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]

os.environ["OPENAI_API_KEY"] = settings.get("OPENAI_API_KEY")

for pdf_file in pdf_files:
    doc_name = os.path.splitext(pdf_file)[0]
    pdf_path = pdf_dir / pdf_file
    output_path = pdf_dir / (doc_name + ".json")
    output_text_path = pdf_dir / (doc_name + ".txt")
    doc_id = compute_mdhash_id(doc_name, "DOC-")

    print(f"Processing document: {doc_name} with ID: {doc_id} at {pdf_path}")
    start_time = time.time()

    engine = GraphEngine(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,
        doc_id=doc_id,
        doc_name=doc_name,
    )

    extractor = PDFProcessor(file_path=pdf_path, output_path=output_path, output_text_path=output_text_path)
    structured_data = extractor.markdown()
    engine.insert(structured_data, mode="LOCAL,VECTOR")

    end_time = time.time()
    print(f"Total processing time for {doc_name}: {end_time - start_time:.2f} seconds")