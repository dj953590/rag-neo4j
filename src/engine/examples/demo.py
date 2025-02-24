import os
from src.engine.graph import GraphEngine, QueryParam
from src.llm.oai import gpt_4o_mini_complete, llama_3_3_70b_turbo, deepseek_distill_llama
from src.llm.oai import llama_3_3_70b_versatile
from dynaconf import settings
from src.docs.parser.pdf_processor import PDFProcessor
from pathlib2 import Path
import json

from src.utils.utils import compute_mdhash_id
relationship = "amazon"
WORKING_DIR = "./" + relationship
doc_name = "citibank-" + relationship

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
pdf_path = (
        Path(__file__).parent / relationship / (doc_name + ".pdf")
)  # Replace with your PDF file path

output_path = Path(__file__).parent / relationship / (doc_name + ".json")  # Output JSON file
output_text_path = Path(__file__).parent / relationship / (doc_name + ".txt")  # Output JSON file

doc_id = compute_mdhash_id(doc_name, "DOC-")

print (f"Credit Agreement : {doc_id}")

os.environ["OPENAI_API_KEY"] = settings.get("OPENAI_API_KEY")
engine = GraphEngine(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,
    doc_id=doc_id,
    doc_name=doc_name,
)

extractor = PDFProcessor(file_path=pdf_path, output_path=output_path, output_text_path=output_text_path)
structured_data = extractor.markdown()
engine.insert(structured_data)


while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == 'bye' or query.lower() == 'exit':
        break
    query_param = QueryParam(mode="hybrid", doc_id=doc_id)
    result = engine.query(query, param=query_param)
    print(result)
