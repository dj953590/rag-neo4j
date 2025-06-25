import os
import time

from src.engine.graph import GraphEngine, QueryParam
from src.llm.oai import gpt_4o_mini_complete, llama_3_3_70b_turbo, deepseek_distill_llama
from src.llm.oai import llama_3_3_70b_versatile
from dynaconf import settings
from src.docs.parser.pdf_processor import PDFProcessor
from pathlib2 import Path

from src.utils.utils import compute_mdhash_id
relationship = "gxo"
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


print(f"Querying document: {doc_name} with ID: {doc_id} at {pdf_path}")

os.environ["OPENAI_API_KEY"] = settings.get("OPENAI_API_KEY")
engine = GraphEngine(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,
    doc_id=doc_id,
    doc_name=doc_name,
)

while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == 'bye' or query.lower() == 'exit':
        break
    query_param = QueryParam(mode="naive", doc_id=doc_id)
    result, ids = engine.query(query, param=query_param)
    print("Query Results:")
    print(result)
    print("Chunk IDs used :")
    print(ids)