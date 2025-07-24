import os
import time

from src.engine.graph import GraphEngine, QueryParam
from src.llm.oai import gpt_4o_mini_complete, llama_3_3_70b_turbo, deepseek_distill_llama
from src.llm.oai import llama_3_3_70b_versatile
from dynaconf import settings
from src.docs.parser.pdf_processor import PDFProcessor
from pathlib2 import Path

from src.storage.db.base import ClassifyParam
from src.utils.utils import compute_mdhash_id

os.environ["OPENAI_API_KEY"] = settings.get("OPENAI_API_KEY")

while True:
    doc_name = input("Enter document name to classify (or type 'exit' to quit): ")
    if doc_name.lower() == 'bye' or doc_name.lower() == 'exit':
        break
    doc_id = compute_mdhash_id(doc_name, "DOC-")
    classify_param = ClassifyParam(
        mode="basic",
        pages=5,
        doc_id=doc_id,
    )
    engine = GraphEngine(
        llm_model_func=gpt_4o_mini_complete,
        doc_id=doc_id,
        doc_name=doc_name,
    )
    response= engine.classify_document(param=classify_param)
    print("Classification Results:")
    print(response)
