import os

from src.engine.graph import GraphEngine
from src.llm.oai import gpt_4o_mini_complete
from dynaconf import settings


from src.storage.db.base import ClassifyParam
from src.utils.utils import compute_mdhash_id

os.environ["OPENAI_API_KEY"] = settings.get("OPENAI_API_KEY")

while True:
    doc_name = input("Enter document name to classify (or type 'exit' to quit): ")
    if doc_name.lower() == 'bye' or doc_name.lower() == 'exit':
        break
    doc_id = compute_mdhash_id(doc_name, "DOC-")
    print(f"Document ID: {doc_id}")
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
