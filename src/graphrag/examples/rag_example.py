import os
from src.graphrag.g_rag import TKGS, QueryParam
from src.llm.oai import gpt_4o_mini_complete
from dynaconf import settings
from src.docs.parser.pdf_processor import PDFProcessor
from pathlib2 import Path

WORKING_DIR = "./amazon"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

os.environ["OPENAI_API_KEY"] = settings.get("OPENAI_API_KEY")
rag = TKGS(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,
    # llm_model_func=gpt_4o_complete
)
pdf_path = (
        Path(__file__).parent / "amazon" / "citibank-amazon.pdf"
)  # Replace with your PDF file path
output_path = Path(__file__).parent / "amazon" / "citibank-amazon.json"  # Output JSON file
output_text_path = Path(__file__).parent / "amazon" / "citibank-amazon.txt"  # Output JSON file

extractor = PDFProcessor(file_path=pdf_path, output_path=output_path, output_text_path=output_text_path)
structured_data = extractor.process()

with open(output_text_path, "r", encoding="utf-8") as f:
    rag.insert(f.read())

# Perform local search
"""
query = "What kind of credit line GXO has with Citibank and provide the exact amount ?"
print(query)
print(
    rag.query(query, param=QueryParam(mode="hybrid"))
)
query = "is this a syndicated or bilateral loan ? Syndicated loan is where we more than one lenders whereas bilateral loan is with only one lender"
print(query)

print(
    rag.query(query, param=QueryParam(mode="hybrid"))
)

query = "how much amount in dollars was associated with the credit agreement ?"
print(query)

print(
    rag.query(query, param=QueryParam(mode="hybrid"))
)
"""
while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    result = rag.query(query, param=QueryParam(mode="hybrid"))
    print(result)
