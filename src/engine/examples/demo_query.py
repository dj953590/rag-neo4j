import os
from pathlib2 import Path
from src.engine.graph import GraphEngine, QueryParam
from src.utils.utils import compute_mdhash_id
from dynaconf import settings
from src.llm.oai import gpt_4o_mini_complete

os.environ["OPENAI_API_KEY"] = settings.get("OPENAI_API_KEY")

def list_pdfs(directory):
    return [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]

def select_pdf(directory):
    pdfs = list_pdfs(directory)
    if not pdfs:
        print("No PDF files found in the directory.")
        return None
    while True:
        print("\nAvailable PDF files:")
        for idx, pdf in enumerate(pdfs):
            print(f"{idx + 1}: {pdf}")
        choice = input("Select a PDF by number (or type 'exit' to quit): ")
        if choice.lower() in ['exit', 'quit']:
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(pdfs):
            return pdfs[int(choice) - 1]
        print("Invalid selection. Try again.")

def main():
    directory = input("Enter the directory containing PDF files: ").strip()
    if not os.path.isdir(directory):
        print("Directory does not exist.")
        return

    while True:
        pdf_file = select_pdf(directory)
        if not pdf_file:
            break
        doc_name = os.path.splitext(pdf_file)[0]
        doc_id = compute_mdhash_id(doc_name, "DOC-")
        pdf_path = Path(directory) / pdf_file
        print(f"\nQuerying document: {doc_name} with ID: {doc_id} at {pdf_path}")

        engine = GraphEngine(
            working_dir=directory,
            llm_model_func=gpt_4o_mini_complete,
            doc_id=doc_id,
            doc_name=doc_name,
        )

        while True:
            query = input("Enter your query (type 'back' to choose another file, 'exit' to quit): ")
            if query.lower() in ['exit', 'quit']:
                return
            if query.lower() == 'back':
                break
            query_param = QueryParam(mode="naive", doc_id=doc_id)
            result, ids, k_wds = engine.query(query, param=query_param)
            print("Query Results:")
            print(result)
            print("Chunk IDs used :")
            print(ids)
            print("Keywords used :")
            print(k_wds)

if __name__ == "__main__":
    main()