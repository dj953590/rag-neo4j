import os
from src.graphrag.g_rag import TKGS, QueryParam
from src.llm.oai import gpt_4o_mini_complete
from dynaconf import settings

WORKING_DIR = "./amazon"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

os.environ["OPENAI_API_KEY"] = settings.get("OPENAI_API_KEY")
rag = TKGS(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,
    # llm_model_func=gpt_4o_complete
)


with open("./Citibank-Amazon.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())




# Perform local search
print(
    rag.query("What is Origination date or executed date for the Loan ?", param=QueryParam(mode="hybrid"))
)

print(
    rag.query("What kind of credit line Amazon has ?", param=QueryParam(mode="hybrid"))
)

print(
    rag.query("is this a syndicated or bilateral loan ? Syndicated loan is where we more than one lenders whereas bilateral loan is with only one lender", param=QueryParam(mode="hybrid"))
)



"""
# Perform global search
print(
    rag.query("What a MarketCap of Tata Give it in numbers with appropriate symbol ?", param=QueryParam(mode="hybrid"))
)

# Perform hybrid search
print(
    rag.query("Who was the author of this document this is not the same as who this document is meant for its the analyst who wrote this document ?", param=QueryParam(mode="hybrid"))
)

print(
    rag.query("Summarize TTL Company Description, Investment Strategy, Valuation and Risks ?", param=QueryParam(mode="hybrid"))
)
"""