import instructor
import os
import json
from groq import Groq
from dynaconf import settings
from typing import List, Literal

from pydantic import Field, BaseModel

os.environ['OPENAI_API_KEY'] = settings.get('OPENAI_API_KEY', '')
os.environ['GROQ_API_KEY'] = settings.get('GROQ_API_KEY', '')
"""
This file contains description of how to use instructor classes with FY14 attributes and have structured Query plan
"""

client = Groq(
    api_key=os.environ.get('GROQ_API_KEY'),
)

client = instructor.from_groq(client, mode=instructor.Mode.TOOLS)


class Query(BaseModel):
    """Class representing a single question in a query plan."""

    id: int = Field(..., description="Unique id of the query")
    question: str = Field(
        ...,
        description="Question asked using a question answering system",
    )
    high_level_keywords: List[str] = Field(
        default_factory=list,
        description="High level keywords for the query",
    )
    low_level_keywords: List[str] = Field(
        default_factory=list,
        description="low level keywords for the query",
    )
    dependencies: List[int] = Field(
        default_factory=list,
        description="List of sub questions that need to be answered before asking this question",
    )
    node_type: Literal["SINGLE", "MERGE_MULTIPLE_RESPONSES"] = Field(
        default="SINGLE",
        description="Type of question, either a single question or a multi-question merge",
    )


class QueryPlan(BaseModel):
    """Container class representing a tree of questions to ask a question answering system."""

    query_graph: List[Query] = Field(
        ..., description="The query graph representing the plan"
    )

    def _dependencies(self, ids: List[int]) -> List[Query]:
        """Returns the dependencies of a query given their ids."""
        return [q for q in self.query_graph if q.id in ids]


Query.model_rebuild()
QueryPlan.model_rebuild()


def query_planner(question: str) -> QueryPlan:
    PLANNING_MODEL = f"llama-3.1-70b-versatile"
    messages = [
        {
            "role": "system",
            "content": "You are a world class query planning algorithm capable of breaking apart questions related to "
                       "credit agreements attributes (reported in FRY-14 reports) into its dependency queries such that"
                       " the answers can be used to inform the parent question. Do not"
                       "answer the questions, simply provide a correct compute graph with good specific questions to "
                       "ask and relevant dependencies. Before you call the function, think step-by-step to get a "
                       "better understanding of the problem. for each query identified provide high and low level keyowrds "
                       "Question will have name of attribute in FRY-14 report, "
                       "its description which provides details about how to extract the attribute from credit agreement"
                       " and rules for what attribute value should look like, its important that final extracted "
                       "value follows the rules",
        },
        {
            "role": "user",
            "content": f"Consider: {question}\nGenerate the correct query plan.",
        },
    ]

    root = client.chat.completions.create(
        model=PLANNING_MODEL,
        temperature=0,
        response_model=QueryPlan,
        messages=messages,
        max_tokens=1000,
    )
    return root


plan = query_planner(
    "Create query plan for "
    "Attribute: InterestRateIndex "
    "Description: For floating rate credit facilities list base interest rate using integer code. If obligor has an "
    "option, select the index actually in use. If the credit facility is fixed (Variability of current "
    "interest rates (Fixed, Floating, or Mixed) to maturity.) choose "
    "the integer for “Not applicable (Fixed)”. For credit facilities where the base interest rate is "
    "mixed, choose the integer for “Mixed.”"
    "Rules around the attribute: InterestRateIndex is a one of these values "
    "LIBOR, PRIME or Base, Treasury Index, Other, Not applicable (Fixed), Mixed"
)

# Convert the model instance to a dictionary and then to a JSON string
query_plan_json = plan.model_dump_json(indent=4)

# Print the query plan to the console
print(query_plan_json)

