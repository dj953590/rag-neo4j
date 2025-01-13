import os
from pydantic import BaseModel, Field
from typing import List
from groq import Groq
import instructor
from dynaconf import settings

os.environ['OPENAI_API_KEY'] = settings.get('OPENAI_API_KEY', '')
os.environ['GROQ_API_KEY'] = settings.get('GROQ_API_KEY', '')
"""
This file contains description of how to use instructor classes with Groq and have structured output
"""

class Character(BaseModel):
    name: str
    fact: List[str] = Field(..., description="A list of facts about the subject")


client = Groq(
    api_key=os.environ.get('GROQ_API_KEY'),
)

client = instructor.from_groq(client, mode=instructor.Mode.TOOLS)

resp = client.chat.completions.create(
    model=f"llama-3.1-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": "Tell me about the company Tesla",
        }
    ],
    response_model=Character,
)
print(resp.model_dump_json(indent=2))
"""
{
  "name": "Tesla",
  "fact": [
    "electric vehicle manufacturer",
    "solar panel producer",
    "based in Palo Alto, California",
    "founded in 2003 by Elon Musk"
  ]
}
"""
