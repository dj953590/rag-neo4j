import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chains import LLMChain
from dynaconf import settings
from langchain.globals import set_debug

set_debug(True)

# Set up your GROQ API key
api_groq_key = settings.GROQ_API_KEY

# Initialize the GROQ language model
llm = ChatGroq(temperature=0.0, model_name=f"llama-3.1-70b-versatile", api_key=api_groq_key)

# Define the response schemas
response_schemas = [
    ResponseSchema(name="summary", description="A brief summary of the topic"),
    ResponseSchema(name="key_points", description="List of key points about the topic"),
    ResponseSchema(name="example", description="An example related to the topic")
]

# Create the output parser
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Create the prompt template
prompt_template = ChatPromptTemplate.from_template(
    "Provide information about {topic}. " +
    "Please structure your response as follows:\n" +
    "{format_instructions}"
)

# Create the LLMChain
#chain = LLMChain(llm=llm, prompt=prompt_template)

# Create the chain using RunnableSequence
chain = (
        {"topic": RunnablePassthrough(), "format_instructions": lambda _: output_parser.get_format_instructions()}
        | prompt_template
        | llm
        | StrOutputParser()
)

# Define the topic
topic = "artificial intelligence"

# Generate the response

# response = chain.run(
# topic=topic,
# format_instructions=output_parser.get_format_instructions()
# )

response = chain.invoke(topic)
# Parse the response
parsed_response = output_parser.parse(response)

# Print the parsed response
print("Summary:", parsed_response["summary"])
print("\nKey Points:", parsed_response["key_points"])
print("\nExample:", parsed_response["example"])