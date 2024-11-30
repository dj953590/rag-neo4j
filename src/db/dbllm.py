from langchain_community.utilities.sql_database import SQLDatabase
from langchain_ollama import ChatOllama
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_sql_query_chain
from langchain.chains.sql_database import query
from dynaconf import settings
from langchain_groq import ChatGroq
import os

os.environ['OPENAI_API_KEY'] = settings.get('OPENAI_API_KEY', '')
os.environ['GROQ_API_KEY'] = settings.get('GROQ_API_KEY', '')


class DBLLM:
    def __init__(self, db_url: str = None, llm_url: str = "http://localhost:11434", verbose: bool = True):
        """
        Initialize the database connection and LLM.

        :param db_url: PostgreSQL database URL in the format: postgresql://user:password@host:port/database
        :param llm_url: The URL where the Ollama Llama 3.1 model is hosted.
        :param verbose: Whether to enable verbose mode for debugging.
        """
        if db_url is None:
            db_url = settings.get('DATABASE_URL')

        self.db = SQLDatabase.from_uri(db_url)

        #self.llm = ChatOllama(base_url=llm_url, model="sqlcoder:latest", temperature=0.0)
        #self.llm = OpenAI(temperature=0.0)

        api_key = settings.get('GROQ_API_KEY')
        self.llm = ChatGroq(temperature=0.0, model_name=f"llama-3.1-70b-versatile", api_key=api_key)

        template = """
        Based on the table schema below write a SQL query that would answer the user's quertion:
        {schema}
        
        Question: {question}
        SQL Query
        """
        self.prompt = ChatPromptTemplate.from_template(template)

        self.schema = self.db.get_table_info()

    def run_query(self, query: str) -> str:
        """
        Execute a SQL query directly.

        :param query: SQL query string to execute.
        :return: Result of the query.
        """
        try:
            result = self.db.run(query)
            return result
        except Exception as e:
            return f"Error executing query: {e}"

    def get_schema(self, _):
        return self.schema

    def query_with_llm(self, natural_language_query: str) -> str:
        """
        Use LLM to generate and execute a SQL query based on natural language input.

        :param natural_language_query: The natural language query to interpret.
        :return: Result of the interpreted and executed query.
        """
        try:
            sql_chain = create_sql_query_chain(self.llm, self.db)
            sql_chain.get_prompts()[0].pretty_print()
            response = sql_chain.invoke({"question": natural_language_query})
            print(sql_chain)
            return response
        except Exception as e:
            return f"Error in LLM query: {e}"

    def close_connection(self):
        """
        Close the database connection.
        """
        try:
            self.db._engine.dispose()
            return "Database connection closed."
        except Exception as e:
            return f"Error closing connection: {e}"


# Example usage
if __name__ == "__main__":

    db = DBLLM()

    # Example of direct query
    print(db.run_query("SELECT * FROM customer LIMIT 5;"))

    # Example of LLM-assisted query
    print(db.query_with_llm("Provide me all the names, invoice id and total amount due of customers who live in country Brazil? Please do not put any limits clause " +
                            "Also provide me another query which totals the invoices for brazil customer"))

    #complex query

    print(db.query_with_llm("Provide me album title, name of playlist and artist name bought by customers in Brazil ? Please do not put any Limits clause and use the WITH clause "))

    # Close the connection
    print(db.close_connection())
