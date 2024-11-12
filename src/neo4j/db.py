import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access variables
database_url = os.getenv("DB_URI")
user = os.getenv("DB_USR")
pwd = os.getenv("DB_PWD")

driver = GraphDatabase.driver(database_url, auth=(user, pwd))


# Example function to run a Cypher query
def test_connection():
    with driver.session() as session:
        # Run a simple query to return the Neo4j version
        result = session.run("RETURN 'Connection successful!' AS message")
        for record in result:
            print(record["message"])


# Run the test function
test_connection()

# Close the driver when done
driver.close()
