import networkx as nx
from neo4j import (
    GraphDatabase,
)
from pathlib2 import Path

from src.utils.log import logger
from dynaconf import settings


# GraphML Loader
class GraphMLLoader:
    def __init__(self):
        self.G = None
        uri = settings.get('NEO4j_URI', None)
        user = settings.get('NEO4J_USERNAME', None)
        pwd = settings.get('NEO4J_PASSWORD', None)
        self.driver = GraphDatabase.driver(uri, auth=(user, pwd))

    def close(self):
        self.driver.close()

    # Insert nodes
    def create_nodes(self, tx):
        for node, data in self.G.nodes(data=True):
            query = "MERGE (n:Entity {id: $id}) SET n += $properties"
            tx.run(query, id=node, properties=data)

    # Insert relationships
    def create_relationships(self, tx):
        for u, v, data in self.G.edges(data=True):
            query = """
            MATCH (a:Entity {id: $source})
            MATCH (b:Entity {id: $target})
            MERGE (a)-[r:CONNECTED_TO]->(b)
            SET r += $properties
            """
            tx.run(query, source=u, target=v, properties=data)

    def store_graph(self):
        with self.driver.session() as session:
            session.execute_write(self.create_nodes)
            session.execute_write(self.create_relationships)

    def load_graphml(self, file_name):
        """
        Load a GraphML file into Neo4j using APOC.
        """
        self.G = nx.read_graphml(file_name)


if __name__ == "__main__":
    # Initialize the loader
    relationship = "amazon"
    doc_name = "citibank-" + relationship
    graphml_loader = GraphMLLoader()
    file_loc = (
            Path(
                __file__).parent.parent.parent / 'engine' / 'examples' / relationship / "graph_chunk_entity_relation.graphml"
    )  # Replace with your PDF file path
    # Load the GraphML file
    neo4j_file_path = str(file_loc)
    graphml_loader.load_graphml(neo4j_file_path)
    # Store the graph
    graphml_loader.store_graph()
    # Close connection
    graphml_loader.close()
