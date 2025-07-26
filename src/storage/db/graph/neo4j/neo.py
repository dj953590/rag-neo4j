import asyncio
import os
from dataclasses import dataclass
from typing import Any, Union, Tuple, List, Dict
import inspect
from src.utils.log import logger
from dynaconf import settings
from src.storage.db.base import BaseGraphStorage, QueryParam

from neo4j import (
    AsyncGraphDatabase,
    exceptions as neo4jExceptions,
    AsyncDriver,
    AsyncManagedTransaction,
)

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.utils.utils import escape_cypher_properties, escape_cypher_node


@dataclass
class Neo4JStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name):
        print("no preloading of graph with neo4j in production")

    def __init__(self, namespace, global_config, embedding_func):
        super().__init__(
            namespace=namespace,
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self._driver = None
        self._driver_lock = asyncio.Lock()
        uri = settings.get('NEO4j_URI', None)
        user = settings.get('NEO4J_USERNAME', None)
        pwd = settings.get('NEO4J_PASSWORD', None)
        max_pool_size = settings.get('NEO4J_MAX_POOL_SIZE', 200)
        max_connect_timeout = settings.get('NEO4J_MAX_CONNECT_TIMEOUT', 120)
        self._database = settings.get('NEO4J_DATABASE', "neo4j")
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(
            uri, auth=(user, pwd),
            max_connection_pool_size=max_pool_size,
            connection_timeout=max_connect_timeout,
        )

    def __post_init__(self):
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def close(self):
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def __aexit__(self, exc_type, exc, tb):
        if self._driver:
            await self._driver.close()

    async def index_done_callback(self):
        print("KG successfully indexed.")

    def _validate_doc_id(self, param: QueryParam) -> bool:
        """
        Validate if doc_id is present in QueryParam.
        Logs a warning if doc_id is missing.
        """
        if not param or not hasattr(param, "doc_id") or not param.doc_id:
            logger.warning("doc_id is missing in QueryParam. Ensure doc_id is provided for proper querying.")
            return False
        return True

    async def has_node(self, node_id: str, param: QueryParam = None) -> bool:
        """
        Check if a node exists in the graph, filtered by doc_id if provided.
        """
        if not self._validate_doc_id(param):
            return False  # Return False if doc_id is missing

        entity_name_label = node_id.strip('"')
        doc_id = param.doc_id if param else None

        async with self._driver.session(database=self._database) as session:
            query = (
                f"MATCH (n:`{entity_name_label}` {{doc_id: $doc_id}}) "
                "RETURN count(n) > 0 AS node_exists"
            )
            result = await session.run(query, doc_id=doc_id)
            single_result = await result.single()
            logger.debug(
                f'{inspect.currentframe().f_code.co_name}:query:{query}:result:{single_result["node_exists"]}'
            )
            return single_result["node_exists"]

    async def has_edge(self, source_node_id: str, target_node_id: str, param: QueryParam = None) -> bool:
        """
        Check if an edge exists between two nodes, filtered by doc_id if provided.
        """
        if not self._validate_doc_id(param):
            return False  # Return False if doc_id is missing

        entity_name_label_source = source_node_id.strip('"')
        entity_name_label_target = target_node_id.strip('"')
        doc_id = param.doc_id if param else None

        async with self._driver.session(database=self._database) as session:
            query = (
                f"MATCH (a:`{entity_name_label_source}` {{doc_id: $doc_id}})-[r]-(b:`{entity_name_label_target}` {{doc_id: $doc_id}}) "
                "RETURN COUNT(r) > 0 AS edgeExists"
            )
            result = await session.run(query, doc_id=doc_id)
            single_result = await result.single()
            logger.debug(
                f'{inspect.currentframe().f_code.co_name}:query:{query}:result:{single_result["edgeExists"]}'
            )
            return single_result["edgeExists"]

    async def get_node(self, node_id: str, param: QueryParam = None) -> Union[dict, None]:
        """
        Retrieve a node by its ID, filtered by doc_id if provided.
        """
        if not self._validate_doc_id(param):
            return None  # Return None if doc_id is missing

        entity_name_label = node_id.strip('"')
        doc_id = param.doc_id if param else None

        async with self._driver.session(database=self._database) as session:
            query = f"MATCH (n:`{entity_name_label}` {{doc_id: $doc_id}}) RETURN n"
            result = await session.run(query, doc_id=doc_id)
            record = await result.single()
            if record:
                node = record["n"]
                node_dict = dict(node)
                logger.debug(
                    f"{inspect.currentframe().f_code.co_name}: query: {query}, result: {node_dict}"
                )
                return node_dict
            return None

    async def node_degree(self, node_id: str, param: QueryParam = None) -> int:
        """
        Calculate the degree of a node, filtered by doc_id if provided.
        """
        if not self._validate_doc_id(param):
            return 0  # Return 0 if doc_id is missing

        entity_name_label = node_id.strip('"')
        doc_id = param.doc_id if param else None

        async with self._driver.session(database=self._database) as session:
            query = f"""
                MATCH (n:`{entity_name_label}` {{doc_id: $doc_id}})
                RETURN COUNT{{ (n)--() }} AS totalEdgeCount
            """
            result = await session.run(query, doc_id=doc_id)
            record = await result.single()
            if record:
                edge_count = record["totalEdgeCount"]
                logger.debug(
                    f"{inspect.currentframe().f_code.co_name}:query:{query}:result:{edge_count}"
                )
                return edge_count
            else:
                return 0  # Return 0 if no edges are found

    async def edge_degree(self, src_id: str, tgt_id: str, param: QueryParam = None) -> int:
        entity_name_label_source = src_id.strip('"')
        entity_name_label_target = tgt_id.strip('"')
        src_degree = await self.node_degree(entity_name_label_source, param=param)
        trg_degree = await self.node_degree(entity_name_label_target, param=param)

        # Convert None to 0 for addition
        src_degree = 0 if src_degree is None else src_degree
        trg_degree = 0 if trg_degree is None else trg_degree

        degrees = int(src_degree) + int(trg_degree)
        logger.debug(
            f"{inspect.currentframe().f_code.co_name}:query:src_Degree+trg_degree:result:{degrees}"
        )
        return degrees

    async def get_edge(
            self, source_node_id: str, target_node_id: str, param: QueryParam = None
    ) -> Union[dict, None]:
        """
        Retrieve an edge between two nodes, filtered by doc_id if provided.
        """
        if not self._validate_doc_id(param):
            return None  # Return None if doc_id is missing

        entity_name_label_source = source_node_id.strip('"')
        entity_name_label_target = target_node_id.strip('"')
        doc_id = param.doc_id if param else None

        async with self._driver.session(database=self._database) as session:
            query = f"""
            MATCH (start:`{entity_name_label_source}` {{doc_id: $doc_id}})-[r]->(end:`{entity_name_label_target}` {{doc_id: $doc_id}})
            RETURN properties(r) as edge_properties
            LIMIT 1
            """
            result = await session.run(query, doc_id=doc_id)
            record = await result.single()
            if record:
                result = dict(record["edge_properties"])
                logger.debug(
                    f"{inspect.currentframe().f_code.co_name}:query:{query}:result:{result}"
                )
                return result
            else:
                return None

    async def get_node_edges(self, source_node_id: str, param: QueryParam = None) -> List[Tuple[str, str]]:
        """
        Retrieve all edges connected to a node, filtered by doc_id if provided.
        """
        if not self._validate_doc_id(param):
            return []  # Return an empty list if doc_id is missing

        node_label = source_node_id.strip('"')
        doc_id = param.doc_id if param else None

        query = f"""
            MATCH (n:`{node_label}` {{doc_id: $doc_id}})
            OPTIONAL MATCH (n)-[r]-(connected {{doc_id: $doc_id}})
            RETURN n, r, connected
        """
        async with self._driver.session(database=self._database) as session:
            results = await session.run(query, doc_id=doc_id)
            edges = []
            async for record in results:
                source_node = record["n"]
                connected_node = record["connected"]

                source_label = (
                    list(source_node.labels)[0] if source_node.labels else None
                )
                target_label = (
                    list(connected_node.labels)[0]
                    if connected_node and connected_node.labels
                    else None
                )

                if source_label and target_label:
                    edges.append((source_label, target_label))

            return edges

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                    neo4jExceptions.ServiceUnavailable,
                    neo4jExceptions.TransientError,
                    neo4jExceptions.WriteServiceUnavailable,
                    neo4jExceptions.ClientError,
            )
        ),
    )
    async def upsert_node(self, node_id: str, node_data: Dict[str, Any]):
        """
        Upsert a node in the Neo4j database.

        Args:
            node_id: The unique identifier for the node (used as label)
            node_data: Dictionary of node properties
        """

        if not node_id:
            logger.info(
                f"Node label is empty. Cannot upsert node for data: {node_data}"
            )
            return
        label = escape_cypher_node(node_id)
        properties = escape_cypher_properties(node_data)

        async def _do_upsert(tx: AsyncManagedTransaction):
            query = f"""
            MERGE (n:`{label}`)
            SET n += $properties
            """
            await tx.run(query, properties=properties)
            logger.debug(
                f"Upserted node with label '{label}' and properties: {properties}"
            )

        try:
            async with self._driver.session(database=self._database) as session:
                await session.execute_write(_do_upsert)
        except Exception as e:
            logger.error(f"Error during upsert: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                    neo4jExceptions.ServiceUnavailable,
                    neo4jExceptions.TransientError,
                    neo4jExceptions.WriteServiceUnavailable,
            )
        ),
    )
    async def upsert_edge(
            self, source_node_id: str, target_node_id: str, edge_data: Dict[str, Any]
    ):
        """
        Upsert an edge and its properties between two nodes identified by their labels.

        Args:
            source_node_id (str): Label of the source node (used as identifier)
            target_node_id (str): Label of the target node (used as identifier)
            edge_data (dict): Dictionary of properties to set on the edge
        """

        if not source_node_id or not target_node_id:
            logger.info(
                f"Source or target node label is empty. Cannot upsert edge for data: {edge_data}"
            )
            return
        source_node_label = escape_cypher_node(source_node_id)
        target_node_label = escape_cypher_node(target_node_id)
        edge_properties = escape_cypher_properties(edge_data)

        async def _do_upsert_edge(tx: AsyncManagedTransaction):
            query = f"""
            MATCH (source:`{source_node_label}`)
            WITH source
            MATCH (target:`{target_node_label}`)
            MERGE (source)-[r:DIRECTED]->(target)
            SET r += $properties
            RETURN r
            """
            await tx.run(query, properties=edge_properties)
            logger.debug(
                f"Upserted edge from '{source_node_label}' to '{target_node_label}' with properties: {edge_properties}"
            )

        try:
            async with self._driver.session(database=self._database) as session:
                await session.execute_write(_do_upsert_edge)
        except Exception as e:
            logger.error(f"Error during edge upsert: {str(e)}")
            raise

    async def get_node_count(self, ) -> int:
        """
        Retrieve the count of nodes in the graph, filtered by doc_id if provided.
        """
        try:
            async with self._driver.session(database=self._database) as session:
                result = await session.run("MATCH (n) RETURN count(n) as totalNodes")

                # Check if result itself is None
                if result is None:
                    logger.error("session.run returned None. Please check your driver connection and configuration.")
                    return 0

                record = await result.single()

                # Check if single() returns None (e.g., no record was produced)
                if record is None:
                    logger.info("No record returned; assuming count is 0.")
                    return 0

                total_nodes = record.get("totalNodes", 0)
                return total_nodes
        except Exception as e:
            return 0

    async def get_database(self):
        async with self._driver.session(database=self._database) as session:
            query = "SHOW DATABASES"
            result = await session.run(query)
            databases = await result.values()
            current_database = next((db[0] for db in databases if db[0] == self._database), None)
            return current_database

    async def _node2vec_embed(self):
        print("Implemented but never called.")


if __name__ == "__main__":
    # write code to get the data from neo4j database and display it
    neodb = Neo4JStorage("rag_neo4j", "rag_neo4j", "rag_neo4j")
    try:
        dbresult = asyncio.run(neodb.get_database())
        print(dbresult)
        node_count = asyncio.run(neodb.get_node_count())
        print(node_count)
    finally:
        asyncio.run(neodb.close())
