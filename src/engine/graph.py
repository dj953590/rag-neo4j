import asyncio
import os
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import cast

from src.llm.oai import (
    gpt_4o_mini_complete,
    openai_embedding,
)
from src.engine.operations import (
    extract_entities,
    # local_query,global_query,hybrid_query,
    kg_query,
    naive_query,
)
from src.docs.chunker.chunks import extract_chunks, extract_chunks_md, extract_page_chunks_md

from src.utils.log import logger

from src.utils.utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
)
from src.storage.db.base import (
    StorageNameSpace,
    QueryParam,
)

from src.storage.db.kv.kv_json import JsonKVStorage
from src.storage.db.vector.pg.pg import PGVectorStorage
from src.storage.db.graph.networkx.netx import NetworkXStorage
from src.storage.db.graph.neo4j.neo import Neo4JStorage


def lazy_external_import(module_name: str, class_name: str):
    """
    Lazily import a class from an external module based on the package of the caller.

    Args:
        module_name (str): The name of the module to import.
        class_name (str): The name of the class to import.
 Returns:
        type: The imported class.
    """

    # Get the caller's module and package
    import inspect

    caller_frame = inspect.currentframe().f_back
    module = inspect.getmodule(caller_frame)
    package = module.__package__ if module else None

    def import_class(*args, **kwargs):
        import importlib

        # Import the module using importlib
        module = importlib.import_module(module_name, package=package)

        # Get the class from the module and instantiate it
        cls = getattr(module, class_name)
        return cls(*args, **kwargs)

    return import_class


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.

    This function tries to get the current event loop. If the current event loop is closed or does not exist,
    it creates a new event loop and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:
        # Try to get the current event loop
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:
        # If no event loop exists or it is closed, create a new one
        logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop


@dataclass
class GraphEngine:
    """
    Main class for Graph Engine for Retrieval and Reasoning.

    Args:
        llm_model_func (EmbeddingFunc): The function to compute node embeddings.
        working_dir (str): The directory where the cache will be stored.
        kv_storage (str): The storage type for key-value pairs.
        vector_storage (str): The storage type for vector data.
        graph_storage (str): The storage type for graph data.
        graphdb_storage (str): The storage type for graph database data.
        llm_model_name (str): The name of the LLM model to use for embedding.
        llm_model_max_token_size (int): The maximum token size for the LLM model.
        llm_model_max_async (int): The maximum number of asynchronous LLM calls.
        llm_model_kwargs (dict): Additional keyword arguments for the LLM model.
        vector_db_storage_cls_kwargs (dict): Additional keyword arguments for the vector database storage class.
        log_level (str): The log level for the GRAG system.

    Returns:
          GRAG: An instance of the GRAG class.
    """
    working_dir: str = field(
        default_factory=lambda: f"./tkgs_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    # Default not to use embedding cache
    embedding_cache_config: dict = field(
        default_factory=lambda: {
            "enabled": False,
            "similarity_threshold": 0.95,
            "use_llm_check": False,
        }
    )
    kv_storage: str = field(default="JsonKVStorage")
    vector_storage: str = field(default="PGVectorStorage")
    graph_storage: str = field(default="NetworkXStorage")
    graphdb_storage: str = field(default="Neo4JStorage")

    current_log_level = logger.level
    log_level: str = field(default=current_log_level)

    # text chunking
    chunk_token_size: int = 1024
    min_percentage: int = 10
    chunk_overlap_token_size: int = 50

    tiktoken_model_name: str = "gpt-4o-mini"

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 512
    entity_extract_batch_size: int = 4

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 4
    embedding_func_max_async: int = 4

    # LLM
    llm_model_func: callable = gpt_4o_mini_complete
    llm_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 2
    llm_model_kwargs: dict = field(default_factory=dict)

    # storage
    doc_id: str = None
    doc_name: str = None

    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)

    # extension
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json
    chunk_entity_relation_graph = None

    def __post_init__(self):
        logger.setLevel(self.log_level)

        logger.info(f"Logger initialized for working directory: {self.working_dir}")

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"Graph RAG init with param:\n  {_print_config}\n")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )

        # Initialize storage instances
        # full_docs : JsonKVStorage
        self.full_docs = (JsonKVStorage(namespace="full_docs", global_config=asdict(self),
                                        embedding_func=self.embedding_func, ))
        # text_chunks : JsonKVStorage
        self.text_chunks = ((JsonKVStorage(namespace="text_chunks", global_config=asdict(self),
                                           embedding_func=self.embedding_func, )))
        # chunk_entity_relation_graph : NetworkXStorage
        self.chunk_entity_relation_graph = (NetworkXStorage(namespace="chunk_entity_relation",
                                                            global_config=asdict(self),
                                                            embedding_func=self.embedding_func, ))
        # chunk_entity_relation_graphdb : Neo4JStorage
        self.chunk_entity_relation_graphdb = (Neo4JStorage(namespace="chunk_entity_relation",
                                                           global_config=asdict(self),
                                                           embedding_func=self.embedding_func, ))
        """
        if self.doc_id and self.doc_name:
            asyncio.run(self.chunk_entity_relation_graphdb._create_doc_node(self.doc_id, self.doc_name))
        else:
            logger.warning("Document ID and name not provided. Skipping DOC node creation.")
            raise ValueError("Document ID and name not provided.")
        """
        ####
        # add embedding func by walter over
        ####
        self.entities_vdb = (PGVectorStorage(namespace="entities",
                                             global_config=asdict(self),
                                             embedding_func=self.embedding_func,
                                             meta_fields={"entity_name"},
                                             )
                             )
        self.relationships_vdb = (PGVectorStorage(namespace="relationships",
                                                  global_config=asdict(self),
                                                  embedding_func=self.embedding_func,
                                                  meta_fields={"src_id", "tgt_id"},
                                                  )
                                  )
        self.chunks_vdb = (PGVectorStorage(namespace="chunks",
                                           global_config=asdict(self),
                                           embedding_func=self.embedding_func,
                                           )
                           )

    def insert(self, data: list):
        """
        Insert JSON data into the storage.

        Args:
            data (dict): The JSON data to be inserted.
        Returns:
                None
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(data))

    async def check_docs(self, docs: str):
        """
        Check if the documents are already in the storage.

        Args:
            docs (List[str]): The list of documents to check.
        Returns:
                List[str]: The list of documents that are not in the storage.
        """
        if isinstance(docs, str):
            docs = [docs]

        new_docs = {
            compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
            for c in docs
        }
        _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
        new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
        if not len(new_docs):
            logger.warning("All docs are already in the storage")
            return None
        return new_docs

    async def ainsert(self, data: list):
        """
        Insert one or more strings into the storage asynchronously.

        Args:
            data (dict): json data of the text extracted from PDF.
        Returns:
                None
        """
        update_storage = False
        chunk_data, docs = extract_page_chunks_md(data)
        try:
            new_docs = await self.check_docs(docs)
            if new_docs is None:
                return
            update_storage = True

            logger.info(f"[New Docs] inserting {len(new_docs)} docs into memory storage")

            inserting_chunks = {}
            chunk_sequence = 0  # Initialize chunk sequence
            """for doc_key, doc in tqdm_async(new_docs.items(), desc="Chunking documents", unit="doc"): 
            new_docs.items() returns an iterator over the key-value pairs (document key and document content) in the 
            new_docs dictionary. tqdm_async is used to display a progress bar for the iteration, with the description 
            "Chunking documents" and the unit "doc".
            """

            for doc_key, doc in tqdm_async(
                    new_docs.items(), desc="Chunking documents", unit="doc"
            ):
                chunks = {}
                for chunk in chunk_data:
                    chunk_text = chunk["chunk_text"]
                    page_no = chunk["page"]
                    positions = chunk.get("positions", [])
                    chunk_id = compute_mdhash_id(chunk_text, prefix="chunk-")
                    chunks[chunk_id] = {
                        "content": chunk_text,
                        # "bounding_box": chunk['bounding_box'],
                        # "token_count": chunk['token_count'],
                        "full_doc_id": doc_key,
                        "doc_id": self.doc_id,
                        "doc_name": self.doc_name,
                        "chunk_sequence": chunk_sequence,
                        "page_no": page_no,
                        "positions": positions,
                        "s_id": doc_key,  # Source ID for the chunk
                    }
                    chunk_sequence += 1  # Increment chunk sequence
                inserting_chunks.update(chunks)
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage")
                return

            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks into vector storage")

            await self.chunks_vdb.upsert(inserting_chunks)

            logger.info("[Entity Extraction]...")
            maybe_new_kg = await extract_entities(
                inserting_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                kg_db=self.chunk_entity_relation_graphdb,
                global_config=asdict(self),
            )
            if maybe_new_kg is None:
                logger.warning("No new entities and relationships found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg

            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            if update_storage:
                await self._insert_done()

    async def _insert_done(self):
        """
        Indexing done callback for all storage instances.
        This method is called after all the insertions are done.
        If any storage instance is not initialized, this method will simply return.
        Args:
            None
        Returns:
            None
        """
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.entities_vdb,
            self.relationships_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def query(self, query: str, param: QueryParam = QueryParam()):
        """
        Query the knowledge graph using the given query and parameters.

        Args:
            query (str): The query string to be executed.
            param (QueryParam): The parameters for the query.
        Returns:
                list: The results of the query.
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        """
        Execute the given query and parameters.

        Args:
            query (str): The query string to be executed.
            param (QueryParam): The parameters for the query.
        Returns:
                list: The results of the query.
        """
        if param.mode in ["hybrid"]:
            response = await kg_query(
                query,
                self.chunk_entity_relation_graph,
                self.chunk_entity_relation_graphdb,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )
        elif param.mode == "naive":
            response = await naive_query(
                query,
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        return response

    def delete_by_entity(self, entity_name: str):
        """
        Delete the entity and its relationships from the knowledge graph.

        Args:
            entity_name (str): The name of the entity to be deleted.
            Returns:
                None
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

    async def adelete_by_entity(self, entity_name: str):
        """
        Delete the entity and its relationships from the knowledge graph.

        Args:
            entity_name (str): The name of the entity to be deleted.
        Returns:
            None
        """

        entity_name = f'"{entity_name.upper()}"'

        try:
            await self.entities_vdb.delete_entity(entity_name)
            await self.relationships_vdb.delete_relation(entity_name)
            await self.chunk_entity_relation_graph.delete_node(entity_name)

            logger.info(
                f"Entity '{entity_name}' and its relationships have been deleted."
            )
            await self._delete_by_entity_done()
        except Exception as e:
            logger.error(f"Error while deleting entity '{entity_name}': {e}")

    async def _delete_by_entity_done(self):
        """
        Callback function to be executed after the deletion of an entity is done.
        Deletes the index of the knowledge graph.
        Args:

        Returns:

        """
        tasks = []
        for storage_inst in [
            self.entities_vdb,
            self.relationships_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
