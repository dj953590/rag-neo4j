import chromadb
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Union
from src.db.base import BaseVectorStorage, QueryParam
from src.utils.log import logger


@dataclass
class ChromaDBStorage(BaseVectorStorage):
    """
    ChromaDB vector storage implementation.
    """

    collection_name: str = "documents"
    cosine_threshold: float = 0.2  # Similarity threshold for filtering results

    def __post_init__(self):
        try:
            # Initialize ChromaDB in-memory or persistent storage
            self.client = chromadb.PersistentClient(path="./chroma_db")  # Change to ChromaClient() for in-memory
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity for nearest neighbors
            )

        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {str(e)}")
            raise

    async def upsert(self, data: Dict[str, Dict]):
        """
        Inserts or updates document embeddings in ChromaDB.
        """
        if not data:
            logger.warning("Empty data provided to vector DB")
            return []

        try:
            ids = list(data.keys())
            doc_ids = [v["doc_id"] for v in data.values()]
            doc_names = [v["doc_name"] for v in data.values()]
            documents = [v["content"] for v in data.values()]
            metadatas = [
                {k: v for k, v in item.items() if k in self.meta_fields} or {"_default": "true"}
                for item in data.values()
            ]

            # Generate embeddings in batches
            batch_size = 32  # Adjust based on hardware
            embeddings = []
            for i in range(0, len(documents), batch_size):
                batch = documents[i: i + batch_size]
                embeddings.extend(await self.embedding_func(batch))

            # Store in ChromaDB
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )

            return ids

        except Exception as e:
            logger.error(f"Error during ChromaDB upsert: {str(e)}")
            raise

    async def query(self, query: str, param: QueryParam) -> List[Dict]:
        """
        Performs a similarity search in ChromaDB.
        """
        try:
            top_k = param.top_k if param.top_k else 5
            doc_id = param.doc_id

            query_embedding = await self.embedding_func([query])[0]

            # Search ChromaDB for nearest neighbors
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 2,  # Fetch extra results to apply threshold filtering
                where={"doc_id": doc_id} if doc_id is not None else None,
            )

            # Process results
            filtered_results = [
                                   {
                                       "id": result["id"],
                                       "distance": result["distance"],
                                       "content": result["document"],
                                       **result["metadata"],
                                   }
                                   for result in results["documents"][0]
                                   if result["distance"] >= self.cosine_threshold
                               ][:top_k]

            return filtered_results

        except Exception as e:
            logger.error(f"Error during ChromaDB query: {str(e)}")
            raise

    async def index_done_callback(self):
        """
        Handles post-indexing tasks (ChromaDB handles persistence automatically).
        """
        pass
