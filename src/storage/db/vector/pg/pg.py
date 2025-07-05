import asyncio
from dataclasses import dataclass
from typing import Union
import numpy as np
from dynaconf import settings
from pgvector.sqlalchemy import Vector
from sqlalchemy import create_engine, Column, String, JSON, Integer, func, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from src.storage.db.base import BaseVectorStorage, QueryParam, StorageNameSpace
from src.storage.db.sql.tables.engine_tables import Documents
from src.utils.log import logger

Base = declarative_base()

@dataclass
class PGVectorStorage(BaseVectorStorage):
    """PGVector vector storage implementation."""

    cosine_better_than_threshold: float = 0.4

    def __post_init__(self):
        try:
            # Use global config value if specified, otherwise use default
            self.cosine_better_than_threshold = self.global_config.get(
                "cosine_better_than_threshold", self.cosine_better_than_threshold
            )

            self._connection_string = settings.get('PG_DATABASE_URL', None)
            self._schema = settings.get('PG_SCHEMA', None)
            # Create SQLAlchemy engine and session
            args = {"options": f"-c search_path={self._schema}"}
            self._engine = create_engine(self._connection_string,
                                         connect_args=args)

            self._Session = sessionmaker(bind=self._engine)

            # Create the table if it doesn't exist
            Base.metadata.create_all(self._engine)

            # Set batch size for upsert operations
            self._max_batch_size = settings.get("embedding_batch_num", 8)

            # add meta fields
            self.meta_fields.add("page_no")
            self.meta_fields.add("positions")

        except Exception as e:
            logger.error(f"PGVector initialization failed: {str(e)}")
            raise

    async def upsert(self, data: dict[str, dict]):
        if not data:
            logger.warning("Empty data provided to vector DB")
            return []

        try:
            ids = list(data.keys())
            doc_ids = [v["doc_id"] for v in data.values()]
            doc_names = [v["doc_name"] for v in data.values()]
            documents = [v["content"] for v in data.values()]
            source_ids = [v["s_id"] for v in data.values()]

            metadatas = [
                {k: v for k, v in item.items() if k in self.meta_fields}
                | {StorageNameSpace.NAME_SPACE: self.namespace}
                or {"_default": "true"}
                for item in data.values()
            ]
            chunk_sequences = [v.get("chunk_sequence", 0) for v in data.values()]

            batch_size = 40
            session = self._Session()
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_doc_ids = doc_ids[i:i + batch_size]
                batch_doc_names = doc_names[i:i + batch_size]
                batch_documents = documents[i:i + batch_size]
                batch_source_ids = source_ids[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                batch_chunk_sequences = chunk_sequences[i:i + batch_size]

                # Logic here is to process in batches to avoid memory issues
                # Split data into batches
                # Use asyncio.gather to run embedding tasks concurrently
                # Upsert in batches

                batches = [
                    batch_documents[i: i + self._max_batch_size]
                    for i in range(0, len(batch_documents), self._max_batch_size)
                ]
                # Get embeddings for each batch
                embedding_tasks = [self.embedding_func(batch) for batch in batches]
                embeddings_list = []

                # Pre-allocate embeddings_list with known size
                embeddings_list = [None] * len(embedding_tasks)

                # Use asyncio.gather instead of as_completed if order doesn't matter
                embeddings_results = await asyncio.gather(*embedding_tasks)
                embeddings_list = list(embeddings_results)
                # Flatten embeddings list
                embeddings = np.concatenate(embeddings_list)

                # Upsert in batches

                # Process each batch
                for j in range(len(batch_ids)):
                    vector = Documents(
                        chunk_id=batch_ids[j],
                        doc_id=batch_doc_ids[j],
                        embedding=embeddings[j].tolist(),
                        content=batch_documents[j],
                        mdata=batch_metadatas[j],
                        chunk_sequence=batch_chunk_sequences[j],
                        source_chunk=batch_source_ids[j],
                        updated_on=func.now()
                    )
                    session.merge(vector)  # Upsert operation

                session.commit()

            session.close()
            return ids

        except Exception as e:
            logger.error(f"Error during PGVector upsert: {str(e)}")
            raise

    async def query(self, query: str, param: QueryParam) -> Union[dict, list[dict]]:
        try:
            top_k = 5
            if param.top_k:
                top_k = param.top_k

            doc_id = param.doc_id
            embedding = await self.embedding_func([query])

            session = self._Session()
            query_embedding = embedding.tolist()[0]

            # Use PGVector's cosine similarity operator <=> for querying
            # Sort results by cosine similarity in descending order, then by distance in ascending order
            # Limit results to top_k + 2 to ensure we have at least top_k results and one additional result
            # (for calculating the cosine similarity with the query)
            """
            results = session.query(
                Documents.id,
                Documents.content,
                Documents.mdata,
                text("1 - (embedding <=> :query_embedding) as distance")
            ).order_by(
                text("embedding <=> :query_embedding")
            ).params(
                query_embedding=query_embedding
            ).limit(top_k * 2).all()
            """
            results = session.query(Documents, (1 - Documents.embedding.cosine_distance(query_embedding)).label("cosine")
                                        ).filter(Documents.doc_id == doc_id if doc_id is not None else True
                                        ).order_by((1 - Documents.embedding.cosine_distance(query_embedding)).desc()).limit(top_k * 2).all()
            # Filter results by cosine similarity threshold and take top k
            filtered_results = [
                                   {
                                       "chunk_id": result.Documents.chunk_id,
                                       "cosine": result.cosine,
                                       "content": result.Documents.content,
                                       "source_id": result.Documents.source_chunk,
                                       **result.Documents.mdata,
                                   }
                                   for result in results
                                   if result.cosine >= self.cosine_better_than_threshold and result.Documents.mdata.get(StorageNameSpace.NAME_SPACE) == self.namespace
                               ][:top_k]

            session.close()
            return filtered_results

        except Exception as e:
            logger.error(f"Error during PGVector query: {str(e)}")
            raise

    async def index_done_callback(self):
        # PGVector handles persistence automatically
        pass
