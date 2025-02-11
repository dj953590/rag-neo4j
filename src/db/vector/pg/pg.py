import asyncio
from dataclasses import dataclass
from typing import Union
import numpy as np
from dynaconf import settings
from pgvector.sqlalchemy import Vector
from sqlalchemy import create_engine, Column, String, JSON, text, literal_column, Float, bindparam
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func
from src.db.base import BaseVectorStorage
from src.utils.log import logger

Base = declarative_base()


class VectorTable(Base):
    """Table schema for storing vectors in PostgreSQL."""
    __tablename__ = "documents"
    """
    id: Unique identifier for the document chunk
    doc_id: Unique identifier for the document
    doc_name: Name of the file
    embedding: Vector embedding of the document
    content: Content of the document
    mdata: Metadata of the document
    """
    id = Column(String, primary_key=True)
    doc_id = Column(String, nullable=False)
    doc_name = Column(String, nullable=False)
    embedding = Column(Vector)  # PGVector column type
    content = Column(String)
    mdata = Column(JSON)


@dataclass
class PGVectorStorage(BaseVectorStorage):
    """PGVector vector storage implementation."""

    cosine_better_than_threshold: float = 0.2

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
            self._max_batch_size = settings.get("embedding_batch_num", 32)

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
            metadatas = [
                {k: v for k, v in item.items() if k in self.meta_fields}
                or {"_default": "true"}
                for item in data.values()
            ]

            # Logic here is to process in batches to avoid memory issues
            # Split data into batches
            # Use asyncio.gather to run embedding tasks concurrently
            # Upsert in batches

            batches = [
                documents[i: i + self._max_batch_size]
                for i in range(0, len(documents), self._max_batch_size)
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
            session = self._Session()
            for i in range(0, len(ids), self._max_batch_size):
                batch_slice = slice(i, i + self._max_batch_size)

                for j in range(len(ids[batch_slice])):
                    vector = VectorTable(
                        id=ids[batch_slice][j],
                        doc_id=doc_ids[batch_slice][j],
                        doc_name=doc_names[batch_slice][j],
                        embedding=embeddings[batch_slice][j].tolist(),
                        content=documents[batch_slice][j],
                        mdata=metadatas[batch_slice][j],
                    )
                    session.merge(vector)  # Upsert operation

                session.commit()

            session.close()
            return ids

        except Exception as e:
            logger.error(f"Error during PGVector upsert: {str(e)}")
            raise

    async def query(self, query: str, doc_id: str, top_k=5) -> Union[dict, list[dict]]:
        try:
            embedding = await self.embedding_func([query])

            session = self._Session()
            query_embedding = embedding.tolist()[0]

            # Use PGVector's cosine similarity operator <=> for querying
            # Sort results by cosine similarity in descending order, then by distance in ascending order
            # Limit results to top_k + 2 to ensure we have at least top_k results and one additional result
            # (for calculating the cosine similarity with the query)
            """
            results = session.query(
                VectorTable.id,
                VectorTable.content,
                VectorTable.mdata,
                text("1 - (embedding <=> :query_embedding) as distance")
            ).order_by(
                text("embedding <=> :query_embedding")
            ).params(
                query_embedding=query_embedding
            ).limit(top_k * 2).all()
            """
            results = session.query(VectorTable,
                                    VectorTable.embedding.cosine_distance(query_embedding).label("distance")
                                    ).filter(doc_id == VectorTable.doc_id
                                             ).order_by("distance").limit(top_k * 2).all()

            # Filter results by cosine similarity threshold and take top k
            filtered_results = [
                                   {
                                       "id": result.VectorTable.id,
                                       "distance": result.distance,
                                       "content": result.VectorTable.content,
                                       **result.VectorTable.mdata,
                                   }
                                   for result in results
                                   if result.distance >= self.cosine_better_than_threshold
                               ][:top_k]

            session.close()
            return filtered_results

        except Exception as e:
            logger.error(f"Error during PGVector query: {str(e)}")
            raise

    async def index_done_callback(self):
        # PGVector handles persistence automatically
        pass
