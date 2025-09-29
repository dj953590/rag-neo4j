
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import Table, Column, Integer, String, MetaData, text, func, create_engine, and_
from sqlalchemy.orm import declarative_base, sessionmaker

from src.storage.db.base import StorageNameSpace
from src.storage.db.sql.tables.engine_tables import DocumentMaster
from src.storage.db.sql.templates.classifier import SQL_TEMPLATE_CLASSIFIER
from src.storage.db.sqlbase import SQLBase
from src.utils.log import logger
from dynaconf import settings

Base = declarative_base()

@dataclass
class PGDB(SQLBase):
    _schema: str = field(init=False, default=None)
    _connection_string: str = field(init=False, default=None)
    metadata: MetaData = field(init=False, default_factory=MetaData)

    def __post_init__(self):
        self._connection_string = settings.get('PG_DATABASE_URL', None)
        self._schema = settings.get('PG_SCHEMA', None)
        args = {"options": f"-c search_path={self._schema}"}
        self.engine = create_engine(self._connection_string, connect_args=args)
        self.Session = sessionmaker(bind=self.engine)



    def read(self, query: str, params: dict = None) -> list[Any]:
        """
        Execute a raw SQL query with optional parameter binding.

        Args:
            query : The SQL query string.
            params (dict, optional): Dictionary of parameters to bind.

        Returns:
            list: Query results as a list of rows, or rowcount for non-select queries.
        """
        try:
            stmt = text(query)
            stmt_params = stmt.bindparams(**params)
            compiled = stmt_params.compile(self.engine, compile_kwargs={"literal_binds": True})
            logger.info(f"SQL: {compiled}")
            with self.engine.connect() as conn:
                result = conn.execute(stmt_params)
                rows: list[Any] = result.fetchall()  # Explicitly type rows
                list_of_dicts = [row._asdict() for row in rows]
                return list_of_dicts
        except Exception as e:
            logger.error(f"Error during select: {str(e)}")
            return []

async def main():

    namespace = "example"  # Replace with actual value
    global_config = {}
    pgdb = PGDB(namespace, global_config)
    classic_sql = SQL_TEMPLATE_CLASSIFIER["document_result"]
    doc_id = "DOC-7adaca4f7a065364c6a54b4aab78729ff0f3653a853c6f5dd48743451dd9b941"
    namespace = "chunks"
    params = {
        "doc_id": doc_id,
        "namespace": namespace
    }
    result = pgdb.read(classic_sql, params)
    print(result)
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())