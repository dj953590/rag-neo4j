
from dataclasses import dataclass, field
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



    def execute(self, query):
        compiled = query.compile(self.engine, compile_kwargs={"literal_binds": True})
        logger.info(f"SQL: {compiled}")
        with self.engine.connect() as conn:
            result = conn.execute(query)
            try:
                return result.fetchall()
            except Exception:
                return result.rowcount

async def main():

    namespace = "example"  # Replace with actual value
    global_config = {}
    pgdb = PGDB(namespace, global_config)
    
    """
        columns = [
            Column('id', Integer, primary_key=True),
            Column('name', String)
        ]
        table_name = "example_table"
        data = {"id": 1, "name": "Test Name"}

        pgdb.create_table(table_name, columns)
        pgdb.delete(table_name, text("id=1"))
        print("Deleted data from PostgreSQL")
        pgdb.insert(table_name, data)
        print("Inserted data into PostgreSQL")

        master = DocumentMaster(id="1", name="Test", state="Test", parent="Test", updated_on=func.now())
        pgdb.merge(master)
        result =  pgdb.select(table_name)
        print("Selected data from PostgreSQL:", result)

        result =  pgdb.select("document_master", text("id='1'"))
        print("Selected data from PostgreSQL:", result)
        pgdb.update(table_name, text("id=1"), {"name": "Updated Name"})
        print("Updated data in PostgreSQL")

        # SELECT using execute
        select_query = text(f"SELECT * FROM {table_name} WHERE id=1")
        result = pgdb.execute(select_query)
        print("Execute SELECT result:", result)

        # UPDATE using execute
        update_query = text(f"UPDATE {table_name} SET name='Execute Updated' WHERE id=1")
        update_count = pgdb.execute(update_query)
        print("Execute UPDATE affected rows:", update_count)

        pgdb.delete(table_name, text("id=1"))
        print("Deleted data from PostgreSQL")
        pgdb.delete("document_master", text("id='1'"))
        print("Deleted documents master data from PostgreSQL")
        pgdb.drop_table(table_name)
    """
    classic_sql = text(SQL_TEMPLATE_CLASSIFIER["classic_sql"])
    doc_id = "DOC-236032ad52b5c77d76ca8b5b9d3ee21fa9d36f49e15211204ee2de8f5df0e70a"
    namespace = "chunks"
    result = pgdb.execute(classic_sql.bindparams(doc_id=doc_id, namespace=namespace))
    print(result)
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())