
from dataclasses import dataclass, field
from sqlalchemy import Table, Column, Integer, String, MetaData, text, func, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from src.storage.db.sql.tables.engine_tables import DocumentMaster
from src.storage.db.sqlbase import SQLBase
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

    def create_table(self, table_name, columns):
        table = Table(table_name, self.metadata, *columns)
        with self.engine.begin() as conn:
            self.metadata.create_all(conn)

    def drop_table(self, table_name):
        table = Table(table_name, self.metadata, autoload_with=self.engine)
        with self.engine.begin() as conn:
            self.metadata.drop_all(conn, tables=[table])

    def insert(self, table_name, data):
        table = Table(table_name, self.metadata, autoload_with=self.engine)
        with self.Session() as session:
            session.execute(table.insert().values(data))
            session.commit()

    def merge(self, base: Base):
        with self.Session() as session:
            session.merge(base)
            session.commit()

    def delete(self, table_name, condition):
        table = Table(table_name, self.metadata, autoload_with=self.engine)
        with self.Session() as session:
            session.execute(table.delete().where(condition))
            session.commit()

    def update(self, table_name, condition, data):
        table = Table(table_name, self.metadata, autoload_with=self.engine)
        with self.Session() as session:
            session.execute(table.update().where(condition).values(data))
            session.commit()

    def select(self, table_name, condition=None):
        table = Table(table_name, self.metadata, autoload_with=self.engine)
        with self.Session() as session:
            if condition is not None:
                result = session.execute(table.select().where(condition))
            else:
                result = session.execute(table.select())
            return result.fetchall()


async def main():
    pgdb = PGDB()
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

    pgdb.delete(table_name, text("id=1"))
    print("Deleted data from PostgreSQL")
    pgdb.delete("document_master", text("id='1'"))
    print("Deleted documents master data from PostgreSQL")
    pgdb.drop_table(table_name)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())