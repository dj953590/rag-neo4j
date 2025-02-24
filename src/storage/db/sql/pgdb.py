import asyncio
from dataclasses import dataclass, field

from sqlalchemy import Table, Column, Integer, String, MetaData, text, create_engine, func
from sqlalchemy.ext.asyncio import create_async_engine

from src.storage.db.sql.tables.t_base import DocumentsMaster
from src.storage.db.sqlbase import SQLBase
from dynaconf import settings
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

@dataclass
class PGDB(SQLBase):
    """
        PostgreSQL database implementation.
    """
    _schema: str = field(init=False, default=None)
    session: sessionmaker = field(init=False, default=None)
    engine: create_async_engine = field(init=False, default=None)
    _connection_string: str = field(init=False, default=None)
    metadata: MetaData = field(init=False, default_factory=MetaData)

    def __post_init__(self):

        self._connection_string = settings.get('PG_DATABASE_URL', None)
        self._schema = settings.get('PG_SCHEMA', None)

        args = {"options": f"-c search_path={self._schema}"}
        self.engine = create_engine(self._connection_string, connect_args=args)
        self.session = sessionmaker(bind=self.engine)

    def create_table(self, table_name, columns):
        """
        Create a table in PostgreSQL.
        Args:
            table_name: Name of the table
            columns: List of columns to be added to the table
        """
        table = Table(table_name, self.metadata, *columns)
        with self.engine.begin() as conn:
            self.metadata.create_all(conn)

    def drop_table(self, table_name):
        """
        Drop a table in PostgreSQL.
        Args:
            table_name: Name of the table
        """
        table = Table(table_name, self.metadata, autoload_with=self.engine)
        with self.engine.begin() as conn:
            self.metadata.create_all(conn)

    def insert(self, table_name, data):
        """
        Insert data into PostgreSQL.
        Args:
            table_name: Name of the table
            data: Data to be inserted into the table
        """
        table = Table(table_name, self.metadata, autoload_with=self.engine)
        with self.session() as session:
            session.execute(table.insert().values(data))
            session.commit()

    def merge(self, base: Base,):
        """
        Insert data into PostgreSQL.
        Args:
            base: Base object to be merged into the table
        """
        with self.session() as session:
            session.merge(base)
            session.commit()

    def delete(self, table_name, condition):
        """
        Delete data from PostgreSQL.
        Args:
            table_name: Name of the table
            condition: Condition to be used for deletion
        """
        table = Table(table_name, self.metadata, autoload_with=self.engine)
        with self.session() as session:
            session.execute(table.delete().where(condition))
            session.commit()

    def update(self, table_name, condition, data):
        """
        Update data in PostgreSQL.
        Args:
            table_name: Name of the table
            condition: Condition to be used for update
            data: Data to be updated
        Return:
            result: Result of the update operation
        """
        table = Table(table_name, self.metadata, autoload_with=self.engine)
        with self.session() as session:
            session.execute(table.update().where(condition).values(data))
            session.commit()

    def select(self, table_name, condition=None):
        """
        Select data from PostgreSQL.
        Args:
            table_name: Name of the table
            condition: Condition to be used for selection
        Return:
            result: Result of the select operation
        """
        table = Table(table_name, self.metadata, autoload_with=self.engine)
        with self.session() as session:
            if condition is not None:
                result = session.execute(table.select().where(condition))
            else:
                result = session.execute(table.select())
            return result.fetchall()


def main():
    # PostgreSQL database URL
    pgdb = PGDB()
    # Example table name and data
    # Example usage
    columns = [
        Column('id', Integer, primary_key=True),
        Column('name', String)
    ]
    # Create table
    pgdb.create_table('example_table', columns)
    table_name = "example_table"
    data = {"id": 1, "name": "Test Name"}

    pgdb.delete(table_name, text("id=1"))
    print("Deleted data from PostgreSQL")
    # Insert data into PostgreSQL
    pgdb.insert(table_name, data)
    print("Inserted data into PostgreSQL")

    master = DocumentsMaster(id="1", name="Test", state="Test", parent="Test", updated_on=func.now())
    pgdb.merge(master)
    # Select data from PostgreSQL
    result = pgdb.select(table_name)
    print("Selected data from PostgreSQL:", result)

    result = pgdb.select("DOCUMENTS_MASTER", text("id='1'"))
    print("Selected data from PostgreSQL:", result)
    # Update data in PostgreSQL
    pgdb.update(table_name, text("id=1"), {"name": "Updated Name"})
    print("Updated data in PostgreSQL")

    # Delete data from PostgreSQL
    pgdb.delete(table_name, text("id=1"))
    print("Deleted data from PostgreSQL")

    pgdb.delete("DOCUMENTS_MASTER", text("id='1'"))
    print("Deleted documents master data from PostgreSQL")
    # Drop table
    pgdb.drop_table('example_table')


if __name__ == "__main__":
    main()
