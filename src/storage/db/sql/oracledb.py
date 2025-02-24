from dataclasses import dataclass, field
from sqlalchemy import Table, Column, Integer, String, MetaData, text, create_engine
from sqlalchemy.orm import sessionmaker
from src.storage.db.sqlbase import SQLBase
from dynaconf import settings


@dataclass
class OracleDB(SQLBase):
    """
    Oracle database implementation.
    """
    _schema: str = field(init=False, default=None)
    session: sessionmaker = field(init=False, default=None)
    engine: create_engine = field(init=False, default=None)
    _connection_string: str = field(init=False, default=None)
    metadata: MetaData = field(init=False, default_factory=MetaData)

    def __post_init__(self):
        self._connection_string = settings.get('ORACLE_DATABASE_URL', None)
        self._schema = settings.get('ORACLE_SCHEMA', None)

        args = {"options": f"-c search_path={self._schema}"}
        self.engine = create_engine(self._connection_string, connect_args=args)
        self.session = sessionmaker(bind=self.engine)

    def create_table(self, table_name, columns):
        """
        Create a table in Oracle.
        Args:
            table_name: Name of the table
            columns: List of columns to be added to the table
        """
        table = Table(table_name, self.metadata, *columns)
        with self.engine.begin() as conn:
            self.metadata.create_all(conn)

    def drop_table(self, table_name):
        """
        Drop a table in Oracle.
        Args:
            table_name: Name of the table
        """
        table = Table(table_name, self.metadata, autoload_with=self.engine)
        with self.engine.begin() as conn:
            self.metadata.drop_all(conn)

    def insert(self, table_name, data):
        """
        Insert data into Oracle.
        Args:
            table_name: Name of the table
            data: Data to be inserted into the table
        """
        table = Table(table_name, self.metadata, autoload_with=self.engine)
        with self.session() as session:
            session.execute(table.insert().values(data))
            session.commit()

    def delete(self, table_name, condition):
        """
        Delete data from Oracle.
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
        Update data in Oracle.
        Args:
            table_name: Name of the table
            condition: Condition to be used for update
            data: Data to be updated
        Return:
            result: Result of the update operation
        """
        table = Table(table_name, self.metadata, autoload_with(self.engine))
        with self.session() as session:
            session.execute(table.update().where(condition).values(data))
            session.commit()

    def select(self, table_name, condition=None):
        """
        Select data from Oracle.
        Args:
            table_name: Name of the table
            condition: Condition to be used for selection
        Return:
            result: Result of the select operation
        """
        table = Table(table_name, self.metadata, autoload_with(self.engine))
        with self.session() as session:
            if condition:
                result = session.execute(table.select().where(condition))
            else:
                result = session.execute(table.select())
            return result.fetchall()


def main():
    # Oracle database URL
    oracledb = OracleDB()
    # Example table name and data
    columns = [
        Column('id', Integer, primary_key=True),
        Column('name', String)
    ]
    # Create table
    oracledb.create_table('example_table', columns)
    table_name = "example_table"
    data = {"id": 1, "name": "Test Name"}

    # Insert data into Oracle
    oracledb.insert(table_name, data)
    print("Inserted data into Oracle")

    # Select data from Oracle
    result = oracledb.select(table_name)
    print("Selected data from Oracle:", result)

    # Update data in Oracle
    oracledb.update(table_name, text("id=1"), {"name": "Updated Name"})
    print("Updated data in Oracle")

    # Delete data from Oracle
    oracledb.delete(table_name, text("id=1"))
    print("Deleted data from Oracle")
    # Drop table
    oracledb.drop_table('example_table')


if __name__ == "__main__":
    main()
