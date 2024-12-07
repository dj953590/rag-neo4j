import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
import logging


class DB:
    def __init__(self, db_config):
        """
        Initialize the DB class with configuration and connection pooling.

        Args:
        db_config (dict): A dictionary containing database configuration.
                          Example:
                          {
                              "dbname": "your_db",
                              "user": "your_user",
                              "password": "your_password",
                              "host": "localhost",
                              "port": "5432",
                              "minconn": 1,
                              "maxconn": 10
                          }
        """
        self.db_config = db_config
        self._pool = None
        self._setup_logging()
        self._initialize_connection_pool()

    def _setup_logging(self):
        """
        Set up logging for database operations.
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("DB")

    def _initialize_connection_pool(self):
        """
        Initialize a connection pool with psycopg2.
        """
        try:
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=self.db_config.get("minconn", 1),
                maxconn=self.db_config.get("maxconn", 10),
                dbname=self.db_config["dbname"],
                user=self.db_config["user"],
                password=self.db_config["password"],
                host=self.db_config["host"],
                port=self.db_config["port"],
            )
            self.logger.info("Connection pool initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize connection pool: {e}")
            raise

    @contextmanager
    def _get_connection(self):
        """
        Context manager to get a connection from the pool.
        """
        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
        finally:
            if conn:
                self._pool.putconn(conn)

    @contextmanager
    def _get_cursor(self, connection):
        """
        Context manager to get a cursor from a connection.
        """
        cursor = None
        try:
            cursor = connection.cursor(cursor_factory=RealDictCursor)
            yield cursor
        finally:
            if cursor:
                cursor.close()

    def execute_query(self, query, params=None):
        """
        Execute a SQL query and return results.

        Args:
        query (str): The SQL query string.
        params (tuple or list): Parameters for the query.

        Returns:
        list: Query results as a list of dictionaries.
        """
        results = []
        with self._get_connection() as conn:
            with self._get_cursor(conn) as cursor:
                try:
                    self.logger.info(f"Executing query: {query}")
                    cursor.execute(query, params)
                    if cursor.description:
                        results = cursor.fetchall()
                except Exception as e:
                    self.logger.error(f"Query execution failed: {e}")
                    conn.rollback()
                    raise
                else:
                    conn.commit()
        return results

    def execute_non_query(self, query, params=None):
        """
        Execute a non-select SQL query.

        Args:
        query (str): The SQL query string.
        params (tuple or list): Parameters for the query.
        """
        with self._get_connection() as conn:
            with self._get_cursor(conn) as cursor:
                try:
                    self.logger.info(f"Executing non-query: {query}")
                    cursor.execute(query, params)
                    conn.commit()
                except Exception as e:
                    self.logger.error(f"Non-query execution failed: {e}")
                    conn.rollback()
                    raise

    def close(self):
        """
        Close all connections in the pool.
        """
        if self._pool:
            self._pool.closeall()
            self.logger.info("Connection pool closed.")


# Example Usage
if __name__ == "__main__":
    db_config = {
        "dbname": "chinook",
        "user": "postgres",
        "password": "postgres",
        "host": "localhost",
        "port": "5432",
        "minconn": 1,
        "maxconn": 5,
    }

    db = DB(db_config)

    try:
        # Fetch data
        rows = db.execute_query("SELECT * FROM customer;")
        print(rows)
    finally:
        db.close()
