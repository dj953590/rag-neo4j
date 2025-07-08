from abc import ABC, abstractmethod
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

@dataclass
class SQLBase(ABC):
    """Base class for SQL database operations."""
    NAME_SPACE = "ns"
    namespace: str
    global_config: dict
    @abstractmethod
    def insert(self, table_name, data):
        pass

    @abstractmethod
    def delete(self, table_name, condition):
        pass

    @abstractmethod
    def update(self, table_name, condition, data):
        pass

    @abstractmethod
    def select(self, table_name, condition=None):
        pass

    def execute(self, query):
        raise NotImplementedError("Subclasses should implement this method.")

