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


    def execute(self, query):
        raise NotImplementedError("Subclasses should implement this method.")

