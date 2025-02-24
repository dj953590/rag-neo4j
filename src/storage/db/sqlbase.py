from abc import ABC, abstractmethod
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker


class SQLBase(ABC):

    @abstractmethod
    async def insert(self, table_name, data):
        pass

    @abstractmethod
    async def delete(self, table_name, condition):
        pass

    @abstractmethod
    async def update(self, table_name, condition, data):
        pass

    @abstractmethod
    async def select(self, table_name, condition=None):
        pass
