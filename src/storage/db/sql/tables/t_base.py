from enum import Enum
from sqlalchemy import create_engine, Column, String, JSON, Integer, func, DateTime
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()


class DocumentsMaster(Base):
    """Table schema for storing vectors in PostgreSQL."""
    __tablename__ = "DOCUMENTS_MASTER"
    """
    id: Unique identifier for the document 
    name: Name of the file
    parent: parent of the document
    state: state of the document
    updated_on: last updated time
    """
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    state = Column(String, nullable=False)
    parent = Column(String, nullable=False)
    updated_on = Column(DateTime, server_default=func.now(), nullable=False)


class DocumentsState(Base):
    """Table schema for storing vectors in PostgreSQL."""
    __tablename__ = "DOCUMENTS_STATE"
    """
    code: Unique identifier for the document state 
    description: Description of the state 
    """
    code = Column(String, primary_key=True)
    description = Column(String, nullable=False)


class DocumentState(Enum):
    pass
