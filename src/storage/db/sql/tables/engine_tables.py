from enum import Enum
from sqlalchemy import create_engine, Column, String, JSON, Integer, func, DateTime, Date, Text
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()


class DocumentMaster(Base):
    __tablename__ = "document_master"
    __table_args__ = {"schema": "rag"}

    id = Column(String, primary_key=True, nullable=False)
    name = Column(String, nullable=False)
    parent = Column(String, nullable=False)
    state = Column(String, nullable=False)
    updated_on = Column(DateTime, server_default=func.now(), nullable=False)
    summary = Column(Text, nullable=True)
    content_length = Column(Integer, nullable=True)
    chunks_count = Column(Integer, nullable=True)
    classification = Column(String, nullable=True)
    mdata = Column(String, nullable=True)

class Documents(Base):
    __tablename__ = "documents"
    __table_args__ = {"schema": "rag"}

    chunk_id = Column(String, primary_key=True, nullable=False)
    content = Column(String, nullable=True)
    mdata = Column(JSON, nullable=True)
    embedding = Column(String, nullable=True)  # Replace with custom type if using pgvector
    doc_id = Column(String, nullable=True)
    chunk_sequence = Column(Integer, nullable=True)
    updated_on = Column(DateTime, server_default=func.now(), nullable=False)
    source_chunk = Column(String, nullable=True)
    chunk_classification = Column(String, nullable=True)
    chunk_summary = Column(String, nullable=True)

class DocumentsState(Base):
    """Table schema for storing vectors in PostgreSQL."""
    __tablename__ = "document_state"
    """
    code: Unique identifier for the document state 
    description: Description of the state 
    """
    code = Column(String, primary_key=True)
    description = Column(String, nullable=False)
