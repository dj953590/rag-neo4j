# Auto-generated on Eastern Standard Time
from enum import Enum


class DocumentState(Enum):
    """
    Document states with descriptions from DOCUMENT_STATE table
    
    Attributes:
        value (str): The actual state value from the database
        description (str): Human-readable description of the state
    """

    def __new__(cls, value, description):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.description = description
        return obj

    VECTORIZATION = ("VECTORIZATION", "generate vectors for chunks")
    READY = ("READY", "document is ready for query")
    PROCESSING = ("PROCESSING", "start processing the document")
    UPLOADING = ("UPLOADING", "document upload to s3")
    CHUNKED = ("CHUNKED", "chunking for the document done")
    ENTITIES = ("ENTITIES", "generate entities from chunks")
    KNOWLEDGEGRAPH = ("KNOWLEDGEGRAPH", "build the Knowledge Graph")
