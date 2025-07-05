# Auto-generated on Eastern Daylight Time
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

    READY = ("Ready", "document is ready for query")
    PROCESSING = ("Processing", "start processing the document")
    CHUNKING = ("Chunking", "chunking in progress")
    ENTITIES = ("Entities", "Entities and relationship extraction")
    KG = ("kg", "Save the Knowledge Graph")
    FAILED = ("Failed", "Processing of document failed")
