import os
import random
from typing import Any, List

from attr import asdict
from mcp.server.fastmcp.server import FastMCP
from dynaconf import settings
from pydantic import BaseModel, Field
from fastmcp.server.auth.providers.jwt import JWTVerifier
from sqlalchemy import text

from src.storage.db.sql.pgdb import PGDB
from src.storage.db.sql.templates.classifier import SQL_TEMPLATE_CLASSIFIER
from src.storage.db.vector.pg.pg import PGVectorStorage

# Use a symmetric key for simplicity in this example
# In a real-world scenario, you would use an asymmetric key (public/private key pair)
# or a JWKS endpoint for external-facing APIs.
"""
JWT_SECRET = os.getenv("JWT_SECRET", "very-strong-secret-that-is-at-least-32-chars-long")
JWT_ISSUER = "https://localhost:8001"
JWT_AUDIENCE = "calculator-service"

verifier = JWTVerifier(
    public_key=JWT_SECRET,
    issuer=JWT_ISSUER,
    audience=JWT_AUDIENCE,
    algorithm="HS256"
)
"""
mcp = FastMCP(
    name="Data Server",
    instructions="This server provides secure data from the database.",
    host="0.0.0.0",
    port=8002,
)

class DocumentRecord(BaseModel):
    """Defines the schema for a single record returned from the database."""
    id: int = Field(description="The unique ID of the processed chunk.")
    status: str = Field(description="The current processing status (e.g., COMPLETED, PENDING).")
    timestamp: str = Field(description="The last update timestamp in ISO format.")

class DocumentStatusOutput(BaseModel):
    """The complete and validated schema for the tool's return value."""
    result: List = Field(description="A list of all processing records found for the document.")


@mcp.tool(
    name="get_document_status",
    description="provide status of document processing from the database."
)

def get_document_status(document_id: str) -> dict:
    """
    Get the status of document processing from the database.
    Args:
        document_id: ID of the document
    Returns:
        A dictionary containing the status of the document processing.
    """
    namespace = "chunks"  # Replace with actual value
    global_config = {}
    pgdb = PGDB(namespace, global_config)
    classic_sql = SQL_TEMPLATE_CLASSIFIER["document_result"]
    params = {
        "doc_id": document_id,
        "namespace": namespace
    }
    result_data: List = pgdb.read(classic_sql, params)
    return DocumentStatusOutput(result=result_data)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
    # for mcp inspector $env:PYTHONPATH="C:\Users\DJ\git\rag-neo4j" mcp dev data.py
