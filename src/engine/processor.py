import asyncio
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import uuid
from typing import Optional

from sqlalchemy import func, text

from src.engine.graph import GraphEngine
from src.storage.db.sql.pgdb import PGDB
from src.storage.s3.s3store import S3Storage
from src.utils.log import logger
from src.utils.utils import compute_mdhash_id
from src.storage.db.sql.tables.engine_tables import DocumentsMaster


@dataclass
class BaseProcessor(ABC):
    @abstractmethod
    async def start_processing(self, file_content: bytes, data: dict) -> str:
        pass

    @abstractmethod
    async def upload_document(self, file_content: bytes, file_name: str) -> str:
        pass

    @abstractmethod
    async def create_process(self, file_name: str, metadata: dict) -> str:
        pass

    @abstractmethod
    async def process_document(self, process_id: str) -> None:
        pass


@dataclass
class DocumentProcessor(BaseProcessor):
    s3_storage: S3Storage = field(init=False)
    pgdb: PGDB = field(init=False)
    graph_engine: GraphEngine = field(init=False)

    def __post_init__(self):
        self.s3_storage = S3Storage()
        self.pgdb = PGDB()
        WORKING_DIR = "./tmp"

        if not os.path.exists(WORKING_DIR):
            os.mkdir(WORKING_DIR)

        self.graph_engine = GraphEngine(working_dir=WORKING_DIR)

    async def start_processing(self, content: bytes, data: dict) -> str:
        """"Full processing pipeline"""
        # Extract file_name from data
        parent_id = data.get("parent_id")
        file_name = data.get("name")
        if not file_name:
            raise ValueError("File name is missing in data")

        # Upload to S3
        object_name = f"{parent_id}/{file_name}"
        object_name = await self.upload_document(content, object_name)
        # Create process record
        p_id = await self.create_process(object_name, data)

        # Start async processing
        await asyncio.create_task(self.process_document(p_id))

        return process_id

    async def upload_document(self, content: bytes, object_name: str) -> str:
        """Upload document to S3 and return object name"""
        if not self.s3_storage.upload_file_content(content, object_name):
            raise RuntimeError("Failed to upload document to S3")
        return object_name

    async def create_process(self, object_name: str, data: dict) -> str:
        """Create document master record and return process_id (doc_id)"""
        parent_id = data.get("parent_id")
        file_name = data.get("name")
        doc_id = compute_mdhash_id(file_name)
        document_master = DocumentsMaster(
            id=doc_id,
            name=object_name,
            state="UPLOADED",
            parent=parent_id,
            updated_on=func.now()
        )
        self.pgdb.merge(document_master)
        return doc_id

    async def process_document(self, process_id: str) -> None:
        """Process document by downloading from S3 and inserting chunks"""
        try:
            # Get document metadata
            doc_master = self.pgdb.select("DOCUMENTS_MASTER", text(f"id='{process_id}'"))
            if not doc_master:
                raise ValueError(f"Process {process_id} not found")

            # Update state to PROCESSING
            self.pgdb.update("DOCUMENTS_MASTER",
                             text(f"id='{process_id}'"),
                             {"state": "PROCESSING"})

            # Download document from S3
            temp_path = f"/tmp/{process_id}"
            if not self.s3_storage.download_file(doc_master.s3_location, temp_path):
                raise RuntimeError("Failed to download document from S3")

            # Process document with Graph Engine
            with open(temp_path, 'r') as f:
                content = f.read()
                self.graph_engine.doc_id = process_id
                self.graph_engine.doc_name = doc_master.name
                self.graph_engine.insert([{"content": content}])

            # Update state to COMPLETED
            self.pgdb.update("DOCUMENTS_MASTER",
                             text(f"id='{process_id}'"),
                             {"state": "COMPLETED"})

        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            self.pgdb.update("DOCUMENTS_MASTER",
                             text(f"id='{process_id}'"),
                             {"state": "FAILED"})
            raise
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    processor = DocumentProcessor()
    file_path = './examples/amazon/citibank-amazon.pdf'

    try:
        with open(file_path, 'rb') as file:
            file_content = file.read()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

    data = {
        "parent_id": "U-241278G",
        "name": "citibank-amazon.pdf"
    }
    process_id = asyncio.run(processor.start_processing(file_content, data))
    logger.info(f"Processing started for {process_id}")
    print(f"Processing started for {process_id}")
