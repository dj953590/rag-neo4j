from typing import Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
from src.utils.log import logger as log
from src.storage.s3.s3store import S3Storage

app = FastAPI()


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), data: dict = None) -> Any:
    """
    Upload a file to the S3 bucket.
    Args :
    file : UploadFile
        The file to upload.
    name : str
        The name of the file.
    group : str
        The group of the file.

    """
    if data is None:
        log.info("No data provided")
        return False

    try:
        parent_id = data.get("parent_id")
        state = data.get("state")
        file_name = data.get("name")

        if not parent_id or not state or not file_name:
            log.info("Incomplete data provided")
            return False

        file_content = await file.read()
        s3store = S3Storage()
        s3store.create_folder(parent_id)
        object_name = f"{parent_id}/{file_name}"
        s3store.upload_file(file_content, object_name)
        log.info(f"File uploaded to '{s3store.bucket_name}/{object_name}'")
        return True
    except (NoCredentialsError, PartialCredentialsError, ClientError) as e:
        log.info(f"Error uploading file to S3: {e}")
        return False

@app.get("/query")
async def query() -> HTMLResponse:
    # Return an HTML response
    return HTMLResponse(content="<h1>Querying the S3 bucket</h1>")

# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
