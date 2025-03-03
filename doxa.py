from typing import Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

from src.storage.s3.s3store import S3Storage

app = FastAPI()


@app.post("/upload")
async def upload_file(file_name: str, group: str, file: UploadFile = File(...), ) -> Any:
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
    try:
        file_content = await file.read()
        s3store = S3Storage()
        s3store.create_folder(group)
        object_name = f"{group}/{file_name}"
        s3store.upload_file(file_content, object_name)
        print(f"File uploaded to '{s3store.bucket_name}/{object_name}'")
        return True
    except NoCredentialsError:
        print("Credentials not available.")
        return False
    except PartialCredentialsError:
        print("Incomplete credentials provided.")
        return False
    except ClientError as e:
        print(f"Client error: {e}")
        return False


@app.get("/query")
async def query() -> HTMLResponse:
    # Return an HTML response
    return HTMLResponse(content="<h1>Querying the S3 bucket</h1>")

# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
