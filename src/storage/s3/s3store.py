import os

import boto3
from botocore.client import Config
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
from dynaconf import settings
from src.utils.log import logger


class S3Storage:

    def __init__(self, ):

        self.endpoint_url = settings.get("S3_ENDPOINT_URL", "http://localhost:9000")
        self.access_key = settings.get("S3_ACCESS_KEY", "admin")
        self.secret_key = settings.get("S3_SECRET_KEY", "password")
        self.bucket_name = settings.get("S3_BUCKET_NAME", "legal")

        # Initialize the S3 client
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version="s3v4"),
        )

    def upload_file(self, file_path: str, object_name: str = None):
        """
        Upload a file to the S3 bucket.

        :param file_path: Path to the file to upload
        :param object_name: Name of the object in S3 (defaults to the file name)
        :return: True if successful, False otherwise
        """
        if object_name is None:
            object_name = os.path.basename(file_path)  # Use the file name as the object name

        try:
            self.s3_client.upload_file(file_path, self.bucket_name, object_name)
            logger.info(f"File '{file_path}' uploaded to '{self.bucket_name}/{object_name}'")
            return True
        except Exception as e:
            logger.info(f"An unexpected error occurred: {e}")
            return False

    def upload_file_content(self, file_content: bytes, file_name: str = None):
        """
        Upload a file to the S3 bucket.
        Args :
        file : UploadFile
            The file to upload.
        name : str
            The name of the file.

        """

        try:
            # Upload the file to S3
            self.s3_client.put_object(Bucket=self.bucket_name, Key=file_name, Body=file_content)
            logger.info(f"filename: {file_name} message: File uploaded successfully")
            return True
        except Exception as e:
            logger.info(f"An unexpected error occurred: {e}")
            return False

    def download_file(self, object_name: str, download_path: str):
        """
        Download a file from the S3 bucket.

        :param object_name: Name of the object in S3
        :param download_path: Path to save the downloaded file
        :return: True if successful, False otherwise
        """
        try:
            self.s3_client.download_file(self.bucket_name, object_name, download_path)
            logger.info(f"File '{object_name}' downloaded to '{download_path}'")
            return True
        except NoCredentialsError:
            logger.info("Credentials not available.")
            return False
        except PartialCredentialsError:
            logger.info("Incomplete credentials provided.")
            return False
        except ClientError as e:
            logger.info(f"Client error: {e}")
            return False

    def delete_file(self, object_name: str):
        """
        Delete a file from the S3 bucket.

        :param object_name: Name of the object in S3
        :return: True if successful, False otherwise
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=object_name)
            logger.info(f"File '{object_name}' deleted from '{self.bucket_name}'")
            return True
        except NoCredentialsError:
            logger.info("Credentials not available.")
            return False
        except PartialCredentialsError:
            logger.info("Incomplete credentials provided.")
            return False
        except ClientError as e:
            logger.info(f"Client error: {e}")
            return False

    def list_files(self):
        """
        List all files in the S3 bucket.

        :return: List of file names in the bucket
        """
        try:
            response = self.s3_client.list_objects(Bucket=self.bucket_name)
            if "Contents" in response:
                files = [obj["Key"] for obj in response["Contents"]]
                logger.info(f"Files in '{self.bucket_name}': {files}")
                return files
            else:
                logger.info(f"No files found in '{self.bucket_name}'")
                return []
        except NoCredentialsError:
            logger.info("Credentials not available.")
            return []
        except PartialCredentialsError:
            logger.info("Incomplete credentials provided.")
            return []
        except ClientError as e:
            logger.info(f"Client error: {e}")
            return []


    def check_folder_exists(self, folder_name: str):
        if not folder_name.endswith('/'):
            folder_name += '/'

        # Check if the folder already exists
        response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=folder_name, Delimiter='/')
        if 'CommonPrefixes' in response and any(prefix['Prefix'] == folder_name for prefix in response['CommonPrefixes']):
            logger.info(f"Folder '{folder_name}' already exists in '{self.bucket_name}'")
            return True
        else:
            logger.info(f"Folder '{folder_name}' does not exist in '{self.bucket_name}'")
            return False


    def create_folder(self, folder_name: str):
        """
        Create a folder in the S3 bucket.

        :param folder_name: Name of the folder to create
        :return: True if successful, False otherwise
        """
        if folder_name and not self.check_folder_exists(folder_name):
            if not folder_name.endswith('/'):
                folder_name += '/'
            try:
                self.s3_client.put_object(Bucket=self.bucket_name, Key=folder_name)
                logger.info(f"Folder '{folder_name}' created in '{self.bucket_name}'")
                return True
            except NoCredentialsError:
                logger.info("Credentials not available.")
                return False
            except PartialCredentialsError:
                logger.info("Incomplete credentials provided.")
                return False
            except ClientError as e:
                logger.info(f"Client error: {e}")
                return False
        else:
            return False

    def delete_folder(self, folder_name: str):
        """
        Delete a folder and its contents from the S3 bucket.

        :param folder_name: Name of the folder to delete
        :return: True if successful, False otherwise
        """
        if not folder_name.endswith('/'):
            folder_name += '/'
        try:
            # List all objects in the folder
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=folder_name)
            if 'Contents' in response:
                # Delete all objects in the folder
                contents = response['Contents']
                objects_to_delete = [{'Key': obj['Key']} for obj in contents]
                self.s3_client.delete_objects(Bucket=self.bucket_name, Delete={'Objects': objects_to_delete})
            logger.info(f"Folder '{folder_name}' and its contents deleted from '{self.bucket_name}'")
            return True
        except NoCredentialsError:
            logger.info("Credentials not available.")
            return False
        except PartialCredentialsError:
            logger.info("Incomplete credentials provided.")
            return False
        except ClientError as e:
            logger.info(f"Client error: {e}")
            return False


if __name__ == "__main__":
    # Initialize the S3Storage class
    s3_storage = S3Storage()

    filepath = os.path.join(os.getcwd(), "examples.txt")
    # Upload a file
    s3_storage.upload_file(filepath)

    # List files in the bucket
    s3_storage.list_files()

    # Download a file
    s3_storage.download_file("example.txt", "downloaded_example.txt")
