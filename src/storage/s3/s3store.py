import os

import boto3
from botocore.client import Config
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError


class S3Storage:

    def __init__(self, endpoint_url: str, access_key: str, secret_key: str, bucket_name: str):

        self.endpoint_url = endpoint_url
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket_name = bucket_name

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
            print(f"File '{file_path}' uploaded to '{self.bucket_name}/{object_name}'")
            return True
        except FileNotFoundError:
            print(f"File '{file_path}' not found.")
            return False
        except NoCredentialsError:
            print("Credentials not available.")
            return False
        except PartialCredentialsError:
            print("Incomplete credentials provided.")
            return False
        except ClientError as e:
            print(f"Client error: {e}")
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
            print(f"File '{object_name}' downloaded to '{download_path}'")
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

    def delete_file(self, object_name: str):
        """
        Delete a file from the S3 bucket.

        :param object_name: Name of the object in S3
        :return: True if successful, False otherwise
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=object_name)
            print(f"File '{object_name}' deleted from '{self.bucket_name}'")
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

    def list_files(self):
        """
        List all files in the S3 bucket.

        :return: List of file names in the bucket
        """
        try:
            response = self.s3_client.list_objects(Bucket=self.bucket_name)
            if "Contents" in response:
                files = [obj["Key"] for obj in response["Contents"]]
                print(f"Files in '{self.bucket_name}': {files}")
                return files
            else:
                print(f"No files found in '{self.bucket_name}'")
                return []
        except NoCredentialsError:
            print("Credentials not available.")
            return []
        except PartialCredentialsError:
            print("Incomplete credentials provided.")
            return []
        except ClientError as e:
            print(f"Client error: {e}")
            return []


if __name__ == "__main__":
    # Initialize S3 storage client
    # Configuration for MinIO (local S3-compatible storage)
    endpoint_url = "http://localhost:9000"  # MinIO server URL
    access_key = "admin"  # MinIO access key
    secret_key = "password"  # MinIO secret key
    bucket_name = "legal"  # Bucket name

    # Initialize the S3Storage class
    s3_storage = S3Storage(endpoint_url, access_key, secret_key, bucket_name)

    filepath = os.path.join(os.getcwd(), "examples.txt")
    # Upload a file
    s3_storage.upload_file(filepath)

    # List files in the bucket
    s3_storage.list_files()

    # Download a file
    s3_storage.download_file("example.txt", "downloaded_example.txt")

