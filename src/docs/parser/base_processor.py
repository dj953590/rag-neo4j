from abc import ABC, abstractmethod


class BaseProcessor(ABC):
    def __init__(self, file_path: str):
        self.file_path = file_path

    @abstractmethod
    def markdown(self):
        """
        Extract text from the file.
        """
        pass

