from abc import ABC, abstractmethod
from typing import Optional


class FileStorage(ABC):
    @abstractmethod
    def is_pdf_by_head(self, bucket: str, key: str) -> bool:
        pass

    @abstractmethod
    def get_object_size(self, bucket: str, key: str) -> int:
        pass

    @abstractmethod
    def delete_object(self, bucket: str, key: str) -> None:
        pass

    @abstractmethod
    def download_to_tmp(self, bucket: str, key: str) -> str:
        pass