from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class StoragePort(ABC):
    @abstractmethod
    def get_object(self, bucket: str, key: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_size(self, bucket: str, key: str) -> Optional[int]:
        pass

    @abstractmethod
    def put_object(self, bucket: str, key: str, data: bytes) -> bool:
        pass