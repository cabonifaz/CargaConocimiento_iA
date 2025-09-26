from abc import ABC, abstractmethod
from typing import Dict, Any


class MessageQueue(ABC):
    @abstractmethod
    def send_message(self, queue_url: str, message_body: Dict[str, Any], message_attributes: Dict[str, Any]) -> str:
        pass