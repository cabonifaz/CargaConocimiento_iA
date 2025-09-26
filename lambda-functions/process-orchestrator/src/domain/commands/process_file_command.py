from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessFileCommand:
    bucket: str
    key: str
    heavy_queue_url: str
    light_queue_url: str