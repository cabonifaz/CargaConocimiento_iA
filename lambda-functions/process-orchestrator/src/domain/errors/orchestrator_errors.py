class OrchestratorError(Exception):
    pass


class FileProcessingError(OrchestratorError):
    def __init__(self, bucket: str, key: str, message: str):
        self.bucket = bucket
        self.key = key
        super().__init__(f"Error processing {bucket}/{key}: {message}")


class RoutingError(OrchestratorError):
    def __init__(self, queue_type: str, message: str):
        self.queue_type = queue_type
        super().__init__(f"Error routing to {queue_type} queue: {message}")