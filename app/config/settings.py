# src/app/config/settings.py
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    COMPANY_FILES_DIR: Path = Field(default=Path("company_files"))
    OUTPUT_DIR: Path = Field(default=Path("data/output"))
    RECURSIVE_SCAN: bool = True
    MAX_PDF_MB: int = 100
    EXCLUDE_GLOBS: list[str] = ["**/~$*", "**/.DS_Store", "**/._*"]

    AWS_PROFILE: Optional[str] = None
    BEDROCK_REGION: str = "us-east-1"
    BEDROCK_MODEL_ID: str = "amazon.titan-embed-text-v2:0"
    BEDROCK_EMBED_DIM: Optional[int] = None
    BEDROCK_BATCH_SIZE: int = 64
    BEDROCK_TIMEOUT_SECS: int = 30

settings = Settings()
