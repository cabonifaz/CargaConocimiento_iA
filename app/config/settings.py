from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    COMPANY_FILES_DIR: Path = Field(default=Path("company_files"))
    RECURSIVE_SCAN: bool = True
    MAX_PDF_MB: int = 100
    EXCLUDE_GLOBS: list[str] = ["**/~$*", "**/.DS_Store", "**/._*"]

    class Config:
        env_file = ".env"

settings = Settings()
