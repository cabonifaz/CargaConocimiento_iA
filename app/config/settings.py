# src/app/config/settings.py
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    COMPANY_FILES_DIR: Path = Field(default=Path("company_files"))
    RECURSIVE_SCAN: bool = True
    MAX_PDF_MB: int = 100
    EXCLUDE_GLOBS: list[str] = ["**/~$*", "**/.DS_Store", "**/._*"]

settings = Settings()
