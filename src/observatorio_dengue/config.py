"""Configurações centralizadas do projeto, carregadas do .env."""

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Configurações tipadas e validadas do projeto."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_raw_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw")
    data_processed_dir: Path = Field(default=PROJECT_ROOT / "data" / "processed")
    duckdb_path: Path = Field(default=PROJECT_ROOT / "data" / "processed" / "observatorio.duckdb")

    gee_project_id: str = Field(default="")
    gee_service_account: str = Field(default="")

    log_level: str = Field(default="INFO")

    def ensure_dirs(self) -> None:
        """Garante que todos os diretórios existam."""
        for path in [self.data_raw_dir, self.data_processed_dir]:
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()