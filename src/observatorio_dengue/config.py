"""Configurações centralizadas do projeto.

Separa duas dimensões:
- Settings: configurações de infraestrutura (paths, log level, credenciais).
  Carregadas do arquivo .env, podem variar por ambiente.
- DengueConfig: configurações de domínio (municípios, lag, coordenadas).
  Constantes do projeto, raramente mudam, ficam versionadas no código.
"""

from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Configurações de infraestrutura, tipadas e validadas.

    Carregadas do .env. Valores podem ser sobrescritos por variáveis de ambiente.
    """

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
        """Garante que todos os diretórios necessários existam."""
        for path in [self.data_raw_dir, self.data_processed_dir]:
            path.mkdir(parents=True, exist_ok=True)


class DengueConfig(BaseModel):
    """Configurações de domínio do Observatório Dengue × Clima.

    Constantes do estudo: região analisada, período, parâmetros temporais.
    """

    # Geocódigos IBGE dos municípios da região metropolitana de Maringá
    municipios: dict[str, int] = Field(
        default_factory=lambda: {
            "Maringá": 4115200,
            "Sarandi": 4126256,
            "Paiçandu": 4117206,
            "Mandaguari": 4114203,
            "Marialva": 4114807,
            "Mandaguaçu": 4114104,
            "Astorga": 4102307,
        }
    )

    # Período coberto pela análise
    ano_inicio: int = Field(default=2020, ge=2010, le=2030)
    ano_fim: int = Field(default=2025, ge=2010, le=2030)

    # Coordenadas geográficas para coleta de clima (Maringá-PR centroide)
    latitude: float = Field(default=-23.4209, ge=-90, le=90)
    longitude: float = Field(default=-51.9331, ge=-180, le=180)

    # Defasagem temporal entre clima e casos (semanas).
    # 4 semanas = ciclo de desenvolvimento do Aedes aegypti + período intrínseco
    lag_semanas: int = Field(default=4, ge=1, le=12)

    @property
    def municipio_principal(self) -> str:
        """Retorna o nome do município principal (primeiro da lista)."""
        return next(iter(self.municipios))

    @property
    def geocode_principal(self) -> int:
        """Retorna o geocode do município principal."""
        return self.municipios[self.municipio_principal]


settings = Settings()
settings.ensure_dirs()

dengue_config = DengueConfig()
