"""Persistência de dados em DuckDB.

DuckDB é um banco analítico embutido (sem servidor). O banco é um único
arquivo .duckdb cujo path vem de Settings.duckdb_path.

Tabelas:
- dengue_raw: dados brutos da API InfoDengue (semanais, por município)
- clima_diario: dados brutos da API Open-Meteo (diários, por ponto geográfico)
- clima_semanal: clima agregado para semana epidemiológica (ISO 8601)

Cada tabela tem coluna 'data_carga' (TIMESTAMP) para rastreabilidade.
Tabelas não têm PRIMARY KEY estrita — permite reinserção sem erro.
Para deduplicar nas análises, use ROW_NUMBER() OVER (ORDER BY data_carga DESC).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd
from loguru import logger

from observatorio_dengue.config import settings

# DDL (Data Definition Language): definição do schema das tabelas.
# Idempotente — pode ser executado várias vezes sem efeito colateral.
SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS dengue_raw (
    geocode INTEGER NOT NULL,
    municipio VARCHAR NOT NULL,
    se INTEGER NOT NULL,
    ano_epi INTEGER NOT NULL,
    semana_epi INTEGER NOT NULL,
    casos INTEGER,
    casos_est DOUBLE,
    nivel INTEGER,
    p_inc100k DOUBLE,
    data_carga TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS clima_diario (
    latitude DOUBLE NOT NULL,
    longitude DOUBLE NOT NULL,
    data DATE NOT NULL,
    temperature_2m_mean DOUBLE,
    temperature_2m_max DOUBLE,
    temperature_2m_min DOUBLE,
    precipitation_sum DOUBLE,
    relative_humidity_2m_mean DOUBLE,
    data_carga TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS clima_semanal (
    latitude DOUBLE NOT NULL,
    longitude DOUBLE NOT NULL,
    ano_epi INTEGER NOT NULL,
    semana_epi INTEGER NOT NULL,
    temperature_2m_mean DOUBLE,
    temperature_2m_max DOUBLE,
    temperature_2m_min DOUBLE,
    precipitation_sum DOUBLE,
    relative_humidity_2m_mean DOUBLE,
    dias_validos INTEGER,
    data_carga TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS satelite_municipio_semanal (
    municipio VARCHAR NOT NULL,
    ano_epi INTEGER NOT NULL,
    semana_epi INTEGER NOT NULL,
    ndvi DOUBLE,
    lst_night_c DOUBLE,
    dias_validos INTEGER,
    data_carga TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


@contextmanager
def get_connection(db_path: Path | None = None) -> Iterator[duckdb.DuckDBPyConnection]:
    """Abre conexão com o DuckDB como context manager (auto-close).

    Args:
        db_path: Path para o arquivo .duckdb. Se None, usa settings.duckdb_path.

    Yields:
        Conexão DuckDB pronta para uso.

    Example:
        >>> with get_connection() as con:
        ...     con.execute("SELECT 1").fetchone()
    """
    path = db_path if db_path is not None else settings.duckdb_path
    path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(path))
    try:
        yield con
    finally:
        con.close()


def criar_schema(db_path: Path | None = None) -> None:
    """Cria as 3 tabelas se ainda não existirem (idempotente).

    Args:
        db_path: Path opcional para o banco. Default: settings.duckdb_path.
    """
    with get_connection(db_path) as con:
        con.execute(SCHEMA_DDL)
    logger.info(f"Schema criado/verificado em {db_path or settings.duckdb_path}")


def salvar_dengue(df: pd.DataFrame, db_path: Path | None = None) -> int:
    """Persiste DataFrame de dengue em dengue_raw.

    O DataFrame deve vir de etl.infodengue.coletar_municipio() ou
    coletar_municipios(), com colunas: geocode, municipio, SE, casos,
    casos_est, nivel, p_inc100k.

    Args:
        df: DataFrame com dados de dengue da API InfoDengue.
        db_path: Path opcional para o banco.

    Returns:
        Número de linhas inseridas.

    Raises:
        ValueError: Se DataFrame não tiver as colunas necessárias.
    """
    if df.empty:
        logger.warning("DataFrame de dengue vazio, nada a salvar")
        return 0

    colunas_necessarias = {"geocode", "municipio", "SE", "casos", "casos_est", "nivel", "p_inc100k"}
    faltando = colunas_necessarias - set(df.columns)
    if faltando:
        raise ValueError(f"Colunas obrigatórias faltando: {faltando}")

    # Decompõe SE (formato YYYYWW) em ano_epi e semana_epi.
    # Reaproveita o parser do módulo infodengue para garantir consistência.
    from observatorio_dengue.etl.infodengue import parse_semana_epidemiologica

    df_preparado = df.copy()
    epi = df_preparado["SE"].apply(parse_semana_epidemiologica)
    df_preparado["ano_epi"] = epi.apply(lambda x: x[0])
    df_preparado["semana_epi"] = epi.apply(lambda x: x[1])
    df_preparado = df_preparado.rename(columns={"SE": "se"})

    colunas_destino = [
        "geocode",
        "municipio",
        "se",
        "ano_epi",
        "semana_epi",
        "casos",
        "casos_est",
        "nivel",
        "p_inc100k",
    ]
    df_preparado = df_preparado[colunas_destino]

    with get_connection(db_path) as con:
        con.register("df_temp", df_preparado)
        con.execute(f"INSERT INTO dengue_raw ({', '.join(colunas_destino)}) SELECT * FROM df_temp")
        con.unregister("df_temp")

    logger.info(f"Dengue: {len(df_preparado)} linhas inseridas em dengue_raw")
    return len(df_preparado)


def salvar_clima_diario(
    df: pd.DataFrame,
    latitude: float,
    longitude: float,
    db_path: Path | None = None,
) -> int:
    """Persiste DataFrame de clima diário em clima_diario.

    Args:
        df: DataFrame de etl.openmeteo.coletar_clima_diario(),
            com coluna 'data' e variáveis climáticas.
        latitude: Latitude do ponto coletado.
        longitude: Longitude do ponto coletado.
        db_path: Path opcional para o banco.

    Returns:
        Número de linhas inseridas.
    """
    if df.empty:
        logger.warning("DataFrame de clima diário vazio, nada a salvar")
        return 0

    if "data" not in df.columns:
        raise ValueError("DataFrame deve ter coluna 'data'")

    df_preparado = df.copy()
    df_preparado["latitude"] = latitude
    df_preparado["longitude"] = longitude

    colunas_destino = [
        "latitude",
        "longitude",
        "data",
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "relative_humidity_2m_mean",
    ]
    df_preparado = df_preparado[colunas_destino]

    with get_connection(db_path) as con:
        con.register("df_temp", df_preparado)
        con.execute(
            f"INSERT INTO clima_diario ({', '.join(colunas_destino)}) SELECT * FROM df_temp"
        )
        con.unregister("df_temp")

    logger.info(f"Clima diário: {len(df_preparado)} linhas inseridas em clima_diario")
    return len(df_preparado)


def salvar_clima_semanal(
    df: pd.DataFrame,
    latitude: float,
    longitude: float,
    db_path: Path | None = None,
) -> int:
    """Persiste DataFrame de clima semanal em clima_semanal.

    Args:
        df: DataFrame de etl.openmeteo.agregar_para_semana_epidemiologica().
        latitude: Latitude do ponto coletado.
        longitude: Longitude do ponto coletado.
        db_path: Path opcional para o banco.

    Returns:
        Número de linhas inseridas.
    """
    if df.empty:
        logger.warning("DataFrame de clima semanal vazio, nada a salvar")
        return 0

    colunas_obrigatorias = {"ano_epi", "semana_epi"}
    faltando = colunas_obrigatorias - set(df.columns)
    if faltando:
        raise ValueError(f"Colunas obrigatórias faltando: {faltando}")

    df_preparado = df.copy()
    df_preparado["latitude"] = latitude
    df_preparado["longitude"] = longitude

    colunas_destino = [
        "latitude",
        "longitude",
        "ano_epi",
        "semana_epi",
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "relative_humidity_2m_mean",
        "dias_validos",
    ]
    df_preparado = df_preparado[colunas_destino]

    with get_connection(db_path) as con:
        con.register("df_temp", df_preparado)
        con.execute(
            f"INSERT INTO clima_semanal ({', '.join(colunas_destino)}) SELECT * FROM df_temp"
        )
        con.unregister("df_temp")

    logger.info(f"Clima semanal: {len(df_preparado)} linhas inseridas em clima_semanal")
    return len(df_preparado)


def carregar(query: str, db_path: Path | None = None) -> pd.DataFrame:
    """Executa SQL e retorna resultado como DataFrame.

    Útil para análises ad-hoc, debugging, exploração de dados.

    Args:
        query: Comando SQL SELECT.
        db_path: Path opcional para o banco.

    Returns:
        DataFrame com o resultado da query.

    Example:
        >>> df = carregar("SELECT * FROM dengue_raw WHERE municipio = 'Maringá' LIMIT 10")
    """
    with get_connection(db_path) as con:
        return con.execute(query).fetchdf()

def salvar_satelite_semanal(
    df: pd.DataFrame,
    municipio: str,
    db_path: Path | None = None,
) -> int:
    """Persiste DataFrame de índices de satélite em satelite_municipio_semanal.

    Args:
        df: DataFrame de etl.gee + openmeteo.agregar_para_semana_epidemiologica().
            Deve conter ao menos: ano_epi, semana_epi.
            Colunas opcionais reconhecidas: ndvi, lst_night_c, dias_validos.
        municipio: Nome do município (gravado em todas as linhas).
        db_path: Path opcional para o banco.

    Returns:
        Número de linhas inseridas.

    Raises:
        ValueError: Se colunas obrigatórias estiverem faltando.
    """
    if df.empty:
        logger.warning("DataFrame de satélite vazio, nada a salvar")
        return 0

    colunas_obrigatorias = {"ano_epi", "semana_epi"}
    faltando = colunas_obrigatorias - set(df.columns)
    if faltando:
        raise ValueError(f"Colunas obrigatórias faltando: {faltando}")

    df_preparado = df.copy()
    df_preparado["municipio"] = municipio

    # Garante que colunas opcionais existam (NULL se não vieram)
    for col in ("ndvi", "lst_night_c", "dias_validos"):
        if col not in df_preparado.columns:
            df_preparado[col] = None

    colunas_destino = [
        "municipio",
        "ano_epi",
        "semana_epi",
        "ndvi",
        "lst_night_c",
        "dias_validos",
    ]
    df_preparado = df_preparado[colunas_destino]

    with get_connection(db_path) as con:
        con.register("df_temp", df_preparado)
        con.execute(
            f"INSERT INTO satelite_municipio_semanal ({', '.join(colunas_destino)}) "
            f"SELECT * FROM df_temp"
        )
        con.unregister("df_temp")

    logger.info(
        f"Satélite semanal: {len(df_preparado)} linhas inseridas em "
        f"satelite_municipio_semanal"
    )
    return len(df_preparado)