"""Coleta de dados climáticos diários via Open-Meteo Archive API.

Documentação: https://open-meteo.com/en/docs/historical-weather-api

A API retorna dados diários (1 valor por dia) em arrays paralelos dentro de 'daily':
- time: lista de datas ISO 8601
- temperature_2m_mean / max / min: temperatura em °C
- precipitation_sum: chuva total em mm
- relative_humidity_2m_mean: umidade relativa em %

A coordenada solicitada é "snap-ada" para o centro da célula da grade ERA5
(~10km), então pequenos ajustes em lat/lon retornado são esperados.
"""

from datetime import date

import pandas as pd
import requests
from epiweeks import Week
from loguru import logger

OPENMETEO_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_TIMEOUT_SECONDS = 30

# Variáveis diárias coletadas. Cada uma vira uma coluna no DataFrame.
DAILY_VARIABLES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "relative_humidity_2m_mean",
]


def coletar_clima_diario(
    latitude: float,
    longitude: float,
    data_inicio: date,
    data_fim: date,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    """Coleta dados climáticos diários para um ponto geográfico.

    Args:
        latitude: Latitude em graus decimais (entre -90 e 90).
        longitude: Longitude em graus decimais (entre -180 e 180).
        data_inicio: Primeira data a coletar (inclusiva).
        data_fim: Última data a coletar (inclusiva).
        timeout: Timeout da requisição em segundos.

    Returns:
        DataFrame com colunas:
        - data (datetime64[ns]): data do registro
        - temperature_2m_mean, _max, _min (float, °C)
        - precipitation_sum (float, mm)
        - relative_humidity_2m_mean (float, %)
        DataFrame vazio se a requisição falhar.

    Raises:
        ValueError: Se data_inicio > data_fim ou coordenadas fora de range.
    """
    if data_inicio > data_fim:
        raise ValueError(f"data_inicio ({data_inicio}) deve ser <= data_fim ({data_fim})")
    if not (-90 <= latitude <= 90):
        raise ValueError(f"Latitude fora de range [-90, 90]: {latitude}")
    if not (-180 <= longitude <= 180):
        raise ValueError(f"Longitude fora de range [-180, 180]: {longitude}")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": data_inicio.isoformat(),
        "end_date": data_fim.isoformat(),
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "America/Sao_Paulo",
    }

    try:
        response = requests.get(OPENMETEO_BASE_URL, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as e:
        logger.error(f"Falha na API Open-Meteo: {e}")
        return pd.DataFrame()
    except ValueError as e:
        logger.error(f"Resposta inválida do Open-Meteo: {e}")
        return pd.DataFrame()

    if "daily" not in payload:
        logger.error(f"Resposta sem campo 'daily': {payload}")
        return pd.DataFrame()

    daily = payload["daily"]
    df = pd.DataFrame(
        {"data": pd.to_datetime(daily["time"])} | {var: daily[var] for var in DAILY_VARIABLES}
    )

    logger.info(
        f"Clima coletado: {len(df)} dias ({data_inicio} a {data_fim}) em ({latitude}, {longitude})"
    )
    return df


def agregar_para_semana_epidemiologica(df_diario: pd.DataFrame) -> pd.DataFrame:
    """Agrega dados diários para semanas epidemiológicas (ISO 8601).

    Usa epiweeks (que respeita anos com 53 semanas) para mapear cada data
    para sua semana epidemiológica. Aplica agregações apropriadas por variável:
    - Chuva: soma (mm/semana é mais informativo que média)
    - Demais: média

    Args:
        df_diario: DataFrame retornado por coletar_clima_diario(),
            com coluna 'data' e variáveis climáticas.

    Returns:
        DataFrame agregado com colunas:
        - ano_epi (int): ano epidemiológico ISO
        - semana_epi (int): semana epidemiológica ISO (1 a 53)
        - temperature_2m_mean, _max, _min (médias semanais)
        - precipitation_sum (soma semanal)
        - relative_humidity_2m_mean (média semanal)
        - dias_validos (int): quantos dias contribuíram para a agregação

    Raises:
        ValueError: Se df_diario não tiver as colunas esperadas.
    """
    if df_diario.empty:
        logger.warning("DataFrame diário vazio, retornando DataFrame vazio")
        return pd.DataFrame()

    if "data" not in df_diario.columns:
        raise ValueError("df_diario deve ter coluna 'data'")

    df = df_diario.copy()

    epi_info = df["data"].apply(_data_para_semana_epi)
    df["ano_epi"] = epi_info.apply(lambda x: x[0])
    df["semana_epi"] = epi_info.apply(lambda x: x[1])

    agregacoes = {
        "temperature_2m_mean": "mean",
        "temperature_2m_max": "mean",
        "temperature_2m_min": "mean",
        "precipitation_sum": "sum",
        "relative_humidity_2m_mean": "mean",
        "data": "count",
    }

    agregacoes = {k: v for k, v in agregacoes.items() if k in df.columns}

    df_semanal = (
        df.groupby(["ano_epi", "semana_epi"])
        .agg(agregacoes)
        .rename(columns={"data": "dias_validos"})
        .reset_index()
    )

    logger.info(f"Agregação semanal: {len(df_diario)} dias → {len(df_semanal)} semanas")
    return df_semanal


def _data_para_semana_epi(d: pd.Timestamp) -> tuple[int, int]:
    """Converte uma data para (ano_epi, semana_epi) usando ISO 8601.

    epiweeks com system='iso' segue padrão ISO 8601 (segunda-feira como
    início da semana, semana 1 contém a primeira quinta-feira do ano).

    Args:
        d: data a converter.

    Returns:
        Tupla (ano_epi, semana_epi).
    """
    week = Week.fromdate(d.date(), system="iso")
    return week.year, week.week
