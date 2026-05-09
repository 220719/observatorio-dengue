"""Coleta de dados de dengue via API do InfoDengue.

Documentação da API: https://info.dengue.mat.br/services/api
A API retorna dados semanais com campos:
- casos / casos_est: notificados / estimados com nowcasting
- nivel: alerta epidemiológico (1=verde, 2=amarelo, 3=laranja, 4=vermelho)
- p_inc100k: incidência por 100 mil habitantes
- tempmin, umidmin: clima embutido (não usaremos, preferimos Open-Meteo)
- SE: semana epidemiológica no formato YYYYWW (ex: 202401 = 2024 semana 1)
"""

from io import StringIO

import pandas as pd
import requests
from loguru import logger

INFODENGUE_BASE_URL = "https://info.dengue.mat.br/api/alertcity"
DEFAULT_TIMEOUT_SECONDS = 30


def coletar_municipio(
    geocode: int,
    nome_municipio: str,
    ano_inicio: int,
    ano_fim: int,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    """Coleta dados semanais de dengue para um município via InfoDengue.

    A API limita cada requisição a 1 ano. Esta função itera por ano e concatena
    os resultados. Se algum ano falhar, registra warning mas continua com os outros.

    Args:
        geocode: Código IBGE do município (ex: 4115200 para Maringá).
        nome_municipio: Nome legível, adicionado como coluna ao DataFrame.
        ano_inicio: Primeiro ano a coletar (inclusivo).
        ano_fim: Último ano a coletar (inclusivo).
        timeout: Timeout em segundos por requisição.

    Returns:
        DataFrame com dados semanais, contendo no mínimo as colunas:
        SE, casos, casos_est, nivel, p_inc100k, municipio, geocode.
        DataFrame vazio se todas as requisições falharem.

    Raises:
        ValueError: Se ano_inicio > ano_fim.
    """
    if ano_inicio > ano_fim:
        raise ValueError(f"ano_inicio ({ano_inicio}) deve ser <= ano_fim ({ano_fim})")

    frames: list[pd.DataFrame] = []

    for ano in range(ano_inicio, ano_fim + 1):
        params = {
            "geocode": geocode,
            "disease": "dengue",
            "format": "csv",
            "ew_start": 1,
            "ew_end": 53,  # 53 cobre anos com 53 semanas ISO; API ignora se não houver
            "ey_start": ano,
            "ey_end": ano,
        }

        try:
            response = requests.get(INFODENGUE_BASE_URL, params=params, timeout=timeout)
            response.raise_for_status()
            df_ano = pd.read_csv(StringIO(response.text))

            if df_ano.empty:
                logger.warning(f"{nome_municipio} ({ano}): API retornou DataFrame vazio")
                continue

            df_ano["municipio"] = nome_municipio
            df_ano["geocode"] = geocode
            frames.append(df_ano)
            logger.debug(f"{nome_municipio} ({ano}): {len(df_ano)} semanas coletadas")

        except requests.RequestException as e:
            logger.warning(f"{nome_municipio} ({ano}): falha na API — {e}")
        except pd.errors.ParserError as e:
            logger.warning(f"{nome_municipio} ({ano}): falha ao parsear CSV — {e}")

    if not frames:
        logger.error(f"{nome_municipio}: nenhum ano coletado com sucesso ({ano_inicio}–{ano_fim})")
        return pd.DataFrame()

    resultado = pd.concat(frames, ignore_index=True)
    logger.info(f"{nome_municipio}: {len(resultado)} semanas totais ({ano_inicio}–{ano_fim})")
    return resultado


def coletar_municipios(
    municipios: dict[str, int],
    ano_inicio: int,
    ano_fim: int,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    """Coleta dados de dengue para múltiplos municípios.

    Args:
        municipios: Dict {nome: geocode_ibge}.
        ano_inicio: Primeiro ano a coletar.
        ano_fim: Último ano a coletar.
        timeout: Timeout por requisição.

    Returns:
        DataFrame consolidado com dados de todos os municípios.
        Inclui coluna 'municipio' e 'geocode' para identificação.
    """
    if not municipios:
        raise ValueError("dict de municípios não pode estar vazio")

    logger.info(f"Iniciando coleta de {len(municipios)} municípios ({ano_inicio}–{ano_fim})")

    frames: list[pd.DataFrame] = []
    for nome, geocode in municipios.items():
        df = coletar_municipio(
            geocode=geocode,
            nome_municipio=nome,
            ano_inicio=ano_inicio,
            ano_fim=ano_fim,
            timeout=timeout,
        )
        if not df.empty:
            frames.append(df)

    if not frames:
        logger.error("Nenhum município coletado com sucesso")
        return pd.DataFrame()

    resultado = pd.concat(frames, ignore_index=True)
    logger.info(
        f"Coleta finalizada: {len(resultado)} registros, "
        f"{resultado['municipio'].nunique()} municípios"
    )
    return resultado


def parse_semana_epidemiologica(se: int | str) -> tuple[int, int]:
    """Decompõe campo SE do InfoDengue em (ano, semana).

    O InfoDengue retorna SE no formato YYYYWW como inteiro ou string
    (ex: 202401 = 2024 semana 1, 202053 = 2020 semana 53).

    Args:
        se: Valor do campo SE.

    Returns:
        Tupla (ano, semana).

    Raises:
        ValueError: Se o formato não for YYYYWW válido.
    """
    se_str = str(int(se))  # int() remove zeros decimais se vier como float
    if len(se_str) != 6:
        raise ValueError(f"SE inválido: {se!r}. Esperado formato YYYYWW de 6 dígitos.")

    ano = int(se_str[:4])
    semana = int(se_str[4:])

    if not (2000 <= ano <= 2100):
        raise ValueError(f"Ano fora de range esperado em SE={se}: {ano}")
    if not (1 <= semana <= 53):
        raise ValueError(f"Semana fora de range esperado em SE={se}: {semana}")

    return ano, semana
