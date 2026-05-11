"""Coleta de índices de satélite via Google Earth Engine.

Fornece NDVI (vegetação) e LST_Night (temperatura de superfície noturna)
para um polígono municipal, agregados diariamente. Os dados saem em formato
compatível com clima_diario para reusar a agregação semanal de openmeteo.py.

Convenções:
- NDVI: MODIS MOD13Q1 (composto de 16 dias, 250m) → forward-fill para diário
- LST_Night: MODIS MOD11A1 (diário, 1km) → conversão Kelvin → Celsius
- Geometria: FAO/GAUL/2015 level 2 (municípios brasileiros)
"""

from __future__ import annotations

import unicodedata
from datetime import date, timedelta

import ee
import pandas as pd
from loguru import logger

DATASET_NDVI = "MODIS/061/MOD13Q1"
DATASET_LST = "MODIS/061/MOD11A1"
DATASET_MUNICIPIOS = "FAO/GAUL/2015/level2"

ESCALA_NDVI = 250  # metros
ESCALA_LST = 1000  # metros
FATOR_NDVI = 10000  # MODIS NDVI vem escalado por 10000
FATOR_LST = 0.02  # MODIS LST: valor real (K) = DN * 0.02
KELVIN_PARA_CELSIUS = 273.15


def inicializar_gee(project_id: str = "observatorio-dengue-maringa") -> None:
    """Inicializa cliente do Earth Engine com o projeto especificado.

    Deve ser chamado uma vez por sessão antes de qualquer coleta.

    Args:
        project_id: ID do projeto GCP registrado no Earth Engine.

    Raises:
        ee.EEException: Se a autenticação falhar.
    """
    ee.Initialize(project=project_id)
    logger.info(f"GEE inicializado com projeto '{project_id}'")


def _remover_acentos(texto: str) -> str:
    """Remove acentos de uma string (NFD + filtro ASCII).

    FAO/GAUL/2015 usa nomes sem acentuação para municípios brasileiros
    (ex: 'Maringa', 'Agua Branca'), então normalizamos antes de filtrar.
    """
    nfd = unicodedata.normalize("NFD", texto)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def obter_geometria_municipio(nome: str = "Maringá") -> ee.Geometry:
    """Retorna a geometria de um município brasileiro via FAO/GAUL.

    Args:
        nome: Nome exato do município (com acentos, conforme FAO/GAUL).

    Returns:
        ee.Geometry do polígono municipal.

    Raises:
        ValueError: Se o município não for encontrado.
    """
    nome_normalizado = _remover_acentos(nome)
    colecao = ee.FeatureCollection(DATASET_MUNICIPIOS)
    feicao = colecao.filter(ee.Filter.eq("ADM2_NAME", nome_normalizado))

    n = feicao.size().getInfo()
    if n == 0:
        raise ValueError(f"município '{nome}' não encontrado em {DATASET_MUNICIPIOS}")
    if n > 1:
        logger.warning(f"{n} municípios chamados '{nome}' encontrados; usando o primeiro")

    geometria = feicao.first().geometry()
    logger.info(f"geometria de '{nome}' obtida ({n} feição(ões))")
    return geometria


def _extrair_serie_temporal(
    colecao: ee.ImageCollection,
    banda: str,
    geometria: ee.Geometry,
    escala: int,
) -> pd.DataFrame:
    """Extrai série temporal de uma banda sobre uma geometria.

    Para cada imagem da coleção, calcula a média da banda sobre o polígono
    e retorna como DataFrame com colunas ['data', banda].

    Args:
        colecao: ImageCollection já filtrada por data e região.
        banda: Nome da banda a extrair (ex: 'NDVI', 'LST_Night_1km').
        geometria: Polígono sobre o qual agregar.
        escala: Resolução em metros para reduceRegion.

    Returns:
        DataFrame com colunas 'data' (datetime) e [banda] (float).
    """

    def _processar_imagem(img: ee.Image) -> ee.Feature:
        valor = (
            img.select(banda)
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometria,
                scale=escala,
                maxPixels=1e9,
            )
            .get(banda)
        )
        return ee.Feature(
            None,
            {
                "data": img.date().format("YYYY-MM-dd"),
                banda: valor,
            },
        )

    feicoes = colecao.map(_processar_imagem)
    info = feicoes.getInfo()

    registros = []
    for f in info["features"]:
        props = f["properties"]
        if props.get(banda) is None:
            continue
        registros.append({"data": props["data"], banda: props[banda]})

    df = pd.DataFrame(registros)
    if df.empty:
        logger.warning(f"nenhum valor válido extraído para '{banda}'")
        return df

    df["data"] = pd.to_datetime(df["data"])
    df = df.sort_values("data").reset_index(drop=True)
    return df


def coletar_ndvi_diario(
    geometria: ee.Geometry,
    data_inicio: date,
    data_fim: date,
) -> pd.DataFrame:
    """Coleta NDVI MODIS MOD13Q1 e expande para diário via forward-fill.

    O MOD13Q1 é composto de 16 dias. Cada valor representa o melhor pixel
    dos 16 dias anteriores. Para integrar com séries diárias de clima,
    fazemos forward-fill: o valor do composto vale para todos os dias do
    período de 16 dias até o próximo composto.

    Args:
        geometria: Polígono sobre o qual agregar.
        data_inicio: Primeira data (inclusiva).
        data_fim: Última data (inclusiva).

    Returns:
        DataFrame com colunas 'data' e 'ndvi' (escala -1 a 1).
    """
    if data_inicio > data_fim:
        raise ValueError(f"data_inicio ({data_inicio}) > data_fim ({data_fim})")

    # Coleta compostos brutos
    colecao = (
        ee.ImageCollection(DATASET_NDVI)
        .filterDate(str(data_inicio), str(data_fim + timedelta(days=1)))
        .filterBounds(geometria)
    )

    df_compostos = _extrair_serie_temporal(colecao, "NDVI", geometria, ESCALA_NDVI)
    if df_compostos.empty:
        return pd.DataFrame(columns=["data", "ndvi"])

    # Escala MODIS: NDVI real = DN / 10000
    df_compostos["ndvi"] = df_compostos["NDVI"] / FATOR_NDVI
    df_compostos = df_compostos[["data", "ndvi"]]

    # Forward-fill para diário
    idx_diario = pd.date_range(start=data_inicio, end=data_fim, freq="D")
    df_diario = pd.DataFrame({"data": idx_diario})
    df_diario = df_diario.merge(df_compostos, on="data", how="left")
    df_diario["ndvi"] = df_diario["ndvi"].ffill()

    logger.info(f"NDVI: {len(df_compostos)} compostos → {len(df_diario)} dias (forward-fill)")
    return df_diario


def coletar_lst_night_diario(
    geometria: ee.Geometry,
    data_inicio: date,
    data_fim: date,
) -> pd.DataFrame:
    """Coleta LST_Night MODIS MOD11A1 (temperatura noturna de superfície).

    Valores em Kelvin × 50 (formato MODIS). Convertido para °C.
    Dias com nuvem retornam NaN — não interpolamos automaticamente.

    Args:
        geometria: Polígono sobre o qual agregar.
        data_inicio: Primeira data (inclusiva).
        data_fim: Última data (inclusiva).

    Returns:
        DataFrame com colunas 'data' e 'lst_night_c' (°C).
    """
    if data_inicio > data_fim:
        raise ValueError(f"data_inicio ({data_inicio}) > data_fim ({data_fim})")

    colecao = (
        ee.ImageCollection(DATASET_LST)
        .filterDate(str(data_inicio), str(data_fim + timedelta(days=1)))
        .filterBounds(geometria)
    )

    df = _extrair_serie_temporal(colecao, "LST_Night_1km", geometria, ESCALA_LST)
    if df.empty:
        return pd.DataFrame(columns=["data", "lst_night_c"])

    # Escala MODIS LST: K = DN * 0.02; °C = K - 273.15
    df["lst_night_c"] = df["LST_Night_1km"] * FATOR_LST - KELVIN_PARA_CELSIUS
    df = df[["data", "lst_night_c"]]

    logger.info(f"LST_Night: {len(df)} dias coletados")
    return df
