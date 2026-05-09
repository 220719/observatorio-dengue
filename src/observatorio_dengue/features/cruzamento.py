"""Cruzamento dengue × clima com defasagem temporal (lag).

Premissa epidemiológica: o ciclo Aedes aegypti + período de incubação
faz com que o clima de N semanas atrás explique melhor os casos atuais
do que o clima da semana atual. Lag típico: 3-5 semanas (default: 4).

Funcionamento:
    Clima da semana N  →  Casos da semana N + lag

Esta camada usa epiweeks para somar lag corretamente, respeitando anos
ISO com 53 semanas (bug do código antigo corrigido).
"""

import pandas as pd
from epiweeks import Week
from loguru import logger


def aplicar_lag_epiweek(ano: int, semana: int, lag: int, system: str = "iso") -> tuple[int, int]:
    """Adiciona N semanas a uma (ano, semana) epidemiológica.

    Usa epiweeks para que a virada de ano respeite anos ISO com 53 semanas.

    Args:
        ano: Ano epidemiológico de origem.
        semana: Semana epidemiológica de origem.
        lag: Quantas semanas adicionar (positivo) ou subtrair (negativo).
        system: Sistema de semana epidemiológica ('iso' ou 'cdc'). Default: 'iso'.

    Returns:
        Tupla (ano_target, semana_target) após aplicar o lag.

    Example:
        >>> aplicar_lag_epiweek(2020, 50, 4)
        (2021, 1)  # 2020 tem 53 semanas, então 50+4 → semana 1 de 2021
    """
    week = Week(ano, semana, system=system)
    week_lag = week + lag
    return week_lag.year, week_lag.week


def adicionar_lag(df_clima: pd.DataFrame, lag: int = 4) -> pd.DataFrame:
    """Adiciona colunas de ano/semana alvo após aplicar lag temporal.

    Para cada linha de clima semanal, calcula a (ano, semana) que essa
    observação climática "explica" — isto é, ano/semana + lag.

    Args:
        df_clima: DataFrame com colunas 'ano_epi' e 'semana_epi'
            (geralmente de openmeteo.agregar_para_semana_epidemiologica).
        lag: Quantas semanas à frente o clima é projetado. Default: 4.

    Returns:
        DataFrame original com colunas adicionais 'ano_target' e 'semana_target'.

    Raises:
        ValueError: Se faltar coluna ano_epi ou semana_epi.
    """
    if df_clima.empty:
        logger.warning("DataFrame de clima vazio")
        return df_clima.copy()

    colunas_obrigatorias = {"ano_epi", "semana_epi"}
    faltando = colunas_obrigatorias - set(df_clima.columns)
    if faltando:
        raise ValueError(f"Colunas obrigatórias faltando: {faltando}")

    df = df_clima.copy()
    targets = df.apply(
        lambda row: aplicar_lag_epiweek(int(row["ano_epi"]), int(row["semana_epi"]), lag),
        axis=1,
    )
    df["ano_target"] = targets.apply(lambda x: x[0])
    df["semana_target"] = targets.apply(lambda x: x[1])

    logger.info(f"Lag de {lag} semanas aplicado a {len(df)} linhas")
    return df


def cruzar_dengue_clima(
    df_dengue: pd.DataFrame,
    df_clima_semanal: pd.DataFrame,
    lag: int = 4,
) -> pd.DataFrame:
    """Junta dengue e clima com defasagem temporal.

    O DataFrame retornado tem os casos de cada semana junto com o clima
    de N semanas atrás (lag). Útil para análise de correlação e modelagem.

    Args:
        df_dengue: DataFrame com colunas 'ano_epi', 'semana_epi', 'casos',
            'casos_est', 'p_inc100k' (de etl.infodengue ou de carregar()).
        df_clima_semanal: DataFrame com colunas 'ano_epi', 'semana_epi'
            e variáveis climáticas (de openmeteo.agregar_para_semana_epidemiologica).
        lag: Defasagem em semanas (clima precede casos). Default: 4.

    Returns:
        DataFrame com uma linha por semana, contendo:
        - ano_epi, semana_epi (da dengue, semana dos casos)
        - casos, casos_est, p_inc100k
        - {variavel}_lag{N}: variáveis climáticas defasadas
        - dias_validos_lag{N}: cobertura de dias na semana climática

    Raises:
        ValueError: Se faltarem colunas em algum dos DataFrames.
    """
    if df_dengue.empty or df_clima_semanal.empty:
        logger.warning("DataFrame de dengue ou clima vazio, retornando vazio")
        return pd.DataFrame()

    obrig_dengue = {"ano_epi", "semana_epi"}
    if not obrig_dengue.issubset(df_dengue.columns):
        raise ValueError(f"df_dengue precisa das colunas {obrig_dengue}")

    obrig_clima = {"ano_epi", "semana_epi"}
    if not obrig_clima.issubset(df_clima_semanal.columns):
        raise ValueError(f"df_clima_semanal precisa das colunas {obrig_clima}")

    # 1. Aplica lag ao clima (calcula a semana "alvo" de cada observação)
    clima_com_lag = adicionar_lag(df_clima_semanal, lag=lag)

    # 2. Identifica colunas climáticas (todas exceto chaves e auxiliares)
    colunas_chave_clima = {"ano_epi", "semana_epi", "ano_target", "semana_target"}
    colunas_climaticas = [c for c in clima_com_lag.columns if c not in colunas_chave_clima]

    # 3. Renomeia colunas climáticas com sufixo _lag{N} para deixar explícito
    sufixo = f"_lag{lag}"
    clima_renomeado = clima_com_lag.rename(columns={c: f"{c}{sufixo}" for c in colunas_climaticas})

    # 4. Mantém apenas as chaves de target + colunas climáticas renomeadas
    clima_para_merge = clima_renomeado[
        ["ano_target", "semana_target"] + [f"{c}{sufixo}" for c in colunas_climaticas]
    ]

    # 5. Merge: ano/semana de target do clima = ano/semana epi do dengue
    resultado = df_dengue.merge(
        clima_para_merge,
        left_on=["ano_epi", "semana_epi"],
        right_on=["ano_target", "semana_target"],
        how="left",
    ).drop(columns=["ano_target", "semana_target"])

    casos_total = len(resultado)
    casos_com_clima = resultado[f"temperature_2m_mean{sufixo}"].notna().sum()
    logger.info(
        f"Cruzamento (lag={lag}): {casos_total} semanas de dengue, "
        f"{casos_com_clima} com clima correspondente"
    )

    return resultado
