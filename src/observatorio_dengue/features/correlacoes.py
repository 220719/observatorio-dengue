"""Correlações entre variáveis climáticas defasadas e indicadores de dengue.

Calcula Pearson e Spearman para múltiplas defasagens temporais (lags),
permitindo identificar qual lag maximiza a correlação clima → dengue.

Pearson mede associação linear; Spearman mede associação monotônica
(robusto a não-linearidades). Quando |Spearman| > |Pearson|, há indício
de relação não-linear monotônica entre as variáveis.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from observatorio_dengue.features.cruzamento import cruzar_dengue_clima

METODOS_VALIDOS = ("pearson", "spearman")
N_MINIMO = 3


def correlacao_par(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    metodo: str = "pearson",
) -> dict:
    """Calcula correlação entre duas séries, removendo pares com NaN.

    Args:
        x: Primeira série.
        y: Segunda série (mesmo tamanho que x).
        metodo: 'pearson' ou 'spearman'.

    Returns:
        Dict com chaves: r, p_valor, n, metodo.
        r e p_valor são NaN quando n < 3 ou quando alguma série é constante.

    Raises:
        ValueError: Se método inválido ou tamanhos diferentes.
    """
    if metodo not in METODOS_VALIDOS:
        raise ValueError(f"método '{metodo}' inválido. Use um de: {METODOS_VALIDOS}")

    x = pd.Series(x).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    if len(x) != len(y):
        raise ValueError(f"séries com tamanhos diferentes: len(x)={len(x)}, len(y)={len(y)}")

    # Remove pares com NaN em qualquer uma das séries
    mask = x.notna() & y.notna()
    x_limpo = x[mask]
    y_limpo = y[mask]
    n = int(mask.sum())

    resultado = {"r": np.nan, "p_valor": np.nan, "n": n, "metodo": metodo}

    if n < N_MINIMO:
        logger.debug(f"n={n} < {N_MINIMO}, retornando NaN")
        return resultado

    # Série constante → correlação indefinida
    if x_limpo.nunique() < 2 or y_limpo.nunique() < 2:
        logger.debug("série constante detectada, retornando NaN")
        return resultado

    if metodo == "pearson":
        r, p = stats.pearsonr(x_limpo, y_limpo)
    else:  # spearman
        r, p = stats.spearmanr(x_limpo, y_limpo)

    resultado["r"] = float(r)
    resultado["p_valor"] = float(p)
    return resultado


def correlacoes_lag(
    df_dengue: pd.DataFrame,
    df_clima_semanal: pd.DataFrame,
    var_clima: str,
    lags: Iterable[int] = range(0, 9),
    coluna_dengue: str = "p_inc100k",
    metodos: Iterable[str] = ("pearson", "spearman"),
) -> pd.DataFrame:
    """Varre lags e calcula correlações entre uma variável climática e dengue.

    Para cada lag, chama cruzar_dengue_clima, extrai a coluna
    {var_clima}_lag{lag} e correlaciona com coluna_dengue.

    Args:
        df_dengue: DataFrame de dengue (saída de etl.infodengue).
        df_clima_semanal: DataFrame de clima agregado por semana epi.
        var_clima: Nome-base da variável climática (ex: 'temperature_2m_mean').
        lags: Lags a testar, em semanas.
        coluna_dengue: Coluna alvo (default: 'p_inc100k', incidência por 100k).
        metodos: Métodos de correlação ('pearson', 'spearman' ou ambos).

    Returns:
        DataFrame longo com colunas: var_clima, metodo, lag, r, p_valor, n.
    """
    metodos = tuple(metodos)
    for m in metodos:
        if m not in METODOS_VALIDOS:
            raise ValueError(f"método '{m}' inválido. Use um de: {METODOS_VALIDOS}")

    registros = []
    for lag in lags:
        df_cruzado = cruzar_dengue_clima(df_dengue, df_clima_semanal, lag=lag)
        coluna_clima_lag = f"{var_clima}_lag{lag}"

        if df_cruzado.empty:
            logger.warning(f"cruzamento vazio para lag={lag}")
            continue
        if coluna_clima_lag not in df_cruzado.columns:
            raise KeyError(
                f"coluna '{coluna_clima_lag}' não encontrada no cruzamento. "
                f"Colunas disponíveis: {list(df_cruzado.columns)}"
            )
        if coluna_dengue not in df_cruzado.columns:
            raise KeyError(
                f"coluna alvo '{coluna_dengue}' não encontrada. "
                f"Colunas disponíveis: {list(df_cruzado.columns)}"
            )

        for metodo in metodos:
            res = correlacao_par(
                df_cruzado[coluna_clima_lag],
                df_cruzado[coluna_dengue],
                metodo=metodo,
            )
            registros.append(
                {
                    "var_clima": var_clima,
                    "metodo": metodo,
                    "lag": lag,
                    "r": res["r"],
                    "p_valor": res["p_valor"],
                    "n": res["n"],
                }
            )

    return pd.DataFrame(registros)


def matriz_correlacoes(
    df_dengue: pd.DataFrame,
    df_clima_semanal: pd.DataFrame,
    vars_clima: Iterable[str],
    lags: Iterable[int] = range(0, 9),
    coluna_dengue: str = "p_inc100k",
    metodos: Iterable[str] = ("pearson", "spearman"),
) -> pd.DataFrame:
    """Aplica correlacoes_lag para múltiplas variáveis climáticas.

    Args:
        df_dengue: DataFrame de dengue.
        df_clima_semanal: DataFrame de clima semanal.
        vars_clima: Lista de variáveis climáticas (nomes-base, sem '_lagN').
        lags: Lags a testar.
        coluna_dengue: Coluna alvo de dengue.
        metodos: Métodos de correlação.

    Returns:
        DataFrame longo concatenando os resultados de cada variável.
    """
    blocos = []
    for var in vars_clima:
        bloco = correlacoes_lag(
            df_dengue,
            df_clima_semanal,
            var_clima=var,
            lags=lags,
            coluna_dengue=coluna_dengue,
            metodos=metodos,
        )
        blocos.append(bloco)

    if not blocos:
        return pd.DataFrame(columns=["var_clima", "metodo", "lag", "r", "p_valor", "n"])

    resultado = pd.concat(blocos, ignore_index=True)
    logger.info(
        f"matriz de correlações: {len(vars_clima)} variáveis × "
        f"{len(list(lags))} lags × {len(list(metodos))} métodos"
    )
    return resultado
