"""Onda 2A - Etapa 3.2 (complementar): Interação NDVI × Temperatura.

Hipótese: o risco de dengue é máximo quando NDVI alto E temperatura mínima
alta coexistem — não separadamente. Testa três coisas:

    1. Feature composta ndvi × temp_min (produto z-score) vs variáveis isoladas
    2. Correlação parcial: NDVI controlando temp_min (e vice-versa)
       → NDVI adiciona informação independente ou é apenas proxy de temperatura?
    3. Tabela comparativa final ordenada por |r|

Lags usados: temp_min lag 6 (melhor Onda 1), NDVI lag 8 (melhor Onda 2A).

Saída: data/processed/analise_ndvi_temperatura_maringa_2024.csv
"""

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger

from observatorio_dengue.etl import database
from observatorio_dengue.features.cruzamento import cruzar_dengue_clima
from observatorio_dengue.features.correlacoes import correlacao_par

# Lags ótimos identificados no smoke_correlacoes_v2.py
LAG_TEMP = 6
LAG_NDVI = 8
COLUNA_DENGUE = "p_inc100k"


def _zscore(s: pd.Series) -> pd.Series:
    """Z-score robusto — retorna NaN onde desvio padrão é zero."""
    std = s.std()
    if std == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / std


def correlacao_parcial(x: pd.Series, y: pd.Series, z: pd.Series) -> dict:
    """Correlação parcial de x com y controlando z (Pearson).

    Remove linhas com NaN em qualquer das três séries antes de calcular.

    Args:
        x: Variável de interesse.
        y: Variável alvo (dengue).
        z: Variável de controle.

    Returns:
        Dict com r_parcial, p_valor, n.
    """
    df = pd.DataFrame({"x": x, "y": y, "z": z}).dropna()
    n = len(df)
    if n < 5:
        return {"r_parcial": np.nan, "p_valor": np.nan, "n": n}

    # Resíduos de x ~ z
    slope_xz, intercept_xz, *_ = stats.linregress(df["z"], df["x"])
    resid_x = df["x"] - (intercept_xz + slope_xz * df["z"])

    # Resíduos de y ~ z
    slope_yz, intercept_yz, *_ = stats.linregress(df["z"], df["y"])
    resid_y = df["y"] - (intercept_yz + slope_yz * df["z"])

    r, p = stats.pearsonr(resid_x, resid_y)
    return {"r_parcial": float(r), "p_valor": float(p), "n": n}


def main() -> None:
    print("=" * 75)
    print("INTERAÇÃO NDVI × TEMPERATURA — Maringá 2024")
    print("=" * 75)

    # ── 1. Carrega fontes ────────────────────────────────────────────────
    df_dengue = database.carregar(
        "SELECT * FROM dengue_raw ORDER BY ano_epi, semana_epi"
    )
    df_clima = database.carregar(
        "SELECT * FROM clima_semanal ORDER BY ano_epi, semana_epi"
    )
    df_satelite = database.carregar(
        "SELECT * FROM satelite_municipio_semanal ORDER BY ano_epi, semana_epi"
    )
    print(f"\nFontes: dengue={len(df_dengue)} sem, clima={len(df_clima)} sem, "
          f"satélite={len(df_satelite)} sem")

    # ── 2. Cruzamentos com lag ótimo de cada variável ────────────────────
    cruzado_temp = cruzar_dengue_clima(df_dengue, df_clima, lag=LAG_TEMP)
    cruzado_ndvi = cruzar_dengue_clima(df_dengue, df_satelite, lag=LAG_NDVI)

    col_temp = f"temperature_2m_min_lag{LAG_TEMP}"
    col_ndvi = f"ndvi_lag{LAG_NDVI}"

    # ── 3. DataFrame unificado (inner join nas semanas com ambas as vars) ─
    df = cruzado_temp[["ano_epi", "semana_epi", COLUNA_DENGUE, col_temp]].merge(
        cruzado_ndvi[["ano_epi", "semana_epi", col_ndvi]],
        on=["ano_epi", "semana_epi"],
        how="inner",
    ).dropna(subset=[col_temp, col_ndvi, COLUNA_DENGUE])

    print(f"Semanas com temp_min lag{LAG_TEMP} + NDVI lag{LAG_NDVI} + dengue: {len(df)}")

    # ── 4. Feature composta: produto dos z-scores ────────────────────────
    df["z_temp"] = _zscore(df[col_temp])
    df["z_ndvi"] = _zscore(df[col_ndvi])
    df["ndvi_x_temp"] = df["z_ndvi"] * df["z_temp"]

    # ── 5. Correlações individuais e composta ────────────────────────────
    dengue = df[COLUNA_DENGUE]
    resultados = []

    for label, serie in [
        (f"temperature_2m_min (lag {LAG_TEMP})", df[col_temp]),
        (f"ndvi (lag {LAG_NDVI})",               df[col_ndvi]),
        (f"ndvi × temp_min (lag {LAG_NDVI}/{LAG_TEMP})", df["ndvi_x_temp"]),
    ]:
        for metodo in ("pearson", "spearman"):
            res = correlacao_par(serie, dengue, metodo=metodo)
            resultados.append({
                "variavel": label,
                "metodo": metodo,
                "r": res["r"],
                "p_valor": res["p_valor"],
                "n": res["n"],
            })

    df_res = pd.DataFrame(resultados)
    df_res["abs_r"] = df_res["r"].abs()

    print("\n" + "=" * 75)
    print("CORRELAÇÕES: isoladas vs composta (ordenado por |r|)")
    print("=" * 75)
    print(
        df_res.sort_values("abs_r", ascending=False)
        [["variavel", "metodo", "r", "p_valor", "n"]]
        .to_string(index=False)
    )

    # ── 6. Correlação parcial ────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("CORRELAÇÃO PARCIAL (Pearson)")
    print("=" * 75)

    # NDVI controlando temp_min
    cp_ndvi = correlacao_parcial(df[col_ndvi], dengue, df[col_temp])
    # temp_min controlando NDVI
    cp_temp = correlacao_parcial(df[col_temp], dengue, df[col_ndvi])

    print(f"  NDVI | controlando temp_min  → r_parcial={cp_ndvi['r_parcial']:.3f}  "
          f"p={cp_ndvi['p_valor']:.4f}  n={cp_ndvi['n']}")
    print(f"  temp_min | controlando NDVI  → r_parcial={cp_temp['r_parcial']:.3f}  "
          f"p={cp_temp['p_valor']:.4f}  n={cp_temp['n']}")

    if abs(cp_ndvi["r_parcial"]) > 0.2 and cp_ndvi["p_valor"] < 0.05:
        print("\n  → NDVI contribui de forma INDEPENDENTE da temperatura ✓")
    else:
        print("\n  → NDVI é em grande parte proxy da temperatura (correlação parcial fraca)")

    if abs(cp_temp["r_parcial"]) > 0.2 and cp_temp["p_valor"] < 0.05:
        print("  → temp_min contribui de forma INDEPENDENTE do NDVI ✓")
    else:
        print("  → temp_min é em grande parte proxy do NDVI (correlação parcial fraca)")

    # ── 7. Salva CSV ─────────────────────────────────────────────────────
    out_path = "data/processed/analise_ndvi_temperatura_maringa_2024.csv"
    df_res.drop(columns=["abs_r"]).to_csv(out_path, index=False)
    print(f"\n💾 Resultados salvos em {out_path}")

    # ── 8. Dados cruzados (úteis para modelagem futura) ──────────────────
    out_dados = "data/processed/features_ndvi_temperatura_maringa_2024.csv"
    df[["ano_epi", "semana_epi", COLUNA_DENGUE,
        col_temp, col_ndvi, "ndvi_x_temp"]].to_csv(out_dados, index=False)
    print(f"💾 Features cruzadas salvas em {out_dados}")
    print(f"   ({len(df)} semanas × 4 features + dengue)")


if __name__ == "__main__":
    main()
