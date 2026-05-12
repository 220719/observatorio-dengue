"""Onda 3 - Etapa 1.5: Validação das features com dados expandidos (2020-2025).

Roda com 314 semanas (vs 52 na Onda 2A) para confirmar quais features
são realmente robustas antes da modelagem. Testa:

    1. Correlações isoladas: todas as variáveis no lag ótimo (8)
    2. Correlação parcial: cada par controlando a terceira
    3. Multicolinearidade (VIF): features redundantes?
    4. Estacionariedade: a relação é estável ao longo dos anos?
    5. Feature composta ndvi × temp (revisitar com mais dados)

Saída: data/processed/validacao_features_2020_2025.csv
"""

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger

from observatorio_dengue.etl import database
from observatorio_dengue.features.cruzamento import cruzar_dengue_clima
from observatorio_dengue.features.correlacoes import correlacao_par


# ── Parâmetros ───────────────────────────────────────────────────────────
LAG = 8  # lag ótimo confirmado pelo smoke expandido
COLUNA_DENGUE = "p_inc100k"

# Top 3 variáveis do smoke expandido
VARS_CLIMA = {"temperature_2m_min": "clima", "temperature_2m_mean": "clima"}
VARS_SAT = {"lst_night_c": "satélite", "ndvi": "satélite"}


def _zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    if std == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / std


def correlacao_parcial(x: pd.Series, y: pd.Series, z: pd.Series) -> dict:
    """Correlação parcial de x com y controlando z (Pearson)."""
    df = pd.DataFrame({"x": x, "y": y, "z": z}).dropna()
    n = len(df)
    if n < 5:
        return {"r_parcial": np.nan, "p_valor": np.nan, "n": n}
    slope_xz, intercept_xz, *_ = stats.linregress(df["z"], df["x"])
    resid_x = df["x"] - (intercept_xz + slope_xz * df["z"])
    slope_yz, intercept_yz, *_ = stats.linregress(df["z"], df["y"])
    resid_y = df["y"] - (intercept_yz + slope_yz * df["z"])
    r, p = stats.pearsonr(resid_x, resid_y)
    return {"r_parcial": float(r), "p_valor": float(p), "n": n}


def calcular_vif(df: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    """Variance Inflation Factor para detectar multicolinearidade."""
    from numpy.linalg import LinAlgError
    resultados = []
    df_limpo = df[colunas].dropna()
    for i, col in enumerate(colunas):
        y = df_limpo[col]
        X = df_limpo[[c for c in colunas if c != col]]
        if len(X.columns) == 0:
            resultados.append({"variavel": col, "VIF": 1.0})
            continue
        try:
            X_with_const = np.column_stack([np.ones(len(X)), X.values])
            beta = np.linalg.lstsq(X_with_const, y.values, rcond=None)[0]
            y_pred = X_with_const @ beta
            ss_res = ((y - y_pred) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            vif = 1 / (1 - r2) if r2 < 1 else float("inf")
        except LinAlgError:
            vif = float("inf")
        resultados.append({"variavel": col, "VIF": round(vif, 2)})
    return pd.DataFrame(resultados)


def main() -> None:
    print("=" * 75)
    print("VALIDAÇÃO DE FEATURES — Dados Expandidos 2020-2025")
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
    print(f"\nFontes: dengue={len(df_dengue)}, clima={len(df_clima)}, "
          f"satélite={len(df_satelite)} semanas")

    # ── 2. Cruzamentos com lag 8 ────────────────────────────────────────
    cruzado_clima = cruzar_dengue_clima(df_dengue, df_clima, lag=LAG)
    cruzado_sat = cruzar_dengue_clima(df_dengue, df_satelite, lag=LAG)

    col_temp_min = f"temperature_2m_min_lag{LAG}"
    col_temp_mean = f"temperature_2m_mean_lag{LAG}"
    col_lst = f"lst_night_c_lag{LAG}"
    col_ndvi = f"ndvi_lag{LAG}"

    # DataFrame unificado
    df = cruzado_clima[["ano_epi", "semana_epi", COLUNA_DENGUE,
                         col_temp_min, col_temp_mean]].merge(
        cruzado_sat[["ano_epi", "semana_epi", col_lst, col_ndvi]],
        on=["ano_epi", "semana_epi"],
        how="inner",
    ).dropna(subset=[col_temp_min, col_lst, col_ndvi, COLUNA_DENGUE])

    print(f"Semanas com todas as features + dengue: {len(df)}")

    # ── 3. Correlações isoladas ─────────────────────────────────────────
    dengue = df[COLUNA_DENGUE]
    print("\n" + "=" * 75)
    print(f"CORRELAÇÕES ISOLADAS (lag {LAG}, ordenado por |r|)")
    print("=" * 75)

    resultados = []
    for label, serie in [
        ("temperature_2m_min", df[col_temp_min]),
        ("temperature_2m_mean", df[col_temp_mean]),
        ("lst_night_c", df[col_lst]),
        ("ndvi", df[col_ndvi]),
    ]:
        for metodo in ("pearson", "spearman"):
            res = correlacao_par(serie, dengue, metodo=metodo)
            resultados.append({
                "variavel": label, "metodo": metodo,
                "r": res["r"], "p_valor": res["p_valor"], "n": res["n"],
            })

    df_corr = pd.DataFrame(resultados)
    df_corr["abs_r"] = df_corr["r"].abs()
    print(
        df_corr.sort_values("abs_r", ascending=False)
        [["variavel", "metodo", "r", "p_valor", "n"]]
        .to_string(index=False)
    )

    # ── 4. Correlação parcial (todos os pares) ──────────────────────────
    print("\n" + "=" * 75)
    print("CORRELAÇÃO PARCIAL (Pearson) — cada variável controlando as outras")
    print("=" * 75)

    pares_parcial = [
        ("lst_night_c", col_lst, "temp_min", col_temp_min),
        ("temp_min", col_temp_min, "lst_night_c", col_lst),
        ("ndvi", col_ndvi, "temp_min", col_temp_min),
        ("temp_min", col_temp_min, "ndvi", col_ndvi),
        ("ndvi", col_ndvi, "lst_night_c", col_lst),
        ("lst_night_c", col_lst, "ndvi", col_ndvi),
    ]

    resultados_parcial = []
    for var_nome, var_col, ctrl_nome, ctrl_col in pares_parcial:
        cp = correlacao_parcial(df[var_col], dengue, df[ctrl_col])
        sig = "✓" if abs(cp["r_parcial"]) > 0.15 and cp["p_valor"] < 0.05 else "✗"
        print(f"  {var_nome:20s} | controlando {ctrl_nome:15s} → "
              f"r_parcial={cp['r_parcial']:.3f}  p={cp['p_valor']:.4f}  "
              f"n={cp['n']}  {sig}")
        resultados_parcial.append({
            "variavel": var_nome, "controlando": ctrl_nome,
            "r_parcial": cp["r_parcial"], "p_valor": cp["p_valor"], "n": cp["n"],
        })

    # ── 5. Multicolinearidade (VIF) ─────────────────────────────────────
    print("\n" + "=" * 75)
    print("MULTICOLINEARIDADE (VIF) — regra: VIF > 10 = problema")
    print("=" * 75)

    colunas_features = [col_temp_min, col_lst, col_ndvi]
    vif = calcular_vif(df, colunas_features)
    for _, row in vif.iterrows():
        status = "⚠️ ALTO" if row["VIF"] > 10 else "✓ OK" if row["VIF"] < 5 else "~ moderado"
        print(f"  {row['variavel']:35s}  VIF={row['VIF']:.1f}  {status}")

    # ── 6. Estabilidade temporal (correlação por ano) ───────────────────
    print("\n" + "=" * 75)
    print("ESTABILIDADE TEMPORAL — correlação Spearman por ano")
    print("=" * 75)
    print(f"  {'ano':>4s}  {'n':>3s}  {'temp_min':>9s}  {'lst_night':>10s}  {'ndvi':>8s}")

    for ano in sorted(df["ano_epi"].unique()):
        mask = df["ano_epi"] == ano
        sub = df[mask]
        if len(sub) < 10:
            continue
        r_temp = correlacao_par(sub[col_temp_min], sub[COLUNA_DENGUE], "spearman")["r"]
        r_lst = correlacao_par(sub[col_lst], sub[COLUNA_DENGUE], "spearman")["r"]
        r_ndvi = correlacao_par(sub[col_ndvi], sub[COLUNA_DENGUE], "spearman")["r"]
        print(f"  {ano:>4d}  {len(sub):>3d}  {r_temp:>9.3f}  {r_lst:>10.3f}  {r_ndvi:>8.3f}")

    # ── 7. Feature composta revisitada ──────────────────────────────────
    print("\n" + "=" * 75)
    print("FEATURE COMPOSTA (revisitar com 314 semanas)")
    print("=" * 75)

    df["z_temp"] = _zscore(df[col_temp_min])
    df["z_lst"] = _zscore(df[col_lst])
    df["z_ndvi"] = _zscore(df[col_ndvi])
    df["lst_x_temp"] = df["z_lst"] * df["z_temp"]
    df["ndvi_x_temp"] = df["z_ndvi"] * df["z_temp"]

    for label, serie in [
        ("lst × temp_min", df["lst_x_temp"]),
        ("ndvi × temp_min", df["ndvi_x_temp"]),
    ]:
        for metodo in ("pearson", "spearman"):
            res = correlacao_par(serie, dengue, metodo=metodo)
            sig = "✓" if res["p_valor"] < 0.05 else "n.s."
            print(f"  {label:20s} {metodo:>8s}  r={res['r']:.3f}  "
                  f"p={res['p_valor']:.4f}  {sig}")

    # ── 8. Recomendação final ────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("RECOMENDAÇÃO PARA MODELAGEM (Onda 3)")
    print("=" * 75)

    # Identifica features independentes significativas
    features_recomendadas = []
    for row in resultados_parcial:
        if abs(row["r_parcial"]) > 0.15 and row["p_valor"] < 0.05:
            features_recomendadas.append(row["variavel"])
    features_recomendadas = sorted(set(features_recomendadas))

    print(f"  Features com contribuição independente confirmada:")
    for f in features_recomendadas:
        print(f"    → {f}")
    print(f"\n  Lag unificado: {LAG} semanas")
    print(f"  Semanas disponíveis: {len(df)}")
    print(f"  Período: {df['ano_epi'].min()}-{df['ano_epi'].max()}")

    # ── 9. Salva resultados ──────────────────────────────────────────────
    out_corr = "data/processed/validacao_features_2020_2025.csv"
    df_corr.drop(columns=["abs_r"]).to_csv(out_corr, index=False)

    out_parcial = "data/processed/correlacao_parcial_2020_2025.csv"
    pd.DataFrame(resultados_parcial).to_csv(out_parcial, index=False)

    out_features = "data/processed/features_modelagem_2020_2025.csv"
    df[["ano_epi", "semana_epi", COLUNA_DENGUE,
        col_temp_min, col_lst, col_ndvi]].to_csv(out_features, index=False)

    print(f"\n💾 Correlações: {out_corr}")
    print(f"💾 Parciais: {out_parcial}")
    print(f"💾 Features prontas pra modelagem: {out_features}")
    print(f"   ({len(df)} semanas × 3 features + dengue)")


if __name__ == "__main__":
    main()