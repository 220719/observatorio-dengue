"""Onda 3 - Etapa 2: Modelagem preditiva (RF com features autorregressivas).

Features:
  - temperature_2m_min_lag8, ndvi_lag8 (clima/satélite)
  - casos_lag1..casos_lag4 (autorregressivas — casos das últimas 4 semanas)

Target: p_inc100k
Validação: walk-forward temporal
Modelos: baseline (média móvel 4 sem) vs RF clima-only vs RF clima+autoregr

Saída:
  - data/processed/resultados_modelagem_2020_2025.csv
  - data/processed/previsoes_walkforward_2020_2025.csv
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from observatorio_dengue.etl import database
from observatorio_dengue.features.cruzamento import cruzar_dengue_clima


# ── Parâmetros ───────────────────────────────────────────────────────────
LAG = 8
COLUNA_DENGUE = "p_inc100k"
JANELA_TREINO_MIN = 52
JANELA_MEDIA_MOVEL = 4
LAGS_AUTOREGR = [1, 2, 3, 4]  # semanas anteriores de casos


def preparar_dados() -> pd.DataFrame:
    """Carrega, cruza e adiciona features autorregressivas."""
    df_dengue = database.carregar(
        "SELECT * FROM dengue_raw ORDER BY ano_epi, semana_epi"
    )
    df_clima = database.carregar(
        "SELECT * FROM clima_semanal ORDER BY ano_epi, semana_epi"
    )
    df_satelite = database.carregar(
        "SELECT * FROM satelite_municipio_semanal ORDER BY ano_epi, semana_epi"
    )

    cruzado_clima = cruzar_dengue_clima(df_dengue, df_clima, lag=LAG)
    cruzado_sat = cruzar_dengue_clima(df_dengue, df_satelite, lag=LAG)

    col_temp = f"temperature_2m_min_lag{LAG}"
    col_ndvi = f"ndvi_lag{LAG}"

    df = cruzado_clima[["ano_epi", "semana_epi", COLUNA_DENGUE, col_temp]].merge(
        cruzado_sat[["ano_epi", "semana_epi", col_ndvi]],
        on=["ano_epi", "semana_epi"],
        how="inner",
    )

    # Ordena cronologicamente
    df = df.sort_values(["ano_epi", "semana_epi"]).reset_index(drop=True)

    # Features autorregressivas: casos das últimas N semanas
    for lag_ar in LAGS_AUTOREGR:
        df[f"casos_lag{lag_ar}"] = df[COLUNA_DENGUE].shift(lag_ar)

    # Remove linhas com NaN (primeiras semanas sem lag completo)
    df = df.dropna().reset_index(drop=True)

    return df


def walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward: 3 modelos comparados a cada passo."""
    col_temp = f"temperature_2m_min_lag{LAG}"
    col_ndvi = f"ndvi_lag{LAG}"
    cols_clima = [col_temp, col_ndvi]
    cols_autoregr = [f"casos_lag{i}" for i in LAGS_AUTOREGR]
    cols_completo = cols_clima + cols_autoregr

    rf_params = dict(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )

    resultados = []

    for i in range(JANELA_TREINO_MIN, len(df)):
        treino = df.iloc[:i]
        teste = df.iloc[i:i+1]
        y_real = teste[COLUNA_DENGUE].values[0]

        # 1. Baseline: média móvel
        prev_baseline = float(treino[COLUNA_DENGUE].tail(JANELA_MEDIA_MOVEL).mean())

        # 2. RF clima only
        rf_clima = RandomForestRegressor(**rf_params)
        rf_clima.fit(treino[cols_clima], treino[COLUNA_DENGUE])
        prev_rf_clima = float(rf_clima.predict(teste[cols_clima])[0])

        # 3. RF completo (clima + autorregressivo)
        rf_completo = RandomForestRegressor(**rf_params)
        rf_completo.fit(treino[cols_completo], treino[COLUNA_DENGUE])
        prev_rf_completo = float(rf_completo.predict(teste[cols_completo])[0])

        resultados.append({
            "ano_epi": int(teste["ano_epi"].values[0]),
            "semana_epi": int(teste["semana_epi"].values[0]),
            "real": y_real,
            "prev_baseline": max(0, prev_baseline),
            "prev_rf_clima": max(0, prev_rf_clima),
            "prev_rf_completo": max(0, prev_rf_completo),
        })

        if (i - JANELA_TREINO_MIN) % 50 == 0:
            print(f"      ... {i - JANELA_TREINO_MIN}/{len(df) - JANELA_TREINO_MIN} previsões")

    return pd.DataFrame(resultados)


def calcular_metricas(y_real: np.ndarray, y_pred: np.ndarray, nome: str) -> dict:
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2 = r2_score(y_real, y_pred)
    mask = y_real > 0
    mape = np.mean(np.abs((y_real[mask] - y_pred[mask]) / y_real[mask])) * 100 if mask.sum() > 0 else np.nan
    return {"modelo": nome, "MAE": mae, "RMSE": rmse, "R²": r2, "MAPE_%": mape}


def main() -> None:
    print("=" * 75)
    print("MODELAGEM PREDITIVA v2 — Onda 3, Maringá 2020-2025")
    print("=" * 75)

    # ── 1. Prepara dados ─────────────────────────────────────────────────
    print("\n[1/4] Preparando dados (clima + satélite + autorregressivas)...")
    df = preparar_dados()
    col_temp = f"temperature_2m_min_lag{LAG}"
    col_ndvi = f"ndvi_lag{LAG}"
    cols_autoregr = [f"casos_lag{i}" for i in LAGS_AUTOREGR]
    print(f"      {len(df)} semanas disponíveis")
    print(f"      Features clima: {col_temp}, {col_ndvi}")
    print(f"      Features autoregr: {', '.join(cols_autoregr)}")

    # ── 2. Walk-forward ──────────────────────────────────────────────────
    n_prev = len(df) - JANELA_TREINO_MIN
    print(f"\n[2/4] Walk-forward validation ({n_prev} previsões, 3 modelos)...")
    previsoes = walk_forward(df)
    print(f"      ✓ {len(previsoes)} previsões geradas")

    # ── 3. Métricas globais ──────────────────────────────────────────────
    y_real = previsoes["real"].values
    metricas = pd.DataFrame([
        calcular_metricas(y_real, previsoes["prev_baseline"].values, "Média Móvel 4sem"),
        calcular_metricas(y_real, previsoes["prev_rf_clima"].values, "RF clima-only"),
        calcular_metricas(y_real, previsoes["prev_rf_completo"].values, "RF clima+autoregr"),
    ])

    print("\n" + "=" * 75)
    print("RESULTADOS — Walk-Forward Validation")
    print("=" * 75)
    print(metricas.to_string(index=False))

    # ── Comparações ──────────────────────────────────────────────────────
    print("\n" + "-" * 75)
    bl_mae = metricas.iloc[0]["MAE"]
    for i, nome in enumerate(["RF clima-only", "RF clima+autoregr"]):
        rf_mae = metricas.iloc[i+1]["MAE"]
        ganho = (1 - rf_mae / bl_mae) * 100
        print(f"  {nome} vs Baseline:  MAE {ganho:+.1f}%  "
              f"{'✓ melhor' if ganho > 0 else '✗ pior'}")

    # ── Feature importance (modelo completo) ─────────────────────────────
    print("\n" + "=" * 75)
    print("IMPORTÂNCIA DAS FEATURES (modelo completo, último treino)")
    print("=" * 75)

    cols_completo = [col_temp, col_ndvi] + cols_autoregr
    rf_final = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )
    rf_final.fit(df[cols_completo], df[COLUNA_DENGUE])
    importancias = sorted(
        zip(cols_completo, rf_final.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    for feat, imp in importancias:
        print(f"  {feat:35s}  {imp:.3f}  {'█' * int(imp * 50)}")

    # ── Performance por ano ──────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("PERFORMANCE POR ANO")
    print("=" * 75)
    print(f"  {'ano':>4s}  {'n':>3s}  {'MAE_BL':>7s}  {'MAE_RF':>7s}  "
          f"{'R²_RF':>6s}  {'vencedor':>10s}")

    for ano in sorted(previsoes["ano_epi"].unique()):
        sub = previsoes[previsoes["ano_epi"] == ano]
        if len(sub) < 10:
            continue
        mae_bl = mean_absolute_error(sub["real"], sub["prev_baseline"])
        mae_rf = mean_absolute_error(sub["real"], sub["prev_rf_completo"])
        r2_rf = r2_score(sub["real"], sub["prev_rf_completo"]) if sub["real"].std() > 0 else np.nan
        vencedor = "RF" if mae_rf < mae_bl else "Baseline"
        print(f"  {ano:>4d}  {len(sub):>3d}  {mae_bl:>7.1f}  {mae_rf:>7.1f}  "
              f"{r2_rf:>6.3f}  {vencedor:>10s}")

    # ── 4. Salva ─────────────────────────────────────────────────────────
    out_metricas = "data/processed/resultados_modelagem_2020_2025.csv"
    metricas.to_csv(out_metricas, index=False)

    out_prev = "data/processed/previsoes_walkforward_2020_2025.csv"
    previsoes.to_csv(out_prev, index=False)

    print(f"\n💾 Métricas: {out_metricas}")
    print(f"💾 Previsões: {out_prev}")
    print(f"   ({len(previsoes)} previsões walk-forward)")
    print("\n✅ Modelagem v2 concluída!")


if __name__ == "__main__":
    main()