"""Onda 2A - Etapa 3.2: Matriz de correlações com clima + satélite.

Carrega dengue, clima e satélite do DuckDB e roda a varredura completa
de lags (0-8) para todas as variáveis. Foco analítico: comparar a
performance preditiva do LST_Night (satélite) com a do temperature_2m_min
(ERA5, melhor variável da Onda 1).

Resultados salvos em data/processed/correlacoes_clima_satelite_maringa_2024.csv.
"""
import pandas as pd

from observatorio_dengue.etl import database
from observatorio_dengue.features.correlacoes import matriz_correlacoes

from observatorio_dengue.etl import database
from observatorio_dengue.features.correlacoes import matriz_correlacoes


def main() -> None:
    print("=" * 75)
    print("MATRIZ DE CORRELAÇÕES — Clima + Satélite × Dengue, Maringá 2024")
    print("=" * 75)

    # 1) Carrega as 3 fontes do DuckDB
    df_dengue = database.carregar(
        "SELECT * FROM dengue_raw ORDER BY ano_epi, semana_epi"
    )
    df_clima = database.carregar(
        "SELECT * FROM clima_semanal ORDER BY ano_epi, semana_epi"
    )
    df_satelite = database.carregar(
        "SELECT * FROM satelite_municipio_semanal ORDER BY ano_epi, semana_epi"
    )
    print(f"\nFontes carregadas:")
    print(f"  dengue:   {len(df_dengue)} semanas")
    print(f"  clima:    {len(df_clima)} semanas")
    print(f"  satélite: {len(df_satelite)} semanas")

    # 2) Matriz de correlações — clima
    vars_clima = [
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "relative_humidity_2m_mean",
    ]
    print(f"\n[1/2] Calculando correlações para {len(vars_clima)} vars clima...")
    matriz_clima = matriz_correlacoes(
        df_dengue=df_dengue,
        df_clima_semanal=df_clima,
        vars_clima=vars_clima,
        lags=range(0, 9),
        coluna_dengue="p_inc100k",
        metodos=("pearson", "spearman"),
    )

    # 3) Matriz de correlações — satélite (reusa a mesma função!)
    vars_satelite = ["ndvi", "lst_night_c"]
    print(f"[2/2] Calculando correlações para {len(vars_satelite)} vars satélite...")
    matriz_sat = matriz_correlacoes(
        df_dengue=df_dengue,
        df_clima_semanal=df_satelite,  # nome confuso mas a função é genérica
        vars_clima=vars_satelite,
        lags=range(0, 9),
        coluna_dengue="p_inc100k",
        metodos=("pearson", "spearman"),
    )

    # 4) Combina ambas
    matriz_sat["fonte"] = "satélite"
    matriz_clima["fonte"] = "clima"
    matriz = pd.concat([matriz_clima, matriz_sat], ignore_index=True)
    matriz["abs_r"] = matriz["r"].abs()

    # 5) Lag ótimo por variável e método
    print("\n" + "=" * 75)
    print("LAG ÓTIMO POR VARIÁVEL (ordenado por |r|)")
    print("=" * 75)
    otimos = (
        matriz.sort_values("abs_r", ascending=False)
        .groupby(["var_clima", "metodo"], as_index=False)
        .first()
        .sort_values("abs_r", ascending=False)
        [["fonte", "var_clima", "metodo", "lag", "r", "p_valor", "n"]]
    )
    print(otimos.to_string(index=False))

    # 6) Comparação direta: LST_Night satélite vs temperature_2m_min ERA5
    print("\n" + "=" * 75)
    print("FOCO: LST_Night (satélite) vs temperature_2m_min (ERA5)")
    print("=" * 75)
    foco = matriz[
        matriz["var_clima"].isin(["lst_night_c", "temperature_2m_min"])
    ].copy()
    pivot = foco.pivot_table(
        index="lag",
        columns=["fonte", "var_clima", "metodo"],
        values="r",
    ).round(3)
    print(pivot.to_string())

    # 7) Salva CSV
    out_path = "data/processed/correlacoes_clima_satelite_maringa_2024.csv"
    matriz.drop(columns=["abs_r"]).to_csv(out_path, index=False)
    print(f"\n💾 Matriz completa salva em {out_path}")
    print(f"   ({len(matriz)} testes estatísticos)")

if __name__ == "__main__":
    main()