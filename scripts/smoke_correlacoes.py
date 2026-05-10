"""Smoke test: recoleta clima 2024 completo e roda correlações Pearson/Spearman.

Recoleta porque o banco tem só Jan-Fev/2024 (60 dias diários, 9 semanas).
Para correlações estatisticamente úteis precisamos do ano todo (52 semanas).
"""

from datetime import date

import duckdb
from loguru import logger

from observatorio_dengue.etl import database, openmeteo
from observatorio_dengue.features.correlacoes import matriz_correlacoes

# Maringá, PR
LAT_MARINGA = -23.4205
LON_MARINGA = -51.9333

DATA_INICIO = date(2024, 1, 1)
DATA_FIM = date(2024, 12, 31)

DB_PATH = "data/processed/observatorio.duckdb"


def limpar_tabelas_clima(db_path: str) -> None:
    """Apaga clima_diario e clima_semanal antes de recolher."""
    con = duckdb.connect(db_path)
    con.execute("DELETE FROM clima_diario")
    con.execute("DELETE FROM clima_semanal")
    con.close()
    logger.info("tabelas clima_diario e clima_semanal limpas")


def main() -> None:
    # 1) Recoleta clima 2024 inteiro
    logger.info(f"coletando clima Maringá {DATA_INICIO} → {DATA_FIM}")
    df_diario = openmeteo.coletar_clima_diario(
        latitude=LAT_MARINGA,
        longitude=LON_MARINGA,
        data_inicio=DATA_INICIO,
        data_fim=DATA_FIM,
    )
    logger.info(f"coletados {len(df_diario)} dias")

    df_semanal = openmeteo.agregar_para_semana_epidemiologica(df_diario)
    logger.info(f"agregados em {len(df_semanal)} semanas epidemiológicas")

    # 2) Substitui dados antigos no DuckDB
    limpar_tabelas_clima(DB_PATH)
    n_diario = database.salvar_clima_diario(df_diario, latitude=LAT_MARINGA, longitude=LON_MARINGA)
    n_semanal = database.salvar_clima_semanal(
        df_semanal, latitude=LAT_MARINGA, longitude=LON_MARINGA
    )
    logger.info(f"salvos: {n_diario} dias, {n_semanal} semanas")

    # 3) Carrega dengue + clima do banco
    df_dengue = database.carregar("SELECT * FROM dengue_raw ORDER BY ano_epi, semana_epi")
    df_clima = database.carregar("SELECT * FROM clima_semanal ORDER BY ano_epi, semana_epi")
    logger.info(f"dengue: {len(df_dengue)} sem, clima: {len(df_clima)} sem")

    # 4) Roda matriz de correlações
    vars_clima = [
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "relative_humidity_2m_mean",
    ]

    matriz = matriz_correlacoes(
        df_dengue=df_dengue,
        df_clima_semanal=df_clima,
        vars_clima=vars_clima,
        lags=range(0, 9),
        coluna_dengue="p_inc100k",
        metodos=("pearson", "spearman"),
    )

    # 5) Mostra lag ótimo (maior |r|) por variável e método
    print("\n" + "=" * 70)
    print("LAG ÓTIMO POR VARIÁVEL CLIMÁTICA (Maringá 2024)")
    print("=" * 70)

    matriz["abs_r"] = matriz["r"].abs()
    otimos = (
        matriz.sort_values("abs_r", ascending=False)
        .groupby(["var_clima", "metodo"], as_index=False)
        .first()
        .sort_values(["var_clima", "metodo"])[["var_clima", "metodo", "lag", "r", "p_valor", "n"]]
    )
    print(otimos.to_string(index=False))

    print("\n" + "=" * 70)
    print("MATRIZ COMPLETA (lag × variável, Pearson)")
    print("=" * 70)
    pivot_p = matriz[matriz["metodo"] == "pearson"].pivot(
        index="lag", columns="var_clima", values="r"
    )
    print(pivot_p.round(3).to_string())

    print("\n" + "=" * 70)
    print("MATRIZ COMPLETA (lag × variável, Spearman)")
    print("=" * 70)
    pivot_s = matriz[matriz["metodo"] == "spearman"].pivot(
        index="lag", columns="var_clima", values="r"
    )
    print(pivot_s.round(3).to_string())

    # 6) Salva matriz como artefato
    out_path = "data/processed/correlacoes_maringa_2024.csv"
    matriz.drop(columns=["abs_r"]).to_csv(out_path, index=False)
    logger.info(f"matriz salva em {out_path}")


if __name__ == "__main__":
    main()
