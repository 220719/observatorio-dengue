"""Onda 3 - Etapa 1: Coleta expandida 2020-2025 (dengue + clima + satélite).

Recoleta as 3 fontes para o período 2020-2025 e persiste no DuckDB,
substituindo os dados anteriores (que eram só 2024).

Tempo estimado: ~2-5 min (GEE é o gargalo).

Saída: data/processed/observatorio.duckdb atualizado com ~280 semanas.
"""

from datetime import date

import duckdb
from loguru import logger

from observatorio_dengue.etl import database, gee, infodengue, openmeteo

# ── Parâmetros ───────────────────────────────────────────────────────────
GEOCODE_MARINGA = 4115200
NOME_MUNICIPIO = "Maringá"
LAT_MARINGA = -23.4205
LON_MARINGA = -51.9333

ANO_INICIO = 2020
ANO_FIM = 2025

DATA_INICIO = date(ANO_INICIO, 1, 1)
DATA_FIM = date(ANO_FIM, 12, 31)

PROJECT_ID = "observatorio-dengue-maringa"
DB_PATH = "data/processed/observatorio.duckdb"


def limpar_tabelas(db_path: str) -> None:
    """Limpa todas as tabelas antes de recarregar."""
    con = duckdb.connect(db_path)
    for tabela in ["dengue_raw", "clima_diario", "clima_semanal",
                    "satelite_municipio_semanal"]:
        con.execute(f"DELETE FROM {tabela}")
        logger.info(f"tabela {tabela} limpa")
    con.close()


def main() -> None:
    print("=" * 75)
    print(f"COLETA EXPANDIDA {ANO_INICIO}-{ANO_FIM} — Maringá")
    print("=" * 75)

    # ── 0. Limpa banco ───────────────────────────────────────────────────
    print("\n[0/4] Limpando DuckDB...")
    limpar_tabelas(DB_PATH)

    # ── 1. Dengue (InfoDengue) ───────────────────────────────────────────
    print(f"\n[1/4] Coletando dengue {ANO_INICIO}-{ANO_FIM}...")
    df_dengue = infodengue.coletar_municipio(
        geocode=GEOCODE_MARINGA,
        nome_municipio=NOME_MUNICIPIO,
        ano_inicio=ANO_INICIO,
        ano_fim=ANO_FIM,
    )
    n_dengue = database.salvar_dengue(df_dengue)
    print(f"      ✓ {n_dengue} semanas de dengue salvas")

    # ── 2. Clima (Open-Meteo ERA5) ──────────────────────────────────────
    print(f"\n[2/4] Coletando clima {DATA_INICIO} → {DATA_FIM}...")
    df_clima_diario = openmeteo.coletar_clima_diario(
        latitude=LAT_MARINGA,
        longitude=LON_MARINGA,
        data_inicio=DATA_INICIO,
        data_fim=DATA_FIM,
    )
    df_clima_semanal = openmeteo.agregar_para_semana_epidemiologica(df_clima_diario)

    n_diario = database.salvar_clima_diario(
        df_clima_diario, latitude=LAT_MARINGA, longitude=LON_MARINGA
    )
    n_semanal = database.salvar_clima_semanal(
        df_clima_semanal, latitude=LAT_MARINGA, longitude=LON_MARINGA
    )
    print(f"      ✓ {n_diario} dias + {n_semanal} semanas de clima salvas")

    # ── 3. Satélite (GEE MODIS NDVI + LST_Night) ────────────────────────
    print(f"\n[3/4] Coletando satélite {DATA_INICIO} → {DATA_FIM} (pode demorar)...")
    gee.inicializar_gee(project_id=PROJECT_ID)
    geometria = gee.obter_geometria_municipio(NOME_MUNICIPIO)

    df_ndvi = gee.coletar_ndvi_diario(geometria, DATA_INICIO, DATA_FIM)
    print(f"      NDVI: {len(df_ndvi)} dias coletados")

    df_lst = gee.coletar_lst_night_diario(geometria, DATA_INICIO, DATA_FIM)
    print(f"      LST_Night: {len(df_lst)} dias coletados")

    df_sat_diario = df_ndvi.merge(df_lst, on="data", how="outer")
    df_sat_semanal = openmeteo.agregar_para_semana_epidemiologica(df_sat_diario)

    # Adiciona coluna municipio (exigida por salvar_satelite_semanal)
    df_sat_semanal["municipio"] = NOME_MUNICIPIO

    n_sat = database.salvar_satelite_semanal(df_sat_semanal, municipio=NOME_MUNICIPIO)
    print(f"      ✓ {n_sat} semanas de satélite salvas")

    # ── 4. Resumo final ─────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("RESUMO")
    print("=" * 75)
    for tabela in ["dengue_raw", "clima_semanal", "satelite_municipio_semanal"]:
        df = database.carregar(
            f"SELECT MIN(ano_epi) as min_ano, MAX(ano_epi) as max_ano, "
            f"COUNT(*) as n FROM {tabela}"
        )
        info = df.iloc[0]
        print(f"  {tabela}: {info['n']} semanas ({info['min_ano']}-{info['max_ano']})")

    print("\n✅ Coleta expandida concluída!")


if __name__ == "__main__":
    main()