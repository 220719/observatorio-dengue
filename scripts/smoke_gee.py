"""Smoke test do módulo gee.py.

Coleta NDVI (MOD13Q1, forward-fill diário) e LST_Night (MOD11A1, diário)
para Maringá em 2024, agrega para semana epidemiológica reusando
openmeteo.agregar_para_semana_epidemiologica, e imprime estatísticas.

NÃO persiste em DuckDB — isso fica para a próxima etapa.
"""

from datetime import date

from observatorio_dengue.etl import database, gee, openmeteo

PROJECT_ID = "observatorio-dengue-maringa"
MUNICIPIO = "Maringá"
DATA_INICIO = date(2024, 1, 1)
DATA_FIM = date(2024, 12, 31)


def main() -> None:
    print("=" * 70)
    print("SMOKE TEST: coleta GEE para Maringá 2024")
    print("=" * 70)

    # 1) Inicializa cliente
    gee.inicializar_gee(project_id=PROJECT_ID)

    # 2) Geometria do município
    print(f"\n[1/4] Obtendo geometria de '{MUNICIPIO}'...")
    geometria = gee.obter_geometria_municipio(MUNICIPIO)

    # 3) Coleta NDVI (composto 16-day → forward-fill diário)
    print(f"\n[2/4] Coletando NDVI ({DATA_INICIO} → {DATA_FIM})...")
    df_ndvi = gee.coletar_ndvi_diario(geometria, DATA_INICIO, DATA_FIM)
    print(f"      {len(df_ndvi)} dias coletados")
    print(
        f"      NDVI min={df_ndvi['ndvi'].min():.3f} "
        f"max={df_ndvi['ndvi'].max():.3f} "
        f"mean={df_ndvi['ndvi'].mean():.3f}"
    )

    # 4) Coleta LST_Night (diário)
    print(f"\n[3/4] Coletando LST_Night ({DATA_INICIO} → {DATA_FIM})...")
    df_lst = gee.coletar_lst_night_diario(geometria, DATA_INICIO, DATA_FIM)
    n_validos = df_lst["lst_night_c"].notna().sum()
    print(f"      {len(df_lst)} dias coletados ({n_validos} com valor válido)")
    if n_validos > 0:
        print(
            f"      LST_Night min={df_lst['lst_night_c'].min():.2f}°C "
            f"max={df_lst['lst_night_c'].max():.2f}°C "
            f"mean={df_lst['lst_night_c'].mean():.2f}°C"
        )

    # 5) Agregação semanal (reusa função do openmeteo)
    print("\n[4/4] Agregando para semana epidemiológica...")
    df_combinado = df_ndvi.merge(df_lst, on="data", how="outer")
    df_semanal = openmeteo.agregar_para_semana_epidemiologica(df_combinado)
    print(f"      {len(df_semanal)} semanas agregadas")

    print("\n" + "=" * 70)
    print("PRIMEIRAS 5 SEMANAS")
    print("=" * 70)
    print(df_semanal.head().to_string(index=False))

    print("\n" + "=" * 70)
    print("ESTATÍSTICAS POR VARIÁVEL")
    print("=" * 70)
    cols_numericas = df_semanal.select_dtypes(include="number").columns
    print(df_semanal[cols_numericas].describe().round(3).to_string())

    # 6) Persistir no DuckDB
    print("\n[BÔNUS] Persistindo no DuckDB...")
    database.criar_schema()  # idempotente — cria tabela se não existir
    n_salvas = database.salvar_satelite_semanal(df_semanal, municipio=MUNICIPIO)
    print(f"      {n_salvas} linhas salvas em satelite_municipio_semanal")

    # Confirma lendo de volta
    df_check = database.carregar(
        "SELECT COUNT(*) AS n, AVG(ndvi) AS ndvi_avg, "
        "AVG(lst_night_c) AS lst_avg FROM satelite_municipio_semanal"
    )
    print(f"      Confirmação: {df_check.iloc[0]['n']} linhas, "
          f"NDVI avg={df_check.iloc[0]['ndvi_avg']:.3f}, "
          f"LST avg={df_check.iloc[0]['lst_avg']:.2f}°C")

    print("\n✅ Smoke test OK — módulo gee.py funcional end-to-end.")


if __name__ == "__main__":
    main()
