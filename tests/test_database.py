"""Testes para o módulo etl.database."""

from pathlib import Path

import pandas as pd
import pytest

from observatorio_dengue.etl.database import (
    carregar,
    criar_schema,
    get_connection,
    salvar_clima_diario,
    salvar_clima_semanal,
    salvar_dengue,
)


@pytest.fixture
def db_tmp(tmp_path: Path) -> Path:
    """Fixture que cria um banco DuckDB temporário com schema criado."""
    db_path = tmp_path / "test.duckdb"
    criar_schema(db_path)
    return db_path


class TestSchema:
    """Testes de criação de schema."""

    def test_criar_schema_idempotente(self, tmp_path: Path):
        """Chamar criar_schema várias vezes não deve dar erro."""
        db_path = tmp_path / "test.duckdb"
        criar_schema(db_path)
        criar_schema(db_path)  # segunda vez não deve quebrar

        with get_connection(db_path) as con:
            tabelas = con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' ORDER BY table_name"
            ).fetchall()
        nomes = [t[0] for t in tabelas]
        assert "dengue_raw" in nomes
        assert "clima_diario" in nomes
        assert "clima_semanal" in nomes


class TestSalvarDengue:
    """Testes da função salvar_dengue."""

    def test_dataframe_vazio_retorna_zero(self, db_tmp: Path):
        df_vazio = pd.DataFrame()
        assert salvar_dengue(df_vazio, db_tmp) == 0

    def test_falta_coluna_obrigatoria_levanta_erro(self, db_tmp: Path):
        df_sem_se = pd.DataFrame(
            {
                "geocode": [4115200],
                "municipio": ["Maringá"],
                "casos": [10],
            }
        )
        with pytest.raises(ValueError, match="Colunas obrigatórias faltando"):
            salvar_dengue(df_sem_se, db_tmp)

    def test_inserir_e_recuperar(self, db_tmp: Path):
        df = pd.DataFrame(
            {
                "geocode": [4115200, 4115200],
                "municipio": ["Maringá", "Maringá"],
                "SE": [202401, 202402],
                "casos": [10, 25],
                "casos_est": [10.5, 26.0],
                "nivel": [1, 2],
                "p_inc100k": [2.5, 6.3],
            }
        )
        n = salvar_dengue(df, db_tmp)
        assert n == 2

        resultado = carregar(
            "SELECT geocode, municipio, ano_epi, semana_epi, casos "
            "FROM dengue_raw ORDER BY semana_epi",
            db_tmp,
        )
        assert len(resultado) == 2
        assert resultado.iloc[0]["ano_epi"] == 2024
        assert resultado.iloc[0]["semana_epi"] == 1
        assert resultado.iloc[0]["casos"] == 10


class TestSalvarClimaDiario:
    """Testes da função salvar_clima_diario."""

    def test_dataframe_vazio_retorna_zero(self, db_tmp: Path):
        assert salvar_clima_diario(pd.DataFrame(), -23.42, -51.93, db_tmp) == 0

    def test_falta_coluna_data_levanta_erro(self, db_tmp: Path):
        df_sem_data = pd.DataFrame({"temperature_2m_mean": [25.0]})
        with pytest.raises(ValueError, match="coluna 'data'"):
            salvar_clima_diario(df_sem_data, -23.42, -51.93, db_tmp)

    def test_inserir_e_recuperar(self, db_tmp: Path):
        df = pd.DataFrame(
            {
                "data": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "temperature_2m_mean": [26.5, 27.0],
                "temperature_2m_max": [30.0, 31.0],
                "temperature_2m_min": [22.0, 23.0],
                "precipitation_sum": [0.0, 5.2],
                "relative_humidity_2m_mean": [70.0, 75.0],
            }
        )
        n = salvar_clima_diario(df, -23.42, -51.93, db_tmp)
        assert n == 2

        resultado = carregar(
            "SELECT data, temperature_2m_mean, precipitation_sum FROM clima_diario ORDER BY data",
            db_tmp,
        )
        assert len(resultado) == 2
        assert resultado.iloc[1]["precipitation_sum"] == 5.2


class TestSalvarClimaSemanal:
    """Testes da função salvar_clima_semanal."""

    def test_dataframe_vazio_retorna_zero(self, db_tmp: Path):
        assert salvar_clima_semanal(pd.DataFrame(), -23.42, -51.93, db_tmp) == 0

    def test_inserir_e_recuperar(self, db_tmp: Path):
        df = pd.DataFrame(
            {
                "ano_epi": [2024, 2024],
                "semana_epi": [1, 2],
                "temperature_2m_mean": [26.5, 27.0],
                "temperature_2m_max": [30.0, 31.0],
                "temperature_2m_min": [22.0, 23.0],
                "precipitation_sum": [10.0, 25.5],
                "relative_humidity_2m_mean": [70.0, 75.0],
                "dias_validos": [7, 7],
            }
        )
        n = salvar_clima_semanal(df, -23.42, -51.93, db_tmp)
        assert n == 2

        resultado = carregar(
            "SELECT ano_epi, semana_epi, precipitation_sum, dias_validos "
            "FROM clima_semanal ORDER BY semana_epi",
            db_tmp,
        )
        assert len(resultado) == 2
        assert resultado.iloc[0]["precipitation_sum"] == 10.0
        assert resultado.iloc[1]["dias_validos"] == 7


class TestCarregar:
    """Testes da função carregar (queries ad-hoc)."""

    def test_query_simples(self, db_tmp: Path):
        df = carregar("SELECT 1 AS x, 'hello' AS texto", db_tmp)
        assert len(df) == 1
        assert df.iloc[0]["x"] == 1
        assert df.iloc[0]["texto"] == "hello"

    def test_query_em_tabela_vazia(self, db_tmp: Path):
        df = carregar("SELECT * FROM dengue_raw", db_tmp)
        assert df.empty
