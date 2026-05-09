"""Testes para o módulo features.cruzamento."""

import pandas as pd
import pytest

from observatorio_dengue.features.cruzamento import (
    adicionar_lag,
    aplicar_lag_epiweek,
    cruzar_dengue_clima,
)


class TestAplicarLagEpiweek:
    """Testes do helper aplicar_lag_epiweek."""

    def test_lag_dentro_do_mesmo_ano(self):
        """Semana 1 + 4 = semana 5 do mesmo ano."""
        assert aplicar_lag_epiweek(2024, 1, 4) == (2024, 5)

    def test_lag_atravessa_ano_simples(self):
        """Semana 50 de 2024 (52 semanas ISO) + 4 = semana 2 de 2025."""
        # 2024 tem 52 semanas ISO. 50+4=54, ou seja, vira semana 2 do ano seguinte.
        ano, semana = aplicar_lag_epiweek(2024, 50, 4)
        assert ano == 2025
        assert semana == 2

    def test_lag_em_ano_com_53_semanas(self):
        """2020 tem 53 semanas ISO. Semana 50 + 4 = semana 1 de 2021."""
        # Esse era o bug do código antigo (assumia 52 semanas sempre).
        ano, semana = aplicar_lag_epiweek(2020, 50, 4)
        assert ano == 2021
        assert semana == 1

    def test_lag_zero_retorna_original(self):
        """Lag = 0 não muda nada."""
        assert aplicar_lag_epiweek(2024, 10, 0) == (2024, 10)

    def test_lag_negativo(self):
        """Lag negativo subtrai semanas (útil para análise reversa)."""
        # Semana 5 de 2024 - 4 = semana 1 de 2024
        assert aplicar_lag_epiweek(2024, 5, -4) == (2024, 1)


class TestAdicionarLag:
    """Testes da função adicionar_lag."""

    def test_dataframe_vazio_retorna_vazio(self):
        df_vazio = pd.DataFrame()
        resultado = adicionar_lag(df_vazio, lag=4)
        assert resultado.empty

    def test_falta_coluna_levanta_erro(self):
        df_sem_ano = pd.DataFrame({"semana_epi": [1, 2]})
        with pytest.raises(ValueError, match="Colunas obrigatórias faltando"):
            adicionar_lag(df_sem_ano, lag=4)

    def test_adiciona_colunas_target(self):
        df = pd.DataFrame(
            {
                "ano_epi": [2024, 2024],
                "semana_epi": [1, 2],
                "temperature_2m_mean": [25.0, 26.0],
            }
        )
        resultado = adicionar_lag(df, lag=4)
        assert "ano_target" in resultado.columns
        assert "semana_target" in resultado.columns
        assert resultado.iloc[0]["semana_target"] == 5
        assert resultado.iloc[1]["semana_target"] == 6

    def test_nao_modifica_dataframe_original(self):
        """Função não deve alterar o DataFrame de entrada (sem side effects)."""
        df = pd.DataFrame(
            {
                "ano_epi": [2024],
                "semana_epi": [1],
                "temperature_2m_mean": [25.0],
            }
        )
        colunas_originais = set(df.columns)
        _ = adicionar_lag(df, lag=4)
        assert set(df.columns) == colunas_originais


class TestCruzarDengueClima:
    """Testes do cruzamento dengue × clima com lag."""

    def test_dataframes_vazios_retorna_vazio(self):
        assert cruzar_dengue_clima(pd.DataFrame(), pd.DataFrame()).empty

    def test_falta_coluna_em_dengue_levanta_erro(self):
        df_dengue_invalido = pd.DataFrame({"casos": [10]})
        df_clima = pd.DataFrame(
            {
                "ano_epi": [2024],
                "semana_epi": [1],
                "temperature_2m_mean": [25.0],
            }
        )
        with pytest.raises(ValueError, match="df_dengue precisa"):
            cruzar_dengue_clima(df_dengue_invalido, df_clima)

    def test_cruzamento_basico_lag_4(self):
        """Clima da semana 1 deve aparecer na linha da semana 5 do dengue."""
        df_dengue = pd.DataFrame(
            {
                "ano_epi": [2024, 2024, 2024],
                "semana_epi": [3, 4, 5],
                "casos": [10, 15, 20],
            }
        )
        df_clima = pd.DataFrame(
            {
                "ano_epi": [2024],
                "semana_epi": [1],
                "temperature_2m_mean": [28.0],
                "precipitation_sum": [50.0],
            }
        )

        resultado = cruzar_dengue_clima(df_dengue, df_clima, lag=4)

        # 3 linhas (todas as do dengue preservadas com how="left")
        assert len(resultado) == 3

        # Apenas a semana 5 (= semana 1 do clima + lag 4) deve ter clima
        linha_semana_5 = resultado[resultado["semana_epi"] == 5].iloc[0]
        assert linha_semana_5["temperature_2m_mean_lag4"] == 28.0
        assert linha_semana_5["precipitation_sum_lag4"] == 50.0

        # Semanas 3 e 4 não têm clima correspondente → NaN
        linha_semana_3 = resultado[resultado["semana_epi"] == 3].iloc[0]
        assert pd.isna(linha_semana_3["temperature_2m_mean_lag4"])

    def test_lag_diferente_de_4(self):
        """Função deve aceitar lag configurável."""
        df_dengue = pd.DataFrame(
            {
                "ano_epi": [2024],
                "semana_epi": [3],
                "casos": [10],
            }
        )
        df_clima = pd.DataFrame(
            {
                "ano_epi": [2024],
                "semana_epi": [1],
                "temperature_2m_mean": [25.0],
            }
        )

        resultado = cruzar_dengue_clima(df_dengue, df_clima, lag=2)

        # Coluna deve ter sufixo _lag2 (não _lag4)
        assert "temperature_2m_mean_lag2" in resultado.columns
        assert resultado.iloc[0]["temperature_2m_mean_lag2"] == 25.0

    def test_cruzamento_atravessando_ano(self):
        """Clima do fim de 2020 (53 semanas ISO) deve cruzar com início de 2021."""
        df_dengue = pd.DataFrame(
            {
                "ano_epi": [2021],
                "semana_epi": [1],  # primeira semana de 2021
                "casos": [50],
            }
        )
        df_clima = pd.DataFrame(
            {
                "ano_epi": [2020],
                "semana_epi": [50],  # 50 + 4 = vira 1 de 2021 (em 2020 com 53 sem)
                "temperature_2m_mean": [22.0],
            }
        )

        resultado = cruzar_dengue_clima(df_dengue, df_clima, lag=4)

        assert len(resultado) == 1
        # Cruzamento deve funcionar mesmo cruzando o limite do ano
        assert resultado.iloc[0]["temperature_2m_mean_lag4"] == 22.0
