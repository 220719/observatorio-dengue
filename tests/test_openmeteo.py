"""Testes para o módulo etl.openmeteo."""

from datetime import date

import pandas as pd
import pytest

from observatorio_dengue.etl.openmeteo import (
    _data_para_semana_epi,
    agregar_para_semana_epidemiologica,
    coletar_clima_diario,
)


class TestDataParaSemanaEpi:
    """Testes do helper _data_para_semana_epi (mapeamento data → semana ISO)."""

    def test_inicio_de_2024(self):
        """1 de janeiro de 2024 é segunda-feira → ano 2024, semana 1."""
        ts = pd.Timestamp("2024-01-01")
        ano, semana = _data_para_semana_epi(ts)
        assert ano == 2024
        assert semana == 1

    def test_fim_de_2020_tem_53_semanas(self):
        """31 de dezembro de 2020 cai na semana 53 do ano ISO 2020."""
        ts = pd.Timestamp("2020-12-31")
        ano, semana = _data_para_semana_epi(ts)
        assert ano == 2020
        assert semana == 53

    def test_inicio_de_2021_pode_pertencer_a_2020(self):
        """1, 2, 3 de janeiro de 2021 (sex, sab, dom) ainda fazem parte
        da semana 53 do ano ISO 2020 (segunda da semana 1 de 2021 é 4-jan)."""
        ts = pd.Timestamp("2021-01-03")
        ano, semana = _data_para_semana_epi(ts)
        assert ano == 2020
        assert semana == 53


class TestAgregarParaSemanaEpidemiologica:
    """Testes de agregação diária → semanal."""

    def test_dataframe_vazio_retorna_vazio(self):
        df_vazio = pd.DataFrame()
        resultado = agregar_para_semana_epidemiologica(df_vazio)
        assert resultado.empty

    def test_falta_coluna_data_levanta_erro(self):
        df_sem_data = pd.DataFrame({"temperature_2m_mean": [25.0, 26.0]})
        with pytest.raises(ValueError, match="coluna 'data'"):
            agregar_para_semana_epidemiologica(df_sem_data)

    def test_chuva_eh_somada_e_temperatura_media(self):
        """7 dias de uma semana, com agregações apropriadas por variável."""
        df = pd.DataFrame(
            {
                "data": pd.date_range("2024-01-01", periods=7),
                "temperature_2m_mean": [25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0],
                "temperature_2m_max": [30.0] * 7,
                "temperature_2m_min": [20.0] * 7,
                "precipitation_sum": [10.0, 5.0, 0.0, 15.0, 0.0, 0.0, 20.0],
                "relative_humidity_2m_mean": [80.0] * 7,
            }
        )
        resultado = agregar_para_semana_epidemiologica(df)

        assert len(resultado) == 1
        linha = resultado.iloc[0]
        assert linha["ano_epi"] == 2024
        assert linha["semana_epi"] == 1
        assert linha["temperature_2m_mean"] == 28.0
        assert linha["precipitation_sum"] == 50.0
        assert linha["dias_validos"] == 7

    def test_dias_atravessando_duas_semanas(self):
        """Dias do final de 2020 e início de 2021 em semanas distintas."""
        df = pd.DataFrame(
            {
                "data": pd.to_datetime(["2020-12-30", "2020-12-31", "2021-01-01", "2021-01-04"]),
                "temperature_2m_mean": [20.0, 21.0, 22.0, 23.0],
                "temperature_2m_max": [25.0] * 4,
                "temperature_2m_min": [15.0] * 4,
                "precipitation_sum": [0.0] * 4,
                "relative_humidity_2m_mean": [70.0] * 4,
            }
        )
        resultado = agregar_para_semana_epidemiologica(df)

        assert len(resultado) == 2
        anos_semanas = set(zip(resultado["ano_epi"], resultado["semana_epi"], strict=True))
        assert (2020, 53) in anos_semanas
        assert (2021, 1) in anos_semanas


class TestColetarClimaDiario:
    """Testes de validação de entrada do coletor.

    Não testamos a chamada à API real aqui — isso fica para smoke test manual.
    """

    def test_data_inicio_maior_que_fim_levanta_erro(self):
        with pytest.raises(ValueError, match="deve ser <="):
            coletar_clima_diario(
                latitude=-23.42,
                longitude=-51.93,
                data_inicio=date(2024, 6, 1),
                data_fim=date(2024, 1, 1),
            )

    def test_latitude_fora_de_range(self):
        with pytest.raises(ValueError, match="Latitude fora de range"):
            coletar_clima_diario(
                latitude=95.0,
                longitude=-51.93,
                data_inicio=date(2024, 1, 1),
                data_fim=date(2024, 1, 7),
            )

    def test_longitude_fora_de_range(self):
        with pytest.raises(ValueError, match="Longitude fora de range"):
            coletar_clima_diario(
                latitude=-23.42,
                longitude=-181.0,
                data_inicio=date(2024, 1, 1),
                data_fim=date(2024, 1, 7),
            )
