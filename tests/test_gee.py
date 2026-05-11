"""Testes para etl/gee.py.

Usa mocks para não bater no Earth Engine real durante CI. Os testes
validam: assinatura das funções, validação de inputs, transformações
de escala (NDVI / 10000, LST × 0.02 - 273.15), forward-fill do NDVI,
e tratamento de dados vazios.
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from observatorio_dengue.etl import gee


class TestObterGeometriaMunicipio:
    """Testes de obter_geometria_municipio."""

    @patch("observatorio_dengue.etl.gee.ee.Filter")
    @patch("observatorio_dengue.etl.gee.ee.FeatureCollection")
    def test_municipio_encontrado(self, mock_fc, _mock_filter):
        """Quando 1 município é encontrado, retorna geometria."""
        mock_geometria = MagicMock()
        mock_filtered = MagicMock()
        mock_filtered.size.return_value.getInfo.return_value = 1
        mock_filtered.first.return_value.geometry.return_value = mock_geometria
        mock_fc.return_value.filter.return_value = mock_filtered

        resultado = gee.obter_geometria_municipio("Maringá")

        assert resultado is mock_geometria

    @patch("observatorio_dengue.etl.gee.ee.Filter")
    @patch("observatorio_dengue.etl.gee.ee.FeatureCollection")
    def test_municipio_nao_encontrado_raises(self, mock_fc, _mock_filter):
        """Quando 0 municípios são encontrados, levanta ValueError."""
        mock_filtered = MagicMock()
        mock_filtered.size.return_value.getInfo.return_value = 0
        mock_fc.return_value.filter.return_value = mock_filtered

        with pytest.raises(ValueError, match="não encontrado"):
            gee.obter_geometria_municipio("Atlantis")


class TestColetarNdviDiario:
    """Testes de coletar_ndvi_diario."""

    def test_data_inicio_maior_que_fim_raises(self):
        geometria = MagicMock()
        with pytest.raises(ValueError, match="data_inicio"):
            gee.coletar_ndvi_diario(
                geometria,
                data_inicio=date(2024, 12, 31),
                data_fim=date(2024, 1, 1),
            )

    @patch("observatorio_dengue.etl.gee._extrair_serie_temporal")
    @patch("observatorio_dengue.etl.gee.ee.ImageCollection")
    def test_forward_fill_expande_para_diario(self, mock_ic, mock_extrair):
        """3 compostos quinzenais devem virar ~31 dias após forward-fill."""
        # Simula 3 compostos: dia 1, 17, 33 de janeiro
        df_compostos = pd.DataFrame(
            {
                "data": pd.to_datetime([date(2024, 1, 1), date(2024, 1, 17), date(2024, 2, 2)]),
                "NDVI": [5000, 6000, 7000],  # escala MODIS
            }
        )
        mock_extrair.return_value = df_compostos
        mock_ic.return_value.filterDate.return_value.filterBounds.return_value = MagicMock()

        geometria = MagicMock()
        df = gee.coletar_ndvi_diario(
            geometria,
            data_inicio=date(2024, 1, 1),
            data_fim=date(2024, 1, 31),
        )

        # 31 dias em janeiro
        assert len(df) == 31
        assert list(df.columns) == ["data", "ndvi"]
        # Conversão de escala: 5000 / 10000 = 0.5
        assert df.iloc[0]["ndvi"] == pytest.approx(0.5)
        # Forward-fill: dia 10 (entre composto 1 e 2) deve ter valor do dia 1
        assert df.iloc[9]["ndvi"] == pytest.approx(0.5)
        # Dia 17 em diante: composto 2 = 6000 / 10000 = 0.6
        assert df.iloc[16]["ndvi"] == pytest.approx(0.6)

    @patch("observatorio_dengue.etl.gee._extrair_serie_temporal")
    @patch("observatorio_dengue.etl.gee.ee.ImageCollection")
    def test_compostos_vazios_retorna_df_vazio(self, mock_ic, mock_extrair):
        mock_extrair.return_value = pd.DataFrame()
        mock_ic.return_value.filterDate.return_value.filterBounds.return_value = MagicMock()

        df = gee.coletar_ndvi_diario(
            MagicMock(),
            data_inicio=date(2024, 1, 1),
            data_fim=date(2024, 1, 31),
        )
        assert df.empty
        assert list(df.columns) == ["data", "ndvi"]


class TestColetarLstNightDiario:
    """Testes de coletar_lst_night_diario."""

    def test_data_inicio_maior_que_fim_raises(self):
        with pytest.raises(ValueError, match="data_inicio"):
            gee.coletar_lst_night_diario(
                MagicMock(),
                data_inicio=date(2024, 12, 31),
                data_fim=date(2024, 1, 1),
            )

    @patch("observatorio_dengue.etl.gee._extrair_serie_temporal")
    @patch("observatorio_dengue.etl.gee.ee.ImageCollection")
    def test_conversao_kelvin_para_celsius(self, mock_ic, mock_extrair):
        """LST MODIS DN → K (× 0.02) → °C (- 273.15)."""
        # DN = 14750 → K = 14750 * 0.02 = 295.0 → °C = 21.85
        df_bruto = pd.DataFrame(
            {
                "data": pd.to_datetime([date(2024, 1, 1), date(2024, 1, 2)]),
                "LST_Night_1km": [14750.0, 15000.0],
            }
        )
        mock_extrair.return_value = df_bruto
        mock_ic.return_value.filterDate.return_value.filterBounds.return_value = MagicMock()

        df = gee.coletar_lst_night_diario(
            MagicMock(),
            data_inicio=date(2024, 1, 1),
            data_fim=date(2024, 1, 2),
        )

        assert list(df.columns) == ["data", "lst_night_c"]
        # 14750 * 0.02 - 273.15 = 21.85
        assert df.iloc[0]["lst_night_c"] == pytest.approx(21.85, abs=0.01)
        # 15000 * 0.02 - 273.15 = 26.85
        assert df.iloc[1]["lst_night_c"] == pytest.approx(26.85, abs=0.01)

    @patch("observatorio_dengue.etl.gee._extrair_serie_temporal")
    @patch("observatorio_dengue.etl.gee.ee.ImageCollection")
    def test_dados_vazios_retorna_df_vazio(self, mock_ic, mock_extrair):
        mock_extrair.return_value = pd.DataFrame()
        mock_ic.return_value.filterDate.return_value.filterBounds.return_value = MagicMock()

        df = gee.coletar_lst_night_diario(
            MagicMock(),
            data_inicio=date(2024, 1, 1),
            data_fim=date(2024, 1, 31),
        )
        assert df.empty
        assert list(df.columns) == ["data", "lst_night_c"]


class TestConstantes:
    """Sanity check das constantes — fáceis de mexer sem querer."""

    def test_datasets_corretos(self):
        assert gee.DATASET_NDVI == "MODIS/061/MOD13Q1"
        assert gee.DATASET_LST == "MODIS/061/MOD11A1"
        assert gee.DATASET_MUNICIPIOS == "FAO/GAUL/2015/level2"

    def test_fatores_escala(self):
        assert gee.FATOR_NDVI == 10000
        assert gee.FATOR_LST == 0.02
        assert gee.KELVIN_PARA_CELSIUS == 273.15
