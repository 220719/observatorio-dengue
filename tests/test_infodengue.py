"""Testes para o módulo etl.infodengue."""

import pytest

from observatorio_dengue.etl.infodengue import parse_semana_epidemiologica


class TestParseSemanaEpidemiologica:
    """Testes do parser de campo SE do InfoDengue."""

    def test_parse_basico(self):
        """SE 202401 = ano 2024, semana 1."""
        assert parse_semana_epidemiologica(202401) == (2024, 1)

    def test_parse_string(self):
        """Aceita também strings."""
        assert parse_semana_epidemiologica("202401") == (2024, 1)

    def test_parse_semana_53(self):
        """Anos com 53 semanas ISO (ex: 2020) devem funcionar."""
        assert parse_semana_epidemiologica(202053) == (2020, 53)

    def test_parse_primeira_semana_do_ano(self):
        """SE 202301 = primeira semana de 2023."""
        ano, semana = parse_semana_epidemiologica(202301)
        assert ano == 2023
        assert semana == 1

    def test_rejeita_formato_curto(self):
        """SE com menos de 6 dígitos é inválido."""
        with pytest.raises(ValueError, match="formato YYYYWW"):
            parse_semana_epidemiologica(20241)

    def test_rejeita_ano_fora_range(self):
        """Anos absurdos devem ser rejeitados."""
        with pytest.raises(ValueError, match="Ano fora de range"):
            parse_semana_epidemiologica(199001)

    def test_rejeita_semana_invalida(self):
        """Semana 0 ou >53 deve ser rejeitada."""
        with pytest.raises(ValueError, match="Semana fora de range"):
            parse_semana_epidemiologica(202400)
        with pytest.raises(ValueError, match="Semana fora de range"):
            parse_semana_epidemiologica(202454)
