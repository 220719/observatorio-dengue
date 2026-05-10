"""Testes de features/correlacoes.py."""

import numpy as np
import pandas as pd
import pytest

from observatorio_dengue.features.correlacoes import (
    correlacao_par,
    correlacoes_lag,
    matriz_correlacoes,
)

# ---------- correlacao_par: núcleo ----------


class TestCorrelacaoPar:
    """Testes da função correlacao_par."""

    def test_correlacao_positiva_conhecida(self):
        """y = 0.8x + ruído pequeno → r ≈ 0.8 (Pearson)."""
        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, 100)
        y = 0.8 * x + rng.normal(0, 0.5, 100)
        res = correlacao_par(x, y, metodo="pearson")
        assert res["r"] > 0.9  # ruído baixo, r alto
        assert res["p_valor"] < 0.001
        assert res["n"] == 100
        assert res["metodo"] == "pearson"

    def test_correlacao_perfeita_negativa(self):
        x = np.arange(50)
        y = -x.astype(float)
        res = correlacao_par(x, y, metodo="pearson")
        assert res["r"] == pytest.approx(-1.0, abs=1e-9)
        assert res["n"] == 50

    def test_correlacao_zero(self):
        """Séries independentes → r próximo de zero."""
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 500)
        y = rng.normal(0, 1, 500)
        res = correlacao_par(x, y, metodo="pearson")
        assert abs(res["r"]) < 0.15

    def test_spearman_maior_que_pearson_em_relacao_nao_linear(self):
        """y = exp(x): monotônico mas não-linear → |Spearman| > |Pearson|."""
        x = np.linspace(0, 5, 100)
        y = np.exp(x)
        r_pearson = correlacao_par(x, y, metodo="pearson")["r"]
        r_spearman = correlacao_par(x, y, metodo="spearman")["r"]
        assert r_spearman == pytest.approx(1.0, abs=1e-9)
        assert r_spearman > r_pearson

    def test_remove_nan_pareados(self):
        x = pd.Series([1, 2, np.nan, 4, 5])
        y = pd.Series([2, 4, 6, np.nan, 10])
        res = correlacao_par(x, y, metodo="pearson")
        assert res["n"] == 3  # pares (1,2), (2,4), (5,10)
        assert res["r"] == pytest.approx(1.0, abs=1e-9)

    def test_serie_constante_retorna_nan(self):
        x = [1, 2, 3, 4, 5]
        y = [7, 7, 7, 7, 7]
        res = correlacao_par(x, y, metodo="pearson")
        assert np.isnan(res["r"])
        assert np.isnan(res["p_valor"])
        assert res["n"] == 5

    def test_n_menor_que_3_retorna_nan(self):
        res = correlacao_par([1, 2], [3, 4], metodo="pearson")
        assert np.isnan(res["r"])
        assert res["n"] == 2

    def test_metodo_invalido_raises(self):
        with pytest.raises(ValueError, match="inválido"):
            correlacao_par([1, 2, 3], [4, 5, 6], metodo="kendall")

    def test_tamanhos_diferentes_raises(self):
        with pytest.raises(ValueError, match="tamanhos diferentes"):
            correlacao_par([1, 2, 3], [4, 5], metodo="pearson")


# ---------- correlacoes_lag e matriz_correlacoes: integração ----------


@pytest.fixture
def df_dengue_sintetico():
    """52 semanas de dengue com pico no meio do ano."""
    semanas = list(range(1, 53))
    casos = [10 + 50 * np.exp(-((s - 26) ** 2) / 50) for s in semanas]
    return pd.DataFrame(
        {
            "ano_epi": [2024] * 52,
            "semana_epi": semanas,
            "casos": [int(c) for c in casos],
            "casos_est": [int(c) for c in casos],
            "p_inc100k": [c / 4.3 for c in casos],  # ~430k hab Maringá
        }
    )


@pytest.fixture
def df_clima_sintetico():
    """52 semanas de clima com temperatura defasando em 4 sem do pico de dengue."""
    semanas = list(range(1, 53))
    # pico de temperatura em sem 22 → dengue em sem 26 (lag=4)
    temp = [20 + 8 * np.exp(-((s - 22) ** 2) / 50) for s in semanas]
    return pd.DataFrame(
        {
            "ano_epi": [2024] * 52,
            "semana_epi": semanas,
            "temperature_2m_mean": temp,
            "precipitation_sum": [5.0] * 52,
            "relative_humidity_2m_mean": [70.0] * 52,
            "dias_validos": [7] * 52,
        }
    )


class TestCorrelacoesLag:
    def test_retorna_dataframe_com_estrutura_correta(self, df_dengue_sintetico, df_clima_sintetico):
        df = correlacoes_lag(
            df_dengue_sintetico,
            df_clima_sintetico,
            var_clima="temperature_2m_mean",
            lags=[0, 4],
            metodos=["pearson"],
        )
        assert set(df.columns) == {
            "var_clima",
            "metodo",
            "lag",
            "r",
            "p_valor",
            "n",
        }
        assert len(df) == 2  # 2 lags x 1 método
        assert set(df["lag"].tolist()) == {0, 4}

    def test_lag_otimo_corresponde_ao_sintetico(self, df_dengue_sintetico, df_clima_sintetico):
        """Dados sintéticos com lag=4 verdadeiro: |r| em lag=4 > |r| em lag=0."""
        df = correlacoes_lag(
            df_dengue_sintetico,
            df_clima_sintetico,
            var_clima="temperature_2m_mean",
            lags=range(0, 9),
            metodos=["pearson"],
        )
        df_pearson = df[df["metodo"] == "pearson"].set_index("lag")
        r_lag4 = abs(df_pearson.loc[4, "r"])
        r_lag0 = abs(df_pearson.loc[0, "r"])
        assert r_lag4 > r_lag0

    def test_pearson_e_spearman_ambos_retornados(self, df_dengue_sintetico, df_clima_sintetico):
        df = correlacoes_lag(
            df_dengue_sintetico,
            df_clima_sintetico,
            var_clima="temperature_2m_mean",
            lags=[4],
            metodos=["pearson", "spearman"],
        )
        assert set(df["metodo"].unique()) == {"pearson", "spearman"}

    def test_coluna_inexistente_raises(self, df_dengue_sintetico, df_clima_sintetico):
        with pytest.raises(KeyError, match="não encontrada"):
            correlacoes_lag(
                df_dengue_sintetico,
                df_clima_sintetico,
                var_clima="variavel_que_nao_existe",
                lags=[0],
                metodos=["pearson"],
            )


class TestMatrizCorrelacoes:
    def test_concatena_multiplas_variaveis(self, df_dengue_sintetico, df_clima_sintetico):
        df = matriz_correlacoes(
            df_dengue_sintetico,
            df_clima_sintetico,
            vars_clima=["temperature_2m_mean", "precipitation_sum"],
            lags=[0, 4],
            metodos=["pearson"],
        )
        assert set(df["var_clima"].unique()) == {
            "temperature_2m_mean",
            "precipitation_sum",
        }
        assert len(df) == 4  # 2 vars x 2 lags x 1 método

    def test_vars_vazias_retorna_df_vazio_com_colunas(
        self, df_dengue_sintetico, df_clima_sintetico
    ):
        df = matriz_correlacoes(
            df_dengue_sintetico,
            df_clima_sintetico,
            vars_clima=[],
        )
        assert df.empty
        assert list(df.columns) == [
            "var_clima",
            "metodo",
            "lag",
            "r",
            "p_valor",
            "n",
        ]
