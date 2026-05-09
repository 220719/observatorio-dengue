"""Testes de sanidade da configuração."""

from pathlib import Path

from observatorio_dengue.config import settings


def test_settings_loaded():
    """Verifica que as settings foram carregadas."""
    assert isinstance(settings.data_raw_dir, Path)
    assert settings.data_raw_dir.exists()
    assert settings.data_processed_dir.exists()


def test_log_level_valid():
    """Verifica que o log level é um valor aceito."""
    assert settings.log_level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
