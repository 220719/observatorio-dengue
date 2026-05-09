"""Configuração centralizada de logging."""

import sys

from loguru import logger

from observatorio_dengue.config import settings


def setup_logger() -> None:
    """Configura o logger global."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>",
    )


setup_logger()
