"""
Sistema de logging robusto para o OrionTrader
"""

import sys
from loguru import logger


def setup_logger(level="INFO"):
    """Configura e retorna o logger do sistema"""

    # Remove o handler padrão
    logger.remove()

    # Adiciona handler para stdout
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )

    # Adiciona handler para arquivo
    logger.add(
        "logs/orion_trader.log",
        rotation="10 MB",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )

    return logger