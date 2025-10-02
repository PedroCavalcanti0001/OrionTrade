"""
Sistema de logging robusto para o OrionTrader
"""

import sys
from loguru import logger

# Tentar importar a variável global do main
try:
    from main import get_log_level, log_level as global_log_level
    HAS_GLOBAL_ACCESS = True
except ImportError:
    HAS_GLOBAL_ACCESS = False


def setup_logger(log_level=None):
    """Configura e retorna o logger do sistema"""

    # Prioridade: 1. Parâmetro, 2. Variável global, 3. Default "INFO"
    if log_level is not None:
        final_level = log_level
    elif HAS_GLOBAL_ACCESS:
        try:
            final_level = get_log_level()
            print(f"Usando nível de log global: {final_level}")
        except:
            final_level = "DEBUG"
    else:
        final_level = "DEBUG"

    print(f"Logger configurado com nível: {final_level}")

    # Remove o handler padrão
    logger.remove()

    # Adiciona handler para stdout
    logger.add(
        sys.stdout,
        level=final_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )

    # Adiciona handler para arquivo (sempre DEBUG para arquivo)
    logger.add(
        "logs/orion_trader.log",
        rotation="10 MB",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )

    logger.info(f"Logger configurado com nível: {final_level}")
    return logger


def get_current_log_level():
    """Retorna o nível de log atual (para uso em outros módulos)"""
    if HAS_GLOBAL_ACCESS:
        try:
            return get_log_level()
        except:
            return "DEBUG"
    return "DEBUG"