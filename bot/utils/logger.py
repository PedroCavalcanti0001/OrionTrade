"""
Sistema de logging robusto para o OrionTrader
"""

import sys
from loguru import logger


"""
Sistema de logging robusto para o OrionTrader
"""
import sys
from loguru import logger
from typing import Dict

def setup_logger(config: Dict):
    """
    Configura e retorna o logger do sistema a partir de um dicionário de configuração.
    Esta função deve ser chamada APENAS UMA VEZ no início da aplicação.
    """
    # Obter configurações de log do arquivo de config, com valores padrão seguros
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO').upper()
    log_file = log_config.get('file', 'logs/orion_trader.log')

    # Remove o handler padrão para evitar duplicação
    logger.remove()

    # Adiciona handler para o console (stdout)
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>",
        colorize=True
    )

    # Adiciona handler para o arquivo, usando o mesmo nível de log
    logger.add(
        log_file,
        rotation="10 MB",
        retention="30 days",
        level=log_level,  # ✅ CORREÇÃO: Usa o nível do arquivo de configuração
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )

    logger.info(f"Logger configurado com nível: {log_level}")
    return logger