#!/usr/bin/env python3
"""
OrionTrader - Sistema Adaptativo de Trading Algorítmico Multi-Ativos
Ponto de entrada principal do bot
"""

import argparse
import importlib
import json
import os
import sys
import threading
from pathlib import Path
from dotenv import load_dotenv

# CARREGAR .env
load_dotenv()

# VARIÁVEL GLOBAL para nível de logging
log_level = "INFO"  # Valor padrão


# SILENCIAR ERROS DE DIGITAL OPTIONS
def silence_digital_errors():
    """Silencia os erros específicos de digital options"""
    original_excepthook = threading.excepthook

    def custom_excepthook(args):
        # Ignorar erros específicos de digital options
        if (hasattr(args, 'exc_value') and
                isinstance(args.exc_value, KeyError) and
                str(args.exc_value) == "'underlying'"):
            return  # Silencia este erro
        # Ignorar erros de threads de digital options
        if (hasattr(args, 'exc_type') and
                hasattr(args, 'exc_value') and
                "digital" in str(args.exc_value).lower()):
            return
        original_excepthook(args)

    threading.excepthook = custom_excepthook


# Aplicar o silenciamento
silence_digital_errors()


def load_module(module_path, class_name=None):
    """Carrega módulos dinamicamente para evitar problemas de import"""
    try:
        spec = importlib.util.spec_from_file_location(module_path, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if class_name:
            return getattr(module, class_name)
        return module
    except Exception as e:
        print(f"Erro ao carregar módulo {module_path}: {e}")
        return None


def load_config(config_path="config.json"):
    """Carrega configuração do arquivo JSON"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Erro: Arquivo de configuração {config_path} não encontrado.")
        print("Copie config.json.example para config.json e ajuste as configurações.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Erro: Arquivo de configuração {config_path} contém JSON inválido.")
        sys.exit(1)


def get_log_level():
    """Retorna o nível de logging global"""
    global log_level
    return log_level


def set_log_level(level):
    """Define o nível de logging global"""
    global log_level
    log_level = level
    print(f"Log level definido globalmente para: {log_level}")


def main():
    """Função principal"""
    global log_level  # Declarar como global para modificar a variável externa

    parser = argparse.ArgumentParser(description='OrionTrader - Bot de Trading Adaptativo Multi-Ativos')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['demo', 'live', 'backtest'],
                        help='Modo de operação: demo, live ou backtest')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Caminho para o arquivo de configuração')
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Nível de logging: DEBUG, INFO, WARNING, ERROR, CRITICAL (padrão: INFO)'
    )
    args = parser.parse_args()

    # DEFINIR VARIÁVEL GLOBAL
    log_level = args.log_level
    print(f"Variável global log_level definida para: {log_level}")

    # Carregar configurações
    config = load_config(args.config)

    # Tentar carregar os módulos de forma flexível
    OrionTrader = None
    setup_logger = None

    # Tentar diferentes caminhos de import
    import importlib.util

    # Carregar OrionTrader
    import_paths = [
        ('bot.trader.orion_trader', 'OrionTrader'),
        ('trader.orion_trader', 'OrionTrader'),
        ('orion_trader', 'OrionTrader'),
    ]

    logger_paths = [
        'bot.utils.logger',
        'utils.logger',
        'logger'
    ]

    # Carregar OrionTrader
    for module_path, class_name in import_paths:
        try:
            if '.' in module_path:
                module = __import__(module_path, fromlist=[class_name])
                OrionTrader = getattr(module, class_name)
            else:
                module = __import__(module_path)
                OrionTrader = getattr(module, class_name)
            print(f"✓ OrionTrader carregado de: {module_path}")
            break
        except ImportError as e:
            print(f"✗ Falha ao carregar de {module_path}: {e}")
            continue

    # Carregar setup_logger
    for module_path in logger_paths:
        try:
            if '.' in module_path:
                module = __import__(module_path, fromlist=['setup_logger'])
                setup_logger = getattr(module, 'setup_logger')
            else:
                module = __import__(module_path)
                setup_logger = getattr(module, 'setup_logger')
            print(f"✓ Logger carregado de: {module_path}")
            break
        except ImportError as e:
            print(f"✗ Falha ao carregar logger de {module_path}: {e}")
            continue

    if not OrionTrader or not setup_logger:
        print("ERRO: Não foi possível carregar os módulos necessários!")
        print("Verifique a estrutura de arquivos:")
        print("Deve ter: bot/trader/orion_trader.py OU trader/orion_trader.py")
        sys.exit(1)

    # Configurar logger usando a variável global
    logger = setup_logger(log_level=log_level)

    try:
        # Inicializar e executar o trader
        trader = OrionTrader(config, args.mode, logger)

        # Verificar se a variável global está acessível
        print(f"Log level global antes de executar: {log_level}")

        trader.run()

    except KeyboardInterrupt:
        logger.info("Bot interrompido pelo usuário")
    except Exception as e:
        logger.error(f"Erro fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()