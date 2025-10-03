#!/usr/bin/env python3
"""
OrionTrader - Sistema Adaptativo de Trading Algorítmico Multi-Ativos
Ponto de entrada principal do bot
"""

import argparse
import json
import os
import sys
import threading
from dotenv import load_dotenv

# CARREGAR .env
load_dotenv()


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


def main():
    parser = argparse.ArgumentParser(description='OrionTrader - Bot de Trading Adaptativo Multi-Ativos')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['demo', 'live', 'backtest'],
                        help='Modo de operação: demo, live ou backtest')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Caminho para o arquivo de configuração')
    args = parser.parse_args()

    # Carregar configurações e logger
    config = load_config(args.config)

    # Importar e configurar o logger
    try:
        from bot.utils.logger import setup_logger
        logger = setup_logger(config)
    except ImportError:
        print("ERRO: Não foi possível carregar o módulo do logger a partir de 'bot.utils.logger'.")
        print("Verifique a estrutura de pastas do seu projeto.")
        sys.exit(1)

    # ✅ LÓGICA DE EXECUÇÃO SEPARADA POR MODO
    try:
        if args.mode == 'backtest':
            # ----------------------------------------------------
            # --- MODO BACKTEST: Usa o motor rápido e vetorizado ---
            # ----------------------------------------------------
            logger.info(f"Modo '{args.mode}' selecionado. Iniciando o motor de backtest vetorizado.")
            try:
                from bot.backtest.engine import BacktestEngine
                engine = BacktestEngine(config, logger)
                engine.run()
            except ImportError:
                logger.error("ERRO: Não foi possível carregar o BacktestEngine a partir de 'bot.backtest.engine'.")
                sys.exit(1)

        else:  # Modos 'demo' ou 'live'
            # -----------------------------------------------------------------
            # --- MODO REAL/DEMO: Usa o OrionTrader para operação contínua ---
            # -----------------------------------------------------------------
            logger.info(f"Modo '{args.mode}' selecionado. Iniciando o OrionTrader para operação em tempo real.")
            try:
                from bot.trader.orion_trader import OrionTrader
                trader = OrionTrader(config, args.mode, logger=logger)
                trader.run()
            except ImportError:
                logger.error("ERRO: Não foi possível carregar o OrionTrader a partir de 'bot.trader.orion_trader'.")
                sys.exit(1)


    except KeyboardInterrupt:
        logger.info("Processo interrompido pelo usuário.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Erro fatal não capturado na execução: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()