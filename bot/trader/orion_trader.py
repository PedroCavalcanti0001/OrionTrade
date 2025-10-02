"""
OrionTrader - Classe Principal de Orquestração
"""

import time
import os
from typing import Dict, List
import pandas as pd

from bot.connection.connector import IQConnector
from bot.connection.mock_connector import MockConnector
from bot.analysis.strategy_selector import StrategySelector
from bot.risk_management.risk_manager import RiskManager
from bot.utils.market_hours import MarketHoursValidator
from bot.trader.multi_asset_manager import MultiAssetManager

class OrionTrader:
    """Classe principal que orquestra todo o sistema de trading"""

    def __init__(self, config: Dict, mode: str, logger):
        self.config = config
        self.mode = mode
        self.logger = logger

        # Inicializar componentes
        self._initialize_components()

        # INICIALIZAR MULTI-ASSET MANAGER ← ADICIONE ESTA LINHA
        self.multi_asset_manager = MultiAssetManager(
            self.connector,
            self.strategy_selector,
            self.risk_manager,
            config,
            logger
        )

        # Estado do trader
        self.running = False
        self.open_trades = []
        self.balance = 0.0

        self.logger.info(f"OrionTrader inicializado no modo {mode}")

    def _initialize_components(self):
        """Inicializa todos os componentes do sistema"""

        trading_config = self.config.get('trading', {})

        # Inicializar conector
        if self.mode == 'backtest':
            initial_balance = trading_config.get('initial_balance', 1000.0)
            self.connector = MockConnector(initial_balance)
        elif self.mode in ['demo', 'live']:
            email = os.getenv('IQ_EMAIL')
            password = os.getenv('IQ_PASSWORD')

            if not email or not password:
                raise ValueError("Credenciais IQ Option não encontradas nas variáveis de ambiente")

            account_type = 'PRACTICE' if self.mode == 'demo' else 'REAL'
            self.connector = IQConnector(email, password, account_type)
        else:
            raise ValueError(f"Modo inválido: {self.mode}")

        self.connector.market_validator = MarketHoursValidator(self.connector, self.logger)

        # Inicializar outros componentes
        self.strategy_selector = StrategySelector(self.config)
        self.risk_manager = RiskManager(self.config)

    def connect(self) -> bool:
        """Estabelece conexão com a plataforma"""
        self.logger.info("Conectando à plataforma...")
        return self.connector.connect()

    def get_market_data(self) -> pd.DataFrame:
        """Obtém e formata dados de mercado - VERSÃO SIMPLIFICADA"""
        try:
            asset = self.config['trading']['asset']
            timeframe = self.config['trading']['timeframe']

            # Usar o multi_asset_manager para obter dados
            df = self.multi_asset_manager.get_market_data_for_asset(asset, 100)

            if df.empty:
                self.logger.warning(f"Nenhum dado válido obtido para {asset}")

            return df

        except Exception as e:
            self.logger.error(f"Erro crítico ao obter dados de mercado: {e}")
            return pd.DataFrame()

    def execute_trading_cycle(self):
        """Executa um ciclo completo de trading multi-ativos"""
        try:
            # 1. Atualizar saldo
            self.balance = self.connector.get_balance()

            # 2. Analisar TODOS os ativos
            all_signals = self.multi_asset_manager.analyze_all_assets()

            if not all_signals:
                self.logger.warning("Nenhum sinal gerado para os ativos monitorados")
                return

            # 3. Log do resumo de ativos
            self._log_assets_summary(all_signals)

            # 4. Obter melhores sinais
            top_signals = self.multi_asset_manager.get_top_signals(all_signals)

            # 5. Executar trades nos melhores sinais
            for signal in top_signals:
                if self._should_execute_trade(signal):
                    self._execute_trade(signal)

            # 6. Verificar trades abertas
            #self._check_open_trades()

        except Exception as e:
            self.logger.error(f"Erro no ciclo de trading: {e}")

    def _log_assets_summary(self, all_signals: Dict):
        """Log do resumo de todos os ativos"""
        self.logger.info("=== RESUMO MULTI-ATIVOS ===")

        for asset, signal in all_signals.items():
            analysis = signal.get('analysis', {})
            state = self.multi_asset_manager.asset_states[asset]

            self.logger.info(
                f"{asset}: {signal['regime'].value} | "
                f"Ação: {signal['action']} | "
                f"Conf: {signal['confidence']:.2f} | "
                f"Pri: {signal.get('priority_score', 0):.2f} | "
                f"Trades: {state['open_trades']}"
            )

        self.logger.info("===========================")

    def _should_execute_trade(self, signal: Dict) -> bool:
        """Verifica se deve executar uma trade"""
        asset = signal['asset']

        # Verificar limites globais
        max_trades = self.config['trading'].get('max_open_trades', 3)
        if len(self.open_trades) >= max_trades:
            self.logger.debug(f"Limite global de trades atingido ({max_trades})")
            return False

        # Verificar limites por ativo
        asset_state = self.multi_asset_manager.asset_states[asset]
        max_per_asset = self.config['trading'].get('max_trades_per_asset', 1)
        if asset_state['open_trades'] >= max_per_asset:
            self.logger.debug(f"Limite de trades para {asset} atingido ({max_per_asset})")
            return False

        # Verificar confiança mínima
        if signal.get('confidence', 0) < 0.6:
            return False

        # Verificar regime (não operar em CHOPPY)
        if signal['regime'].value == 'CHOPPY':
            return False

        return True

    def _execute_trade(self, signal: Dict):
        """Executa uma trade para um ativo específico"""
        try:
            asset = signal['asset']
            direction = 'call' if signal['direction'] == 'LONG' else 'put'

            # Calcular tamanho da posição
            position_size = self.risk_manager.calculate_position_size(
                self.balance, signal['confidence']
            )

            # Colocar ordem
            order_id = self.connector.place_order(
                asset=asset,
                direction=direction,
                amount=position_size,
                expiration=1
            )

            if order_id:
                # Registrar trade com informações do ativo
                trade_info = {
                    'order_id': order_id,
                    'asset': asset,
                    'direction': direction,
                    'amount': position_size,
                    'signal': signal
                }
                self.open_trades.append(trade_info)

                # Atualizar contador do ativo
                self.multi_asset_manager.update_trade_count(asset, +1)

                self.logger.info(
                    f"Trade executada - Ativo: {asset}, "
                    f"Direção: {direction}, "
                    f"Valor: ${position_size:.2f}, "
                    f"Regime: {signal['regime'].value}"
                )

        except Exception as e:
            self.logger.error(f"Erro ao executar trade para {asset}: {e}")

    def _check_open_trades(self):
        """Verifica e atualiza trades abertas de todos os ativos"""
        completed_trades = []

        for trade in self.open_trades:
            order_id = trade['order_id']
            asset = trade['asset']

            result = self.connector.check_win(order_id)

            if result is not None:  # Trade finalizada
                completed_trades.append(trade)
                outcome = "GANHOU" if result else "PERDEU"

                self.logger.info(f"Trade {order_id} ({asset}) finalizada: {outcome}")

                # Atualizar contador do ativo
                self.multi_asset_manager.update_trade_count(asset, -1)

        # Remover trades finalizadas
        for trade in completed_trades:
            self.open_trades.remove(trade)

    def _log_market_analysis(self, signal: Dict):
        """Registra análise de mercado detalhada"""
        analysis = signal['analysis']

        self.logger.info(f"""
=== ANÁLISE DE MERCADO ===
Regime: {signal['regime'].value}
ADX: {analysis['adx']:.2f} ({analysis['adx_strength']})
Bandas BB: Largura {analysis['bb_width']:.4f}, Posição {analysis['bb_position']}
ATR: {analysis['atr']:.5f} ({analysis['atr_percent']:.2f}%)
RSI: {analysis['rsi']:.1f} ({analysis['rsi_signal']})
Tendência: {analysis['trend_direction']}
Estratégia: {signal['strategy_used']}
Ação: {signal['action']} {signal['direction'] or ''}
Confiança: {signal['confidence']:.2f}
==========================
        """)

    def _execute_trade(self, df: pd.DataFrame, signal: Dict):
        """Executa uma trade baseada no sinal"""
        try:
            asset = self.config['trading']['asset']
            direction = 'call' if signal['direction'] == 'LONG' else 'put'

            # Calcular tamanho da posição
            position_size = self.risk_manager.calculate_position_size(
                self.balance, signal['confidence']
            )

            # Colocar ordem
            order_id = self.connector.place_order(
                asset=asset,
                direction=direction,
                amount=position_size,
                expiration=1  # 1 minuto para day trade
            )

            if order_id:
                self.open_trades.append(order_id)
                self.logger.info(f"Trade executada - ID: {order_id}, Direção: {direction}, Valor: ${position_size:.2f}")

                # Log de métricas de risco
                risk_metrics = self.risk_manager.get_risk_metrics(df, self.balance)
                self.logger.info(f"Métricas de Risco: {risk_metrics}")

        except Exception as e:
            self.logger.error(f"Erro ao executar trade: {e}")

    def _check_open_trades(self):
        """Verifica e atualiza trades abertas"""
        completed_trades = []

        for order_id in self.open_trades:
            result = self.connector.check_win(order_id)

            if result is not None:  # Trade finalizada
                completed_trades.append(order_id)
                outcome = "GANHOU" if result else "PERDEU"
                self.logger.info(f"Trade {order_id} finalizada: {outcome}")

        # Remover trades finalizadas da lista
        for order_id in completed_trades:
            self.open_trades.remove(order_id)

    def list_available_assets(self):
        """Lista todos os ativos disponíveis na plataforma - VERSÃO CORRIGIDA"""
        try:
            available_assets = []

            # Verificar se market_validator existe
            if (hasattr(self.connector, 'market_validator') and
                    hasattr(self.connector.market_validator, 'get_available_assets')):
                available_assets = self.connector.market_validator.get_available_assets()

            self.logger.info("=== ATIVOS DISPONÍVEIS NA PLATAFORMA ===")

            if available_assets:
                # Filtrar e categorizar ativos
                forex_pairs = [asset for asset in available_assets if
                               any(x in asset for x in ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD'])]
                indices = [asset for asset in available_assets if
                           any(x in asset for x in ['US30', 'SPX', 'NAS', 'GER30', 'FRA40', 'UK100', 'JP225'])]
                commodities = [asset for asset in available_assets if
                               any(x in asset for x in ['XAU', 'XAG', 'OIL', 'GAS'])]
                crypto = [asset for asset in available_assets if
                          any(x in asset for x in ['BTC', 'ETH', 'LTC', 'XRP', 'BCH'])]

                self.logger.info(
                    f"FOREX ({len(forex_pairs)}): {forex_pairs[:8]}{'...' if len(forex_pairs) > 8 else ''}")
                self.logger.info(f"ÍNDICES ({len(indices)}): {indices}")
                self.logger.info(f"COMMODITIES ({len(commodities)}): {commodities}")
                self.logger.info(f"CRYPTO ({len(crypto)}): {crypto}")
            else:
                self.logger.info("Não foi possível obter lista de ativos da plataforma")
                # Mostrar ativos padrão como fallback
                default_assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'XAUUSD', 'XAGUSD',
                                  'BTCUSD']
                self.logger.info(f"Ativos padrão disponíveis: {default_assets}")

            # Mostrar ativos configurados
            configured_assets = self.multi_asset_manager.assets
            self.logger.info(f"ATIVOS CONFIGURADOS NO BOT: {configured_assets}")

            self.logger.info("========================================")

        except Exception as e:
            self.logger.error(f"Erro ao listar ativos disponíveis: {e}")
            # Fallback: mostrar ativos configurados mesmo com erro
            configured_assets = self.multi_asset_manager.assets
            self.logger.info(f"ATIVOS CONFIGURADOS: {configured_assets}")

    def run(self):
        """Loop principal de execução"""
        if not self.connect():
            self.logger.error("Falha na conexão, encerrando...")
            return

        self.running = True
        self.logger.info("Iniciando loop principal de trading...")

        # LISTAR ATIVOS DISPONÍVEIS
        self.list_available_assets()

        # LOG INICIAL DO STATUS DO MERCADO
        self.log_market_status()

        check_interval = self.config.get('execution', {}).get('check_interval', 10)

        try:
            while self.running:
                start_time = time.time()

                self.execute_trading_cycle()

                # Calcular tempo de espera para próximo ciclo
                elapsed = time.time() - start_time
                sleep_time = max(1, check_interval - elapsed)

                time.sleep(sleep_time)

        except KeyboardInterrupt:
            self.logger.info("Interrompido pelo usuário")
        except Exception as e:
            self.logger.error(f"Erro no loop principal: {e}")
        finally:
            self.shutdown()

    def log_market_status(self):
        """Log do status dos ativos - SEM VALIDAÇÃO DE MERCADO"""
        try:
            assets = self.multi_asset_manager.assets

            self.logger.info("=== ATIVOS MONITORADOS ===")

            for asset in assets:
                # SEM verificação de abertura - assume todos abertos
                self.logger.info(f"Ativo: {asset}")

            self.logger.info("========================")

        except Exception as e:
            self.logger.error(f"Erro ao listar ativos: {e}")

    def shutdown(self):
        """Encerra o trader de forma segura"""
        self.logger.info("Encerrando OrionTrader...")
        self.running = False
        self.connector.disconnect()
        self.logger.info("OrionTrader encerrado com sucesso")