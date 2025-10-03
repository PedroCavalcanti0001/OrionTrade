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
from bot.trader.performance_tracker import PerformanceTracker
from bot.utils.market_hours import MarketHoursValidator
from bot.trader.multi_asset_manager import MultiAssetManager


class OrionTrader:
    """Classe principal que orquestra todo o sistema de trading"""

    def __init__(self, config: Dict, mode: str, logger):
        self.config = config
        self.mode = mode
        self.logger = logger

        # ✅ INICIALIZAR PERFORMANCE TRACKER
        self.performance_tracker = PerformanceTracker(logger)
        self.performance_summary = {}  # Armazena o resumo do desempenho

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
            self.connector = MockConnector(initial_balance, logger=self.logger)
        elif self.mode in ['demo', 'live']:
            email = os.getenv('IQ_EMAIL')
            password = os.getenv('IQ_PASSWORD')

            if not email or not password:
                raise ValueError("Credenciais IQ Option não encontradas nas variáveis de ambiente")

            account_type = 'PRACTICE' if self.mode == 'demo' else 'REAL'
            self.connector = IQConnector(email, password, account_type, logger=self.logger)
        else:
            raise ValueError(f"Modo inválido: {self.mode}")

        self.connector.market_validator = MarketHoursValidator(self.connector, self.logger)

        # Inicializar outros componentes
        self.strategy_selector = StrategySelector(self.config, self.logger)
        self.risk_manager = RiskManager(self.config, self.logger)

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
        """Executa um ciclo completo de trading multi-ativos - ✅ VERSÃO DE ELITE"""
        try:
            # 1. Atualizar saldo
            self.balance = self.connector.get_balance()
            if self.balance is None or self.balance <= 0:
                self.logger.error("Balanço inválido ou zerado. Pausando operações.")
                return

            self.performance_summary = self.performance_tracker.get_summary()
            # ✅ ATUALIZAR O SELECTOR DE ESTRATÉGIAS COM O DESEMPENHO
            self.strategy_selector.update_strategy_performance(self.performance_summary)

            # ✅ ATUALIZAR MATRIZ DE CORRELAÇÃO PERIODICAMENTE
            self.multi_asset_manager.update_correlation_matrix_if_needed()

            # 2. Analisar TODOS os ativos
            all_signals = self.multi_asset_manager.analyze_all_assets()

            if not all_signals:
                self.logger.debug("Nenhum sinal gerado para os ativos monitorados neste ciclo.")
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
            self._check_open_trades()

        except Exception as e:
            self.logger.error(f"Erro crítico no ciclo de trading: {e}", exc_info=True)

    def _should_execute_trade(self, signal: Dict) -> bool:
        """Verifica se deve executar uma trade, incluindo filtro de correlação - ✅ VERSÃO DE ELITE"""
        asset = signal['asset']
        strategy_used = signal.get('strategy_used', 'UNKNOWN')

        # Se a estratégia está em cooldown, não deve executar a trade
        if strategy_used.endswith('_COOLDOWN'):
            self.logger.debug(f"Não executando trade para {asset} pois a estratégia {strategy_used} está em cooldown.")
            return False

        # Verificações básicas (limites, confiança, etc.)
        max_trades = self.config['trading'].get('max_open_trades', 3)
        if len(self.open_trades) >= max_trades:
            self.logger.debug(f"Limite global de trades atingido ({max_trades})")
            return False

        asset_state = self.multi_asset_manager.asset_states[asset]
        max_per_asset = self.config['trading'].get('max_trades_per_asset', 1)
        if asset_state['open_trades'] >= max_per_asset:
            self.logger.debug(f"Limite de trades para {asset} atingido ({max_per_asset})")
            return False

        min_confidence = self.config['trading'].get('min_confidence', 0.5)
        if signal.get('confidence', 0) < min_confidence:
            return False

        if signal['regime'].value == 'CHOPPY':
            return False

        # ✅ FILTRO DE ELITE: GERENCIAMENTO DE RISCO DE CORRELAÇÃO
        asset_to_trade = signal['asset']
        direction_to_trade = signal['direction']
        correlation_threshold = 0.80  # Limite de correlação (configurável)

        # Se a matriz de correlação não estiver vazia
        if not self.multi_asset_manager.correlation_matrix.empty:
            for open_trade in self.open_trades:
                open_asset = open_trade['asset']
                open_direction = open_trade['signal']['direction']

                # Se a direção do trade a ser aberto for a mesma do trade já aberto
                if open_direction == direction_to_trade:
                    try:
                        # Obter correlação entre o novo ativo e o ativo já em operação
                        correlation = self.multi_asset_manager.correlation_matrix.loc[asset_to_trade, open_asset]
                        if abs(correlation) > correlation_threshold:
                            self.logger.warning(
                                f"TRADE BLOQUEADA: Risco de correlação. "
                                f"{asset_to_trade} tem correlação de {correlation:.2f} com {open_asset} (limite: {correlation_threshold})."
                            )
                            return False
                    except KeyError:
                        # Acontece se um dos ativos não está na matriz, continuar normalmente
                        pass

        return True

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

    def _execute_trade(self, signal: Dict):
        """Executa uma trade, ajustando o risco com base no desempenho da estratégia - ✅ VERSÃO DE ELITE"""
        try:
            asset = signal['asset']
            direction = 'call' if signal['direction'] == 'LONG' else 'put'
            strategy_used = signal.get('strategy_used', 'UNKNOWN')

            # Limpar o nome do ativo (remover -op se presente)
            clean_asset = asset.replace('-op', '')

            # Verificar se o mercado está aberto para o ativo
            if not self.connector.market_validator.is_market_open(clean_asset):
                self.logger.warning(f"Mercado para {clean_asset} está fechado. Não é possível executar a ordem.")
                return

            # ✅ LÓGICA DE RISCO ADAPTATIVO
            performance_mod = 1.0  # Modificador de risco padrão
            strategy_perf = self.performance_summary.get(strategy_used)

            # Se a estratégia foi usada > 10 vezes e está com P/L negativo, reduzir o risco.
            if strategy_perf and strategy_perf['count'] > 10 and strategy_perf['pnl_total'] < 0:
                self.logger.warning(
                    f"Reduzindo risco para a estratégia '{strategy_used}' devido a desempenho negativo "
                    f"(P/L: {strategy_perf['pnl_total']:.2f} em {strategy_perf['count']} trades)."
                )
                performance_mod = 0.5  # Reduz o risco em 50%

            # Se a estratégia está com ótimo desempenho, bonificar o risco ligeiramente.
            elif strategy_perf and strategy_perf['count'] > 10 and strategy_perf['pnl_total'] > (
                    self.config['trading']['risk_per_trade'] * 5):
                self.logger.info(
                    f"Aumentando risco para a estratégia '{strategy_used}' devido a alto desempenho."
                )
                performance_mod = 1.2  # Aumenta o risco em 20%

            # Calcular tamanho da posição com o modificador
            position_size = self.risk_manager.calculate_position_size(
                self.balance, signal['confidence'], performance_modifier=performance_mod
            )

            # Colocar ordem
            result, order_id = self.connector.api.buy(
                position_size,
                clean_asset, # Usar o nome do ativo limpo
                direction,
                1
            )

            if result:
                trade_info = {
                    'order_id': order_id,
                    'asset': asset,
                    'direction': direction,
                    'amount': position_size,
                    'signal': signal
                }
                self.open_trades.append(trade_info)
                self.multi_asset_manager.update_trade_count(asset, +1)

                self.logger.info(
                    f"🎯 TRADE EXECUTADA - Ativo: {asset}, "
                    f"Direção: {direction}, "
                    f"Valor: ${position_size:.2f} (Mod: {performance_mod}x), "
                    f"Estratégia: {strategy_used}, "
                    f"Conf: {signal['confidence']:.2f}"
                )
            else:
                self.logger.error(f"Falha ao executar ordem para {asset}: {order_id}")

        except Exception as e:
            self.logger.error(f"Erro ao executar trade para {asset}: {e}")

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

    def _check_open_trades(self):
        """Verifica e atualiza trades abertas, registrando a performance - ✅ VERSÃO DE ELITE"""
        for trade_info in list(self.open_trades):
            order_id = trade_info['order_id']
            asset = trade_info['asset']

            is_closed, pnl = self.connector.check_win(order_id)

            if is_closed:
                if pnl > 0:
                    outcome = "GANHOU"
                    self.logger.info(f"✅ Trade {order_id} ({asset}) finalizada: {outcome} | Lucro: ${pnl:.2f}")
                else:
                    outcome = "PERDEU"
                    self.logger.info(f"❌ Trade {order_id} ({asset}) finalizada: {outcome} | Prejuízo: ${pnl:.2f}")

                # ✅ REGISTRAR O RESULTADO NO PERFORMANCE TRACKER
                self.performance_tracker.record_trade(trade_info, pnl)

                self.open_trades.remove(trade_info)
                self.multi_asset_manager.update_trade_count(asset, -1)

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
        """Loop principal de execução - ✅ CORREÇÃO PARA BACKTEST"""
        if not self.connect():
            self.logger.error("Falha na conexão, encerrando...")
            return

        self.running = True
        self.logger.info("Iniciando loop principal de trading...")
        self.list_available_assets()
        self.log_market_status()

        # LÓGICA DE EXECUÇÃO DIFERENTE PARA BACKTEST
        if self.mode == 'backtest':
            self.logger.info("Iniciando simulação de backtest...")
            initial_balance = self.connector.initial_balance

            # Loop de backtest que avança candle a candle
            while self.connector.tick():
                self.execute_trading_cycle()

            final_balance = self.connector.get_balance()
            self.logger.info("===== BACKTEST FINALIZADO =====")
            self.logger.info(f"Balanço Inicial: ${initial_balance:.2f}")
            self.logger.info(f"Balanço Final:   ${final_balance:.2f}")
            profit = final_balance - initial_balance
            profit_percent = (profit / initial_balance) * 100
            self.logger.info(f"Lucro/Prejuízo:  ${profit:.2f} ({profit_percent:.2f}%)")
            self.logger.info("=============================")

        # LÓGICA PARA TRADING REAL OU DEMO
        else:
            check_interval = self.config.get('execution', {}).get('check_interval', 10)
            try:
                while self.running:
                    start_time = time.time()
                    self.execute_trading_cycle()
                    elapsed = time.time() - start_time
                    sleep_time = max(1, check_interval - elapsed)
                    time.sleep(sleep_time)
            except KeyboardInterrupt:
                self.logger.info("Interrompido pelo usuário")
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
