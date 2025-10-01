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


class OrionTrader:
    """Classe principal que orquestra todo o sistema de trading"""

    def __init__(self, config: Dict, mode: str, logger):
        self.config = config
        self.mode = mode
        self.logger = logger

        # Inicializar componentes
        self._initialize_components()

        # INICIALIZAR VALIDADOR DE MERCADO ← ADICIONE ESTA LINHA
        self.market_validator = MarketHoursValidator(self.connector, self.logger)

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

        # Inicializar outros componentes
        self.strategy_selector = StrategySelector(self.config)
        self.risk_manager = RiskManager(self.config)

    def connect(self) -> bool:
        """Estabelece conexão com a plataforma"""
        self.logger.info("Conectando à plataforma...")
        return self.connector.connect()

    def get_market_data(self) -> pd.DataFrame:
        """Obtém e formata dados de mercado"""
        asset = self.config['trading']['asset']
        timeframe = self.config['trading']['timeframe']

        # Obter candles suficientes para análise
        candles = self.connector.get_candles(asset, timeframe, 100)

        if not candles:
            self.logger.warning("Nenhum dado de mercado obtido")
            return pd.DataFrame()

        # Converter para DataFrame
        df = pd.DataFrame(candles)

        # VERIFICAÇÃO CRÍTICA: Log da estrutura dos dados
        self.logger.debug(f"Colunas recebidas: {df.columns.tolist()}")
        self.logger.debug(f"Primeira linha: {df.iloc[0] if len(df) > 0 else 'DF vazio'}")

        # Garantir que as colunas necessárias existem
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            self.logger.error(f"Colunas faltando: {missing_columns}")
            self.logger.error(f"Colunas disponíveis: {df.columns.tolist()}")
            return pd.DataFrame()

        # Renomear colunas se necessário (caso a API use nomes diferentes)
        column_mapping = {
            'from': 'timestamp',
            'min': 'low',
            'max': 'high'
        }

        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df[new_name] = df[old_name]

        # Criar timestamp se necessário
        if 'timestamp' not in df.columns and 'from' in df.columns:
            df['timestamp'] = pd.to_datetime(df['from'], unit='s')
        elif 'timestamp' not in df.columns:
            df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1min')

        df.set_index('timestamp', inplace=True)

        return df

    def execute_trading_cycle(self):
        """Executa um ciclo completo de trading"""
        try:
            # 1. Obter dados de mercado
            df = self.get_market_data()
            if df.empty:
                self.logger.warning("Dados de mercado vazios, aguardando próximo ciclo")
                return

            # DEBUG TEMPORÁRIO - Mostrar estrutura dos dados
            if not df.empty:
                self.logger.info(f"Colunas no DataFrame: {df.columns.tolist()}")
                self.logger.info(f"Primeiras linhas:\\n{df.head(2)}")
            else:
                self.logger.warning("DataFrame vazio")
                return

            # 2. Atualizar saldo
            self.balance = self.connector.get_balance()

            # 3. Gerar sinal
            signal = self.strategy_selector.get_trading_signal(df)

            # 4. Log do diagnóstico
            self._log_market_analysis(signal)

            # 5. Verificar se deve entrar em trade
            if (signal['action'] == 'ENTER' and
                    self.risk_manager.should_enter_trade(
                        signal['regime'].value,
                        len(self.open_trades),
                        signal['confidence']
                    )):
                self._execute_trade(df, signal)

            # 6. Verificar trades abertas
            self._check_open_trades()

            # 7. Verificar trades abertas
            self._check_open_trades()

        except Exception as e:
            self.logger.error(f"Erro no ciclo de trading: {e}")

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

    def run(self):
        """Loop principal de execução"""
        if not self.connect():
            self.logger.error("Falha na conexão, encerrando...")
            return

        self.running = True
        self.logger.info("Iniciando loop principal de trading...")

        # LOG INICIAL DO STATUS DO MERCADO ← ADICIONE ESTE BLOCO
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

        self.running = True
        self.logger.info("Iniciando loop principal de trading...")

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
        """Log do status de abertura dos ativos"""
        try:
            asset = self.config['trading']['asset']
            is_open = self.market_validator.is_asset_open(asset)
            schedule = self.market_validator.get_asset_schedule(asset)

            status = "ABERTO" if is_open else "FECHADO"
            self.logger.info(f"Status do mercado - Ativo: {asset}, Status: {status}")

            if schedule:
                self.logger.debug(f"Detalhes do horário: {schedule}")

        except Exception as e:
            self.logger.error(f"Erro ao verificar status do mercado: {e}")

    def shutdown(self):
        """Encerra o trader de forma segura"""
        self.logger.info("Encerrando OrionTrader...")
        self.running = False
        self.connector.disconnect()
        self.logger.info("OrionTrader encerrado com sucesso")