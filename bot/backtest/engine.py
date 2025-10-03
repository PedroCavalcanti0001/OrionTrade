# file: bot/backtest/engine.py
import pandas as pd
import numpy as np
import os
from bot.analysis.regime_classifier import MarketRegimeClassifier, MarketRegime
from bot.analysis.strategy_selector import StrategySelector


class BacktestEngine:
    """Motor de backtest rápido e vetorizado."""

    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        self.trading_config = config.get('trading', {})
        self.backtest_config = config.get('backtest', {})
        self.assets = self.backtest_config.get('assets', [])
        self.timeframe = self.trading_config.get('timeframe', 1)

        # Instanciar os componentes de análise
        self.regime_classifier = MarketRegimeClassifier(config, logger)
        self.strategy_selector = StrategySelector(config, logger)

    def run(self):
        """Executa o processo de backtest para todos os ativos configurados."""
        self.logger.info("🚀 Iniciando Motor de Backtest Vetorizado...")

        total_pnl = 0
        initial_balance = self.trading_config.get('initial_balance', 1000.0)
        final_balance = initial_balance

        for asset in self.assets:
            self.logger.info(f"--- Processando backtest para o ativo: {asset} ---")

            # 1. Carregar Dados Históricos
            df = self._load_data(asset)
            if df.empty:
                continue

            # 2. Calcular Features e Regimes (VETORIZADO)
            df = self._calculate_all_features(df)

            # 3. Gerar Sinais de Trading (VETORIZADO)
            df['signal'] = self._generate_all_signals(df)

            # 4. Simular Trades e Calcular Resultados
            asset_pnl, num_trades, win_rate = self._simulate_trades(df, initial_balance)

            if num_trades > 0:
                self.logger.info(f"Resultado para {asset}:")
                self.logger.info(f"  Trades Totais: {num_trades}")
                self.logger.info(f"  Taxa de Acerto: {win_rate:.2%}")
                self.logger.info(f"  Lucro/Prejuízo: ${asset_pnl:.2f}")
                final_balance += asset_pnl
            else:
                self.logger.warning(f"Nenhuma trade executada para {asset} no período analisado.")

        # 5. Mostrar Resumo Final
        self._show_summary(initial_balance, final_balance)

    def _load_data(self, asset: str) -> pd.DataFrame:
        """Carrega os dados históricos de um arquivo CSV."""
        file_path = f"historical_data/{asset}_M{self.timeframe}.csv"
        if not os.path.exists(file_path):
            self.logger.error(f"Arquivo de dados '{file_path}' não encontrado para o backtest.")
            self.logger.error("Execute o bot em modo 'backtest' uma vez com o MockConnector para baixar os dados.")
            return pd.DataFrame()

        df = pd.read_csv(file_path)
        self.logger.info(f"Dados para '{asset}' carregados: {len(df)} candles.")
        return df

    def _calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Roda o cálculo de features uma única vez sobre todo o dataset."""
        self.logger.info("Calculando todos os indicadores técnicos de uma só vez...")
        # A função calculate_features já opera sobre o DataFrame inteiro
        df_with_features = self.regime_classifier.calculate_features(df)

        # Classificar o regime para cada candle
        # Usamos .apply, que é mais rápido que um loop em Python
        self.logger.info("Classificando regime de mercado para todo o período...")
        df_with_features['regime'] = df_with_features.apply(
            lambda row: self.regime_classifier.classify_regime(df_with_features.loc[:row.name]),
            axis=1
        )
        return df_with_features

    def _generate_all_signals(self, df: pd.DataFrame) -> pd.Series:
        """Gera sinais de trading de forma vetorizada."""
        self.logger.info("Gerando sinais de trading para todo o período...")

        # Condições para cada estratégia (exemplo simplificado para vetorização)

        # Lógica de Confirmação (a vela fechou na direção da reversão)
        df['bullish_confirmation'] = df['close'] > df['open']
        df['bearish_confirmation'] = df['close'] < df['open']

        # Lógica de Pullback (RSI em zona neutra durante tendência)
        df['pullback_long'] = (df['ema_fast'] > df['ema_slow']) & (df['rsi'].between(40, 55))
        df['pullback_short'] = (df['ema_fast'] < df['ema_slow']) & (df['rsi'].between(45, 60))

        # Lógica de Reversão
        df['reversion_long'] = (df['close'] <= df['bb_lower']) & df['bullish_confirmation']
        df['reversion_short'] = (df['close'] >= df['bb_upper']) & df['bearish_confirmation']

        # Aplicar a lógica baseada no regime de cada candle
        def get_signal(row):
            if row['regime'] in [MarketRegime.UPTREND, MarketRegime.DOWNTREND]:
                if row['pullback_long']:
                    return 'ENTER_LONG'
                if row['pullback_short']:
                    return 'ENTER_SHORT'
            elif row['regime'] == MarketRegime.RANGING:
                if row['reversion_long']:
                    return 'ENTER_LONG'
                if row['reversion_short']:
                    return 'ENTER_SHORT'
            return 'WAIT'

        return df.apply(get_signal, axis=1)

    def _simulate_trades(self, df: pd.DataFrame, balance: float):
        """Simula as trades com base nos sinais e calcula o P/L."""
        self.logger.info("Simulando execuções de trades...")

        pnl_total = 0
        trades_count = 0
        wins = 0
        risk_amount = balance * (self.trading_config.get('risk_per_trade', 1) / 100)
        payout = 0.87  # Payout médio de 87%

        # Iterar sobre os candles onde um sinal de entrada foi gerado
        for i in range(len(df) - 1):  # -1 para garantir que o candle de resultado exista
            if df['signal'].iloc[i] == 'ENTER_LONG':
                trades_count += 1
                entry_price = df['close'].iloc[i]
                result_price = df['close'].iloc[i + 1]  # Expiração de 1 candle
                if result_price > entry_price:
                    pnl_total += risk_amount * payout
                    wins += 1
                else:
                    pnl_total -= risk_amount

            elif df['signal'].iloc[i] == 'ENTER_SHORT':
                trades_count += 1
                entry_price = df['close'].iloc[i]
                result_price = df['close'].iloc[i + 1]
                if result_price < entry_price:
                    pnl_total += risk_amount * payout
                    wins += 1
                else:
                    pnl_total -= risk_amount

        win_rate = (wins / trades_count) if trades_count > 0 else 0
        return pnl_total, trades_count, win_rate

    def _show_summary(self, initial_balance, final_balance):
        """Mostra o resumo final do backtest."""
        profit = final_balance - initial_balance
        profit_percent = (profit / initial_balance) * 100 if initial_balance > 0 else 0

        self.logger.info("=" * 35)
        self.logger.info("🏁 BACKTEST VETORIZADO FINALIZADO 🏁")
        self.logger.info("=" * 35)
        self.logger.info(f"Balanço Inicial: ${initial_balance:,.2f}")
        self.logger.info(f"Balanço Final:   ${final_balance:,.2f}")
        self.logger.info(f"Lucro/Prejuízo:  ${profit:,.2f} ({profit_percent:.2f}%)")
        self.logger.info("=" * 35)