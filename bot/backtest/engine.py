# file: bot/backtest/engine.py
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bot.analysis.regime_classifier import MarketRegimeClassifier, MarketRegime
from bot.analysis.strategies import get_mean_reversion_signals, get_trend_following_signals


class BacktestEngine:
    """Motor de backtest vetorizado com análise de performance e visualização gráfica."""

    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        self.trading_config = config.get('trading', {})
        self.backtest_config = config.get('backtest', {})
        self.assets = self.backtest_config.get('assets', [])
        self.timeframe = self.trading_config.get('timeframe', 1)
        self.regime_classifier = MarketRegimeClassifier(config, logger)

    def __init__(self, config: dict, logger, api_connector=None):
        self.config = config
        self.logger = logger
        self.api_connector = api_connector  # Conector real para downloads

        self.trading_config = config.get('trading', {})
        self.backtest_config = config.get('backtest', {})
        self.assets = self.backtest_config.get('assets', [])
        self.timeframe = self.trading_config.get('timeframe', 1)
        self.data_dir = 'historical_data'

        self.regime_classifier = MarketRegimeClassifier(config, logger)

    def run(self):
        """Executa o processo de backtest para todos os ativos configurados."""
        self.logger.info("🚀 Iniciando Motor de Backtest Vetorizado com Análise Avançada...")

        overall_results = []
        initial_balance = self.trading_config.get('initial_balance', 1000.0)

        for asset in self.assets:
            self.logger.info(f"--- Processando backtest para o ativo: {asset} ---")

            df = self._load_data(asset)  # A lógica de download agora está aqui
            if df is None or df.empty:
                continue

            df = self._calculate_all_features(df)
            df['signal'] = self._generate_all_signals(df)

            trades, metrics = self._simulate_trades(df)
            overall_results.extend(trades)

            if trades:
                self._log_metrics(asset, metrics)
                self._plot_results(asset, df, trades)
            else:
                self.logger.warning(f"Nenhuma trade executada para {asset} no período analisado.")

        self._show_summary(initial_balance, overall_results)

    def _load_data(self, asset: str) -> pd.DataFrame:
        """Verifica, baixa (se necessário) e carrega os dados históricos, garantindo que são suficientes."""
        os.makedirs(self.data_dir, exist_ok=True)
        required_days = self.backtest_config.get('days_to_download', 30)
        file_path = os.path.join(self.data_dir, f"{asset}_M{self.timeframe}.csv")
        file_is_sufficient = False

        if os.path.exists(file_path):
            try:
                df_check = pd.read_csv(file_path)
                # A coluna pode se chamar 'from' ou 'timestamp'
                ts_col = 'timestamp' if 'timestamp' in df_check.columns else 'from'

                if not df_check.empty and ts_col in df_check.columns:
                    timestamps = pd.to_datetime(df_check[ts_col], unit='s')
                    data_duration_days = (timestamps.max() - timestamps.min()).days

                    if data_duration_days >= (required_days - 2):  # Tolerância de 2 dias
                        self.logger.info(
                            f"Arquivo de dados '{file_path}' encontrado e é suficiente ({data_duration_days} dias).")
                        file_is_sufficient = True
                    else:
                        self.logger.warning(
                            f"Arquivo '{file_path}' encontrado, mas é insuficiente ({data_duration_days}/{required_days} dias). Baixando novamente.")
            except Exception as e:
                self.logger.error(
                    f"Não foi possível ler o arquivo de dados existente '{file_path}': {e}. Baixando novamente.")

        if not file_is_sufficient:
            if not self._fetch_data_for_asset(asset, file_path, required_days):
                return None  # Falha no download

        # Carregar o arquivo final
        df = pd.read_csv(file_path, parse_dates=True)
        self.logger.info(f"Dados para '{asset}' carregados: {len(df)} candles.")
        return df

    def _fetch_data_for_asset(self, asset: str, file_path: str, days_to_download: int) -> bool:
        """Usa o conector real para baixar e salvar os dados de um ativo."""
        if not self.api_connector or not self.api_connector.connected:
            self.logger.error("Conector real da API não está disponível ou conectado para download de dados.")
            return False

        try:
            timeframe_seconds = self.timeframe * 60
            candles_per_call = 1000
            total_candles_needed = (days_to_download * 24 * 60) / self.timeframe
            num_calls = int(total_candles_needed / candles_per_call) + 1

            self.logger.info(f"Iniciando download de {days_to_download} dias de dados para {asset}...")

            end_time = time.time()
            all_candles = []

            for i in range(num_calls):
                current_period_str = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d')
                self.logger.info(
                    f"[Download {asset}] Buscando lote {i + 1}/{num_calls}... (Período a partir de ~{current_period_str})")

                candles = self.api_connector.api.get_candles(asset, timeframe_seconds, candles_per_call, end_time)

                if not candles: break

                new_candles = [c for c in candles if c['from'] not in [ac.get('from') for ac in all_candles]]
                all_candles.extend(new_candles)

                if not new_candles: break

                end_time = candles[0]['from']
                time.sleep(1.1)

            if not all_candles:
                self.logger.error(f"Nenhum dado foi baixado para {asset}.")
                return False

            df = pd.DataFrame(all_candles).drop_duplicates(subset=['from']).sort_values(by='from', ascending=True)
            df.to_csv(file_path, index=False)

            self.logger.info(f"Download para {asset} concluído! {len(df)} candles salvos em '{file_path}'.")
            return True
        except Exception as e:
            self.logger.error(f"Falha no download para {asset}: {e}", exc_info=True)
            return False

    def _calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Roda o cálculo de features uma única vez sobre todo o dataset."""
        self.logger.info("Calculando todos os indicadores técnicos...")
        df_with_features = self.regime_classifier.calculate_features(df)

        self.logger.info("Classificando regime de mercado para todo o período...")
        # Criar uma cópia da janela de dados para cada 'apply' para evitar erros de indexação
        df_copy = df_with_features.copy()
        df_with_features['regime'] = df_copy.apply(
            lambda row: self.regime_classifier.classify_regime(df_copy.loc[:row.name]),
            axis=1
        )
        return df_with_features

    def _generate_all_signals(self, df: pd.DataFrame) -> pd.Series:
        """Gera sinais de trading usando as funções de estratégia centralizadas."""
        self.logger.info("Gerando sinais de trading usando módulo centralizado...")

        # 1. Calcula os sinais para cada estratégia, para o dataset inteiro
        trend_signals = get_trend_following_signals(df)
        reversion_signals = get_mean_reversion_signals(df)

        # 2. Define onde cada estratégia deve ser aplicada, com base no regime
        conditions = [
            (df['regime'] == MarketRegime.RANGING) & (reversion_signals != 'WAIT'),
            (df['regime'] == MarketRegime.UPTREND) & (trend_signals == 'ENTER_LONG'),
            (df['regime'] == MarketRegime.DOWNTREND) & (trend_signals == 'ENTER_SHORT')
        ]

        # 3. Define qual o resultado para cada condição
        choices = [
            reversion_signals,  # Se for RANGING, usa o sinal de reversão
            trend_signals,  # Se for UPTREND, usa o sinal de tendência
            trend_signals  # Se for DOWNTREND, usa o sinal de tendência
        ]

        # np.select escolhe o sinal correto para cada candle baseado no seu regime
        return np.select(conditions, choices, default='WAIT')

    def _simulate_trades(self, df: pd.DataFrame):
        """Simula as trades e calcula as métricas de performance por estratégia."""
        self.logger.info("Simulando execuções de trades e calculando métricas...")

        trades = []
        balance = self.trading_config.get('initial_balance', 1000.0)
        risk_amount = balance * (self.trading_config.get('risk_per_trade', 1) / 100)
        payout = 0.87
        balance_history = [balance]

        for i in range(len(df) - 1):
            signal = df['signal'].iloc[i]
            regime = df['regime'].iloc[i]

            strategy_used = 'UNKNOWN'
            if signal != 'WAIT':
                if regime in [MarketRegime.UPTREND, MarketRegime.DOWNTREND]:
                    strategy_used = 'TREND_FOLLOWING'
                elif regime == MarketRegime.RANGING:
                    strategy_used = 'MEAN_REVERSION'

            if signal in ['ENTER_LONG', 'ENTER_SHORT']:
                entry_price = df['close'].iloc[i]
                result_price = df['close'].iloc[i + 1]
                pnl = -risk_amount

                if (signal == 'ENTER_LONG' and result_price > entry_price) or \
                        (signal == 'ENTER_SHORT' and result_price < entry_price):
                    pnl = risk_amount * payout

                balance += pnl
                balance_history.append(balance)
                trades.append({
                    'index': i,
                    'entry_time': df['timestamp'].iloc[i],
                    'direction': 'LONG' if 'LONG' in signal else 'SHORT',
                    'pnl': pnl,
                    'strategy': strategy_used  # ✅ Rastrear a estratégia usada
                })

        if not trades:
            return [], {}

        # --- Cálculo das Métricas (Agora separado por estratégia) ---
        metrics = {'OVERALL': self._calculate_metrics_for_trades(trades, balance_history)}

        trend_trades = [t for t in trades if t['strategy'] == 'TREND_FOLLOWING']
        reversion_trades = [t for t in trades if t['strategy'] == 'MEAN_REVERSION']

        if trend_trades:
            metrics['TREND_FOLLOWING'] = self._calculate_metrics_for_trades(trend_trades)
        if reversion_trades:
            metrics['MEAN_REVERSION'] = self._calculate_metrics_for_trades(reversion_trades)

        return trades, metrics

    # Adicione este novo método à classe BacktestEngine
    def _calculate_metrics_for_trades(self, trades: list, balance_history: list = None):
        """Método auxiliar para calcular métricas para uma lista de trades."""
        pnl_values = [t['pnl'] for t in trades]
        gross_profit = sum(pnl for pnl in pnl_values if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnl_values if pnl < 0))

        max_drawdown = 0
        if balance_history:
            peak = balance_history[0]
            for balance_value in balance_history:
                if balance_value > peak:
                    peak = balance_value
                drawdown = (peak - balance_value) / peak if peak > 0 else 0
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

        return {
            "total_pnl": sum(pnl_values),
            "total_trades": len(trades),
            "win_rate": (sum(1 for pnl in pnl_values if pnl > 0) / len(trades)) if trades else 0,
            "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else float('inf'),
            "max_drawdown": max_drawdown
        }

        # Substitua o método _log_metrics

    def _log_metrics(self, asset, all_metrics):
        """Loga as métricas de performance detalhadas por estratégia."""
        self.logger.info(f"📊 Métricas para {asset}:")
        for strategy_name, metrics in all_metrics.items():
            drawdown_log = f"| Drawdown Máximo: {metrics['max_drawdown']:.2%}" if strategy_name == 'OVERALL' else ""

            # ✅ CORREÇÃO: A formatação da taxa de acerto foi ajustada para ser compatível.
            win_rate_str = f"{metrics['win_rate']:.2%}"

            self.logger.info(
                f"  📈 Estratégia: {strategy_name:<16} "
                f"| Trades: {metrics['total_trades']:<4} "
                f"| Acerto: {win_rate_str:<8} "
                f"| P/L: ${metrics['total_pnl']:<8.2f} "
                f"{drawdown_log}"
            )

    def _plot_results(self, asset, df, trades):
        """Gera um gráfico HTML interativo com os resultados do backtest."""
        self.logger.info(f"📊 Gerando gráfico de resultados para {asset}...")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=(f'Backtest para {asset}', 'RSI'), row_heights=[0.7, 0.3])

        # Gráfico de Preço
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['close'], name='Preço', line=dict(color='gray')), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['bb_upper'], name='BB Upper', line=dict(color='lightgray', dash='dash')),
            row=1, col=1)
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['bb_lower'], name='BB Lower', line=dict(color='lightgray', dash='dash')),
            row=1, col=1)
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['ema_fast'], name='EMA Rápida', line=dict(color='cyan', width=1)), row=1,
            col=1)
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['ema_slow'], name='EMA Lenta', line=dict(color='orange', width=1)),
            row=1, col=1)

        # Adicionar marcadores de trade
        long_trades = [t for t in trades if t['direction'] == 'LONG']
        short_trades = [t for t in trades if t['direction'] == 'SHORT']

        fig.add_trace(go.Scatter(
            x=[df['timestamp'].iloc[t['index']] for t in long_trades],
            y=[df['low'].iloc[t['index']] * 0.999 for t in long_trades],
            mode='markers', name='Compra',
            marker=dict(color='green', symbol='triangle-up', size=8)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[df['timestamp'].iloc[t['index']] for t in short_trades],
            y=[df['high'].iloc[t['index']] * 1.001 for t in short_trades],
            mode='markers', name='Venda',
            marker=dict(color='red', symbol='triangle-down', size=8)
        ), row=1, col=1)

        # Gráfico do RSI
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi'], name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        fig.update_layout(title_text=f'Análise de Backtest para {asset}', xaxis_rangeslider_visible=False)

        # Salvar o gráfico
        os.makedirs('backtest_results', exist_ok=True)
        file_path = f'backtest_results/{asset}_backtest.html'
        fig.write_html(file_path)
        self.logger.info(f"Gráfico salvo em: {file_path}")

    def _show_summary(self, initial_balance, all_trades):
        """Mostra o resumo final do backtest."""
        total_pnl = sum(t['pnl'] for t in all_trades)
        final_balance = initial_balance + total_pnl
        profit_percent = (total_pnl / initial_balance) * 100 if initial_balance > 0 else 0

        self.logger.info("=" * 35)
        self.logger.info("🏁 BACKTEST VETORIZADO FINALIZADO 🏁")
        self.logger.info("=" * 35)
        self.logger.info(f"Balanço Inicial: ${initial_balance:,.2f}")
        self.logger.info(f"Balanço Final:   ${final_balance:,.2f}")
        self.logger.info(f"Lucro/Prejuízo Total: ${total_pnl:,.2f} ({profit_percent:.2f}%)")
        self.logger.info("=" * 35)
