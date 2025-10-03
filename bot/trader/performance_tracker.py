# file: bot/trader/performance_tracker.py
"""
Módulo de Rastreamento de Performance para o OrionTrader
"""
import pandas as pd
import os
from typing import Dict, List


class PerformanceTracker:
    """Registra e analisa o desempenho das estratégias de trading."""

    def __init__(self, logger, history_file='logs/trade_history.csv'):
        self.logger = logger
        self.history_file = history_file
        self.trade_history = self._load_history()

    def _load_history(self) -> pd.DataFrame:
        """Carrega o histórico de trades de um arquivo CSV."""
        if os.path.exists(self.history_file):
            self.logger.info(f"Carregando histórico de trades de {self.history_file}")
            return pd.read_csv(self.history_file, index_col=0, parse_dates=['timestamp'])
        return pd.DataFrame(columns=[
            'timestamp', 'asset', 'strategy', 'regime', 'direction', 'confidence', 'pnl'
        ])

    def _save_history(self):
        """Salva o histórico de trades no arquivo CSV."""
        try:
            self.trade_history.to_csv(self.history_file)
        except Exception as e:
            self.logger.error(f"Erro ao salvar histórico de trades: {e}")

    def record_trade(self, trade_info: Dict, pnl: float):
        """
        Registra uma trade finalizada no histórico.

        Args:
            trade_info (Dict): O dicionário da trade, vindo de self.open_trades.
            pnl (float): O lucro ou prejuízo da trade.
        """
        try:
            signal = trade_info.get('signal', {})
            new_record = {
                'timestamp': pd.Timestamp.now(),
                'asset': trade_info.get('asset'),
                'strategy': signal.get('strategy_used', 'UNKNOWN'),
                'regime': signal.get('regime', 'UNKNOWN').value if hasattr(signal.get('regime'), 'value') else str(
                    signal.get('regime')),
                'direction': trade_info.get('direction'),
                'confidence': signal.get('confidence', 0.0),
                'pnl': pnl
            }

            # Usar pd.concat para adicionar a nova linha
            new_record_df = pd.DataFrame([new_record])
            self.trade_history = pd.concat([self.trade_history, new_record_df], ignore_index=True)

            self.logger.debug(f"Trade registrada: {new_record}")
            self._save_history()
        except Exception as e:
            self.logger.error(f"Erro ao registrar trade: {e}")

    def get_summary(self, last_n_trades: int = 50) -> Dict:
        """
        Gera um resumo do desempenho por estratégia.

        Args:
            last_n_trades (int): O número de trades recentes a serem analisadas.

        Returns:
            Dict: Um dicionário onde as chaves são as estratégias e os valores
                  são dicionários com 'pnl_total' e 'count'.
        """
        if self.trade_history.empty:
            return {}

        recent_trades = self.trade_history.tail(last_n_trades)
        summary = {}

        if not recent_trades.empty:
            grouped = recent_trades.groupby('strategy')['pnl'].agg(['sum', 'count'])
            for strategy, data in grouped.iterrows():
                summary[strategy] = {
                    'pnl_total': data['sum'],
                    'count': data['count']
                }

        return summary