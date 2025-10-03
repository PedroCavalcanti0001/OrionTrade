"""
Gerenciamento de Risco Avançado
"""

import pandas as pd
import pandas_ta as ta
from typing import Dict, Tuple
from bot.analysis.regime_classifier import MarketRegimeClassifier


class RiskManager:
    """Gerencia risco, posição, stop loss e take profit"""

    def __init__(self, config: Dict, logger):
        self.logger = logger
        self.config = config
        self.trading_config = config.get('trading', {})
        self.regime_classifier = MarketRegimeClassifier(config, self.logger)

    def calculate_position_size(self, balance: float, signal_confidence: float,
                                performance_modifier: float = 1.0) -> float:
        """Calcula tamanho da posição com base no risco, confiança e modificador de desempenho - ✅ VERSÃO DE ELITE"""
        risk_per_trade = self.trading_config.get('risk_per_trade', 1.0)
        base_risk_amount = balance * (risk_per_trade / 100)

        # Ajustar risco baseado na confiança do sinal
        # Sinais de alta confiança arriscam mais, até o limite base
        confidence_factor = 0.5 + (signal_confidence / 2)  # Mapeia confiança [0,1] para [0.5, 1]
        adjusted_risk = base_risk_amount * confidence_factor

        # ✅ APLICAR MODIFICADOR DE PERFORMANCE DINÂMICO
        final_risk_amount = adjusted_risk * performance_modifier

        # Limitar posição máxima (não mais que 5% do balanço, por segurança)
        max_position = balance * 0.05
        position_size = min(final_risk_amount, max_position)

        # Garantir tamanho mínimo
        min_position = 1.0
        position_size = max(position_size, min_position)

        return round(position_size, 2)

    def calculate_stop_loss_take_profit(self, df: pd.DataFrame, direction: str,
                                        entry_price: float) -> Tuple[float, float]:
        """Calcula níveis dinâmicos de SL e TP baseado no ATR"""

        df_with_features = self.regime_classifier.calculate_features(df)
        current_atr = df_with_features['atr'].iloc[-1]

        sl_multiplier = self.trading_config.get('stop_loss_atr_multiplier', 1.5)
        tp_multiplier = self.trading_config.get('take_profit_atr_multiplier', 2.0)

        if direction == 'LONG':
            stop_loss = entry_price - (current_atr * sl_multiplier)
            take_profit = entry_price + (current_atr * tp_multiplier)
        else:  # SHORT
            stop_loss = entry_price + (current_atr * sl_multiplier)
            take_profit = entry_price - (current_atr * tp_multiplier)

        return round(stop_loss, 5), round(take_profit, 5)

    def should_enter_trade(self, current_regime, open_trades: int, signal_confidence: float) -> bool:
        """Decide se deve entrar em nova trade baseado em múltiplos fatores"""

        max_trades = self.trading_config.get('max_open_trades', 3)

        # Não entrar se limite de trades abertas atingido
        if open_trades >= max_trades:
            return False

        # Não entrar em mercados CHOPPY
        if current_regime == 'CHOPPY':
            return False

        # Confiança mínima do sinal
        if signal_confidence < 0.5:
            return False

        # Adicionar mais filtros de risco aqui se necessário

        return True

    def get_risk_metrics(self, df: pd.DataFrame, balance: float) -> Dict:
        """Retorna métricas de risco atuais"""

        df_with_features = self.regime_classifier.calculate_features(df)
        current = df_with_features.iloc[-1]

        volatility = (current['atr'] / current['close']) * 100
        regime_risk = {
            'UPTREND': 'MEDIUM',
            'DOWNTREND': 'MEDIUM',
            'RANGING': 'LOW',
            'SQUEEZE': 'HIGH',  # Risco alto devido à imprevisibilidade do rompimento
            'CHOPPY': 'VERY_HIGH'
        }

        analysis = self.regime_classifier.get_detailed_analysis(df)
        current_regime = analysis['regime'].value

        return {
            'current_volatility': volatility,
            'regime_risk_level': regime_risk.get(current_regime, 'UNKNOWN'),
            'atr_value': current['atr'],
            'recommended_position_size': self.calculate_position_size(balance, 0.7),
            'market_condition': 'Favorable' if current_regime in ['UPTREND', 'DOWNTREND', 'RANGING'] else 'Unfavorable'
        }