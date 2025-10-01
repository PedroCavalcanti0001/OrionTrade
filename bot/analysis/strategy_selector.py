"""
Seletor de Estratégias Baseado no Regime
"""

import pandas as pd
from typing import Dict, Optional, Tuple
from .regime_classifier import MarketRegime, MarketRegimeClassifier


class StrategySelector:
    """Seleciona e executa estratégias baseadas no regime de mercado"""

    def __init__(self, config: Dict):
        self.config = config
        self.regime_classifier = MarketRegimeClassifier(config)
        self.analysis_config = config.get('analysis', {})

    def get_trading_signal(self, df: pd.DataFrame) -> Dict:
        """
        Gera sinal de trading baseado no regime e estratégia apropriada

        Returns:
            Dict com sinal e metadados
        """

        # Calcular features e classificar regime
        analysis = self.regime_classifier.get_detailed_analysis(df)
        regime = analysis['regime']

        signal = {
            'action': 'WAIT',
            'direction': None,
            'confidence': 0.0,
            'regime': regime,
            'analysis': analysis,
            'strategy_used': None
        }

        # Não operar em mercados perigosos
        if regime == MarketRegime.CHOPPY:
            signal['action'] = 'WAIT'
            signal['strategy_used'] = 'SAFETY_FILTER'
            return signal

        # Aplicar estratégia baseada no regime
        if regime == MarketRegime.UPTREND:
            return self._trend_following_strategy(df, 'LONG', analysis)
        elif regime == MarketRegime.DOWNTREND:
            return self._trend_following_strategy(df, 'SHORT', analysis)
        elif regime == MarketRegime.RANGING:
            return self._mean_reversion_strategy(df, analysis)
        elif regime == MarketRegime.SQUEEZE:
            return self._breakout_watch_strategy(df, analysis)

        return signal

    def _trend_following_strategy(self, df: pd.DataFrame, trend_direction: str,
                                  analysis: Dict) -> Dict:
        """Estratégia de seguir tendência com entrada em pullbacks"""

        df_with_features = self.regime_classifier.calculate_features(df)
        current = df_with_features.iloc[-1]
        prev = df_with_features.iloc[-2]

        # EMA para identificar pullbacks
        ema_fast = self.analysis_config.get('ema_fast', 9)
        ema_slow = self.analysis_config.get('ema_slow', 21)

        signal = {
            'action': 'WAIT',
            'direction': trend_direction,
            'confidence': 0.0,
            'regime': analysis['regime'],
            'analysis': analysis,
            'strategy_used': 'TREND_FOLLOWING'
        }

        if trend_direction == 'LONG':
            # Entrar em pullback: preço abaixo da EMA rápida mas acima da lenta
            pullback_condition = (
                    current['close'] < current['ema_fast'] and
                    current['close'] > current['ema_slow'] and
                    current['close'] > current['bb_middle']
            )

            # Confirmação: RSI saindo de sobrevenda
            rsi_confirmation = current['rsi'] > 35 and prev['rsi'] <= 35

            if pullback_condition and rsi_confirmation:
                signal['action'] = 'ENTER'
                signal['confidence'] = min(0.8, analysis['adx'] / 50)

        else:  # SHORT
            # Entrar em pullback: preço acima da EMA rápida mas abaixo da lenta
            pullback_condition = (
                    current['close'] > current['ema_fast'] and
                    current['close'] < current['ema_slow'] and
                    current['close'] < current['bb_middle']
            )

            # Confirmação: RSI saindo de sobrecompra
            rsi_confirmation = current['rsi'] < 65 and prev['rsi'] >= 65

            if pullback_condition and rsi_confirmation:
                signal['action'] = 'ENTER'
                signal['confidence'] = min(0.8, analysis['adx'] / 50)

        return signal

    def _mean_reversion_strategy(self, df: pd.DataFrame, analysis: Dict) -> Dict:
        """Estratégia de reversão à média em mercado lateral"""

        df_with_features = self.regime_classifier.calculate_features(df)
        current = df_with_features.iloc[-1]
        prev = df_with_features.iloc[-2]

        signal = {
            'action': 'WAIT',
            'direction': None,
            'confidence': 0.0,
            'regime': analysis['regime'],
            'analysis': analysis,
            'strategy_used': 'MEAN_REVERSION'
        }

        # Toque na banda superior + RSI sobrecomprado -> SHORT
        if (current['close'] >= current['bb_upper'] * 0.995 or
                current['rsi'] > 70):
            signal['action'] = 'ENTER'
            signal['direction'] = 'SHORT'
            signal['confidence'] = 0.7

        # Toque na banda inferior + RSI sobrevendido -> LONG
        elif (current['close'] <= current['bb_lower'] * 1.005 or
              current['rsi'] < 30):
            signal['action'] = 'ENTER'
            signal['direction'] = 'LONG'
            signal['confidence'] = 0.7

        return signal

    def _breakout_watch_strategy(self, df: pd.DataFrame, analysis: Dict) -> Dict:
        """Estratégia de observação para rompimentos (não opera, apenas monitora)"""

        signal = {
            'action': 'WAIT',
            'direction': None,
            'confidence': 0.0,
            'regime': analysis['regime'],
            'analysis': analysis,
            'strategy_used': 'BREAKOUT_WATCH'
        }

        # Em SQUEEZE, apenas observamos e registramos
        # Em uma implementação futura, poderíamos entrar no rompimento
        signal['action'] = 'OBSERVE'

        return signal