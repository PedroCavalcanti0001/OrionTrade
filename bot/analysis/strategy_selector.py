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

    def debug_confidence_calculation(self, df: pd.DataFrame, signal: Dict):
        """Debug do cálculo de confiança"""
        current = df.iloc[-1]

        print(f"=== DEBUG CONFIANÇA ===")
        print(f"Regime: {signal['regime'].value}")
        print(f"Ação: {signal['action']}")
        print(f"Close: {current['close']:.5f}")
        print(f"BB Upper: {current.get('bb_upper', 0):.5f}")
        print(f"BB Lower: {current.get('bb_lower', 0):.5f}")
        print(f"ADX: {current.get('adx', 0):.2f}")
        print(f"RSI: {current.get('rsi', 0):.2f}")
        print(f"EMA Fast: {current.get('ema_fast', 0):.5f}")
        print(f"EMA Slow: {current.get('ema_slow', 0):.5f}")
        print(f"Volume: {current.get('volume', 0)}")
        print("========================")

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

        # DEBUG TEMPORÁRIO
        self.debug_confidence_calculation(df, signal)

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

    def _trend_following_strategy(self, df: pd.DataFrame, trend_direction: str, analysis: Dict) -> Dict:
        """Estratégia de seguir tendência - COM CONFIANÇA MÍNIMA"""

        df_with_features = self.regime_classifier.calculate_features(df)
        current = df_with_features.iloc[-1]
        prev = df_with_features.iloc[-2]

        signal = {
            'action': 'WAIT',
            'direction': trend_direction,
            'confidence': 0.3,  # CONFIANÇA BASE
            'regime': analysis['regime'],
            'analysis': analysis,
            'strategy_used': 'TREND_FOLLOWING'
        }

        if trend_direction == 'LONG':
            # Condições para LONG
            conditions = [
                current['close'] > current['ema_slow'],
                current['plus_di'] > current['minus_di'],
                current['rsi'] < 70,  # Não sobrecomprado
                current['adx'] > 15  # Alguma tendência
            ]

            if all(conditions):
                # Calcular confiança baseada em múltiplos fatores
                confidence_factors = []

                # Força da tendência
                trend_strength = min(current['adx'] / 30.0, 1.0)
                confidence_factors.append(trend_strength)

                # Posição nas EMAs
                ema_distance = (current['close'] - current['ema_slow']) / current['ema_slow']
                ema_score = min(abs(ema_distance) * 100, 1.0)
                confidence_factors.append(ema_score)

                # RSI favorável
                rsi_score = 1.0 if current['rsi'] < 60 else 0.7 if current['rsi'] < 70 else 0.4
                confidence_factors.append(rsi_score)

                # Confiança final
                confidence = sum(confidence_factors) / len(confidence_factors)
                signal['confidence'] = max(confidence, 0.5)  # MÍNIMO 50%
                signal['action'] = 'ENTER'

        # Similar para SHORT...

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
        """Estratégia completa de rompimento para mercado em SQUEEZE"""

        df_with_features = self.regime_classifier.calculate_features(df)

        if len(df_with_features) < 10:
            return self._get_breakout_wait_signal(analysis)

        current = df_with_features.iloc[-1]

        signal = {
            'action': 'OBSERVE',
            'direction': None,
            'confidence': 0.0,
            'regime': analysis['regime'],
            'analysis': analysis,
            'strategy_used': 'BREAKOUT',
            'breakout_type': 'WATCHING',
            'risk_reward_ratio': 0.0
        }

        # 1. VERIFICAR SE AINDA ESTÁ EM SQUEEZE
        if current['bb_width'] >= self.analysis_config.get('bb_squeeze_threshold', 0.1):
            signal['breakout_type'] = 'SQUEEZE_ENDED'
            return signal

        # 2. DETECTAR ROMPIMENTO
        breakout_detected, breakout_direction = self._detect_breakout(df_with_features)

        if breakout_detected:
            # 3. CONFIRMAR ROMPIMENTO
            if self._confirm_breakout(df_with_features, breakout_direction, 2):
                # 4. CALCULAR CONFIANÇA E RISCO/RECOMPENSA
                confidence = self._calculate_breakout_confidence(df_with_features, breakout_direction)
                risk_reward = self._calculate_breakout_risk_reward(df_with_features, breakout_direction)

                # 5. FILTRO DE RISCO/RECOMPENSA (mínimo 1:1.5)
                if risk_reward >= 1.5 and confidence >= 0.6:
                    signal.update({
                        'action': 'ENTER',
                        'direction': breakout_direction,
                        'confidence': confidence,
                        'breakout_type': 'CONFIRMED_HIGH_CONFIDENCE',
                        'risk_reward_ratio': risk_reward,
                        'stop_loss': self._calculate_breakout_stop_loss(df_with_features, breakout_direction),
                        'take_profit': self._calculate_breakout_take_profit(df_with_features, breakout_direction)
                    })
                else:
                    signal.update({
                        'breakout_type': 'CONFIRMED_LOW_RR',
                        'risk_reward_ratio': risk_reward,
                        'confidence': confidence
                    })
            else:
                signal['breakout_type'] = 'PENDING_CONFIRMATION'

        return signal

    def _detect_breakout(self, df: pd.DataFrame) -> tuple:
        """Detecta se houve rompimento das bandas de Bollinger"""
        if len(df) < 3:
            return False, None

        current = df.iloc[-1]
        prev = df.iloc[-2]

        # Rompimento para CIMA
        upside_breakout = (
                current['close'] > current['bb_upper'] and
                current['close'] > prev['bb_upper'] and  # Confirmação
                current['volume'] > prev['volume'] * 1.2  # Volume crescente
        )

        # Rompimento para BAIXO
        downside_breakout = (
                current['close'] < current['bb_lower'] and
                current['close'] < prev['bb_lower'] and  # Confirmação
                current['volume'] > prev['volume'] * 1.2  # Volume crescente
        )

        if upside_breakout:
            return True, 'LONG'
        elif downside_breakout:
            return True, 'SHORT'
        else:
            return False, None

    def _confirm_breakout(self, df: pd.DataFrame, direction: str, confirmation_candles: int = 2) -> bool:
        """Confirma o rompimento com candles subsequentes"""
        if len(df) < confirmation_candles + 1:
            return False

        # Verificar os candles após o rompimento
        recent_data = df.tail(confirmation_candles + 1)

        if direction == 'LONG':
            # Confirmação de rompimento para cima: preço se mantém acima da banda média
            confirmation = all(
                candle['close'] > candle['bb_middle']
                for _, candle in recent_data.tail(confirmation_candles).iterrows()
            )
            # Volume consistente ou crescente
            volume_confirmation = recent_data['volume'].is_monotonic_increasing or \
                                  recent_data['volume'].mean() > recent_data['volume'].iloc[
                                      -confirmation_candles - 1] * 1.1

        else:  # SHORT
            # Confirmação de rompimento para baixo: preço se mantém abaixo da banda média
            confirmation = all(
                candle['close'] < candle['bb_middle']
                for _, candle in recent_data.tail(confirmation_candles).iterrows()
            )
            # Volume consistente ou crescente
            volume_confirmation = recent_data['volume'].is_monotonic_increasing or \
                                  recent_data['volume'].mean() > recent_data['volume'].iloc[
                                      -confirmation_candles - 1] * 1.1

        return confirmation and volume_confirmation

    def _calculate_breakout_confidence(self, df: pd.DataFrame, direction: str) -> float:
        """Calcula confiança do rompimento - VERSÃO CORRIGIDA"""
        if len(df) < 5:
            return 0.5  # Confiança base se dados insuficientes

        current = df.iloc[-1]

        try:
            factors = []

            # 1. Força do rompimento (0-1 ponto)
            if direction == 'LONG':
                strength = (current['close'] - current['bb_upper']) / current['bb_upper']
            else:
                strength = (current['bb_lower'] - current['close']) / current['bb_lower']

            strength_score = min(abs(strength) * 500, 1.0)  # Amplificar o efeito
            factors.append(strength_score)

            # 2. Volume relativo (0-1 ponto)
            volume_avg = df['volume'].tail(10).mean()
            volume_ratio = current['volume'] / volume_avg if volume_avg > 0 else 1
            volume_score = min(volume_ratio / 2.0, 1.0)
            factors.append(volume_score * 0.8)  # Peso menor para volume

            # 3. ADX - força da tendência (0-1 ponto)
            adx_score = min(current['adx'] / 40.0, 1.0)
            factors.append(adx_score)

            # 4. Alinhamento com tendência (0-1 ponto)
            if direction == 'LONG':
                trend_alignment = 1.0 if current['ema_fast'] > current['ema_slow'] else 0.3
            else:
                trend_alignment = 1.0 if current['ema_fast'] < current['ema_slow'] else 0.3
            factors.append(trend_alignment)

            # 5. RSI confirmation (0-1 ponto)
            rsi = current.get('rsi', 50)
            if direction == 'LONG':
                rsi_score = 1.0 if rsi < 70 else 0.5 if rsi < 80 else 0.2
            else:
                rsi_score = 1.0 if rsi > 30 else 0.5 if rsi > 20 else 0.2
            factors.append(rsi_score * 0.7)

            # Média ponderada dos fatores
            weights = [0.3, 0.15, 0.25, 0.2, 0.1]
            confidence = sum(f * w for f, w in zip(factors, weights))

            # Garantir confiança mínima em boas condições
            if strength_score > 0.3 and adx_score > 0.4:
                confidence = max(confidence, 0.6)

            return min(confidence, 0.95)

        except Exception as e:
            self.logger.error(f"Erro cálculo confiança: {e}")
            return 0.5  # Fallback

    def _execute_breakout_entry(self, df: pd.DataFrame, direction: str, confidence: float) -> Dict:
        """Executa a entrada no rompimento com gerenciamento de risco"""

        current = df.iloc[-1]

        signal = {
            'action': 'ENTER',
            'direction': direction,
            'confidence': confidence,
            'entry_price': current['close'],
            'stop_loss': self._calculate_breakout_stop_loss(df, direction),
            'take_profit': self._calculate_breakout_take_profit(df, direction),
            'risk_reward_ratio': self._calculate_breakout_risk_reward(df, direction)
        }

        return signal

    def _calculate_breakout_stop_loss(self, df: pd.DataFrame, direction: str) -> float:
        """Calcula stop loss para estratégia de rompimento"""
        current = df.iloc[-1]

        if direction == 'LONG':
            # Stop loss abaixo da banda inferior ou ATR-based
            stop_atr = current['close'] - (current['atr'] * 1.5)
            stop_bb = current['bb_lower'] - (current['atr'] * 0.5)
            return min(stop_atr, stop_bb)
        else:
            # Stop loss acima da banda superior ou ATR-based
            stop_atr = current['close'] + (current['atr'] * 1.5)
            stop_bb = current['bb_upper'] + (current['atr'] * 0.5)
            return max(stop_atr, stop_bb)

    def _calculate_breakout_take_profit(self, df: pd.DataFrame, direction: str) -> float:
        """Calcula take profit para estratégia de rompimento"""
        current = df.iloc[-1]

        if direction == 'LONG':
            # TP baseado na largura das bandas ou ATR
            tp_bb_width = current['close'] + (current['bb_width'] * current['close'] * 2)
            tp_atr = current['close'] + (current['atr'] * 3)
            return max(tp_bb_width, tp_atr)
        else:
            # TP baseado na largura das bandas ou ATR
            tp_bb_width = current['close'] - (current['bb_width'] * current['close'] * 2)
            tp_atr = current['close'] - (current['atr'] * 3)
            return min(tp_bb_width, tp_atr)

    def _calculate_breakout_risk_reward(self, df: pd.DataFrame, direction: str) -> float:
        """Calcula relação risco/recompensa do rompimento"""
        try:
            entry = df.iloc[-1]['close']
            stop_loss = self._calculate_breakout_stop_loss(df, direction)
            take_profit = self._calculate_breakout_take_profit(df, direction)

            if direction == 'LONG':
                risk = entry - stop_loss
                reward = take_profit - entry
            else:
                risk = stop_loss - entry
                reward = entry - take_profit

            return reward / risk if risk > 0 else 0
        except:
            return 1.5  # Default