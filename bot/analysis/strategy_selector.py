"""
Seletor de Estratégias Baseado no Regime
"""

import pandas as pd
from typing import Dict, Optional, Tuple
from .regime_classifier import MarketRegime, MarketRegimeClassifier


class StrategySelector:
    """Seleciona e executa estratégias baseadas no regime de mercado"""

    def __init__(self, config: Dict, logger):
        self.logger = logger
        self.config = config
        self.regime_classifier = MarketRegimeClassifier(config, self.logger)
        self.analysis_config = config.get('analysis', {})

    def debug_confidence_calculation(self, df_with_features: pd.DataFrame, signal: Dict):
        """Debug do cálculo de confiança - VERSÃO CORRIGIDA (sem isEnabledFor)"""

        # ✅ CORREÇÃO: Verificar nível de debug de forma compatível com loguru
        try:
            # Loguru não tem isEnabledFor, então verificamos de outra forma
            # Podemos simplesmente tentar logar em DEBUG - se o nível não for DEBUG, não aparecerá
            if df_with_features is None or df_with_features.empty:
                self.logger.debug("DEBUG CONFIANÇA: DataFrame vazio")
                return

            current = df_with_features.iloc[-1]

            # ✅ USAR logger.debug diretamente - se o nível não for DEBUG, não será exibido
            self.logger.debug("=== DEBUG CONFIANÇA CORRETO ===")
            self.logger.debug(f"Regime: {signal['regime'].value}")
            self.logger.debug(f"Ação: {signal['action']}")
            self.logger.debug(f"Close: {current['close']:.5f}")
            self.logger.debug(f"BB Upper: {current.get('bb_upper', 0):.5f}")
            self.logger.debug(f"BB Lower: {current.get('bb_lower', 0):.5f}")
            self.logger.debug(f"ADX: {current.get('adx', 0):.2f}")
            self.logger.debug(f"RSI: {current.get('rsi', 0):.2f}")
            self.logger.debug(f"EMA Fast: {current.get('ema_fast', 0):.5f}")
            self.logger.debug(f"EMA Slow: {current.get('ema_slow', 0):.5f}")
            self.logger.debug(f"Volume: {current.get('volume', 0)}")
            self.logger.debug("===============================")

        except Exception as e:
            # Silenciosamente ignora erros no debug
            pass

    def get_trading_signal(self, df_main: pd.DataFrame, df_context: pd.DataFrame = None) -> Dict:
        """Gera sinal de trading com base em múltiplos timeframes - ✅ VERSÃO DE ELITE"""
        if df_main is None or df_main.empty:
            self.logger.warning("DataFrame principal vazio recebido para sinal")
            return self._get_empty_signal()

        try:
            # Calcular features UMA VEZ para cada timeframe
            df_main_features = self.regime_classifier.calculate_features(df_main)

            # Obter a tendência principal do timeframe de contexto, se disponível
            context_trend = 'Indefinida'
            if df_context is not None and not df_context.empty:
                df_context_features = self.regime_classifier.calculate_features(df_context)
                context_analysis = self.regime_classifier.get_detailed_analysis(df_context_features)
                context_trend = context_analysis.get('trend_direction', 'Indefinida')
                self.logger.debug(f"Contexto (MTA) definido: Tendência de {context_trend}")

            if df_main_features is None or df_main_features.empty:
                self.logger.warning("DataFrame principal com features vazio")
                return self._get_empty_signal()

            analysis = self.regime_classifier.get_detailed_analysis(df_main_features)
            if analysis is None:
                self.logger.warning("Análise retornou None")
                return self._get_empty_signal()

            regime = analysis['regime']
            signal = self._get_empty_signal()
            signal.update({
                'regime': regime,
                'analysis': analysis
            })

            if regime == MarketRegime.CHOPPY:
                signal['strategy_used'] = 'SAFETY_FILTER'
                return signal

            # Aplicar estratégia baseada no regime, passando o contexto
            if regime == MarketRegime.UPTREND:
                return self._trend_following_strategy(df_main_features, 'LONG', analysis, context_trend)
            elif regime == MarketRegime.DOWNTREND:
                return self._trend_following_strategy(df_main_features, 'SHORT', analysis, context_trend)
            elif regime == MarketRegime.RANGING:
                return self._mean_reversion_strategy(df_main_features, analysis, context_trend)
            elif regime == MarketRegime.SQUEEZE:
                return self._breakout_watch_strategy(df_main_features, analysis)

            return signal

        except Exception as e:
            self.logger.error(f"Erro crítico em get_trading_signal: {e}")
            return self._get_empty_signal()

    def _get_empty_signal(self) -> Dict:
        """Retorna sinal vazio para dados inválidos"""
        return {
            'action': 'WAIT',
            'direction': None,
            'confidence': 0.0,
            'regime': MarketRegime.CHOPPY,
            'analysis': {},
            'strategy_used': 'NO_DATA'
        }

    def _trend_following_strategy(self, df: pd.DataFrame, trend_direction: str, analysis: Dict,
                                  context_trend: str) -> Dict:
        """Estratégia de seguir tendência com filtro MTA e entrada em pullback - ✅ VERSÃO DE ELITE"""
        signal = self._get_wait_signal(analysis, 'TREND_FOLLOWING')
        signal['direction'] = trend_direction

        # ✅ FILTRO DE ELITE 1: Análise Multi-Timeframe (MTA)
        # A operação deve estar a favor da tendência do timeframe maior.
        is_aligned_with_context = (trend_direction == 'LONG' and context_trend == 'Alta') or \
                                  (trend_direction == 'SHORT' and context_trend == 'Baixa') or \
                                  context_trend == 'Indefinida'  # Permite se o contexto não for claro

        if not is_aligned_with_context:
            self.logger.debug(f"Trade {trend_direction} bloqueada. Sinal M1 contra tendência M15 ({context_trend}).")
            return signal

        current = df.iloc[-1]
        rsi = current.get('rsi', 50)
        adx = current.get('adx', 0)

        # ✅ FILTRO DE ELITE 2: Entrada em Pullback
        # Não entramos no pico do movimento, mas sim na retração.
        is_pullback = (trend_direction == 'LONG' and 40 <= rsi <= 55) or \
                      (trend_direction == 'SHORT' and 45 <= rsi <= 60)

        if not is_pullback or adx < 20:  # Exige ADX mínimo para confirmar que ainda é uma tendência
            self.logger.debug(f"Aguardando pullback para {trend_direction}. RSI atual: {rsi:.2f}")
            return signal

        # Se todas as condições de elite forem atendidas, calculamos a confiança.
        adx_score = min(adx / 50.0, 1.0)
        rsi_score = 1.0 - (abs(50 - rsi) / 50.0)  # Mais perto de 50, maior o score no pullback

        # Bônus se o pullback quase tocou a EMA lenta, um ponto de suporte/resistência dinâmico
        ema_slow = current.get('ema_slow', 0)
        pullback_bonus = 0
        if trend_direction == 'LONG' and current['low'] <= ema_slow:
            pullback_bonus = 0.2
        elif trend_direction == 'SHORT' and current['high'] >= ema_slow:
            pullback_bonus = 0.2

        final_confidence = (adx_score * 0.4) + (rsi_score * 0.4) + (pullback_bonus * 0.2)

        # Confiança adicional pela MTA
        if is_aligned_with_context:
            final_confidence = min(final_confidence * 1.2, 0.95)

        if final_confidence >= self.config['trading'].get('min_confidence', 0.5):
            signal.update({
                'action': 'ENTER',
                'confidence': final_confidence
            })

        return signal

    def _get_wait_signal(self, analysis: Dict, strategy: str) -> Dict:
        """Retorna sinal de espera padrão"""
        return {
            'action': 'WAIT',
            'direction': None,
            'confidence': 0.0,
            'regime': analysis['regime'],
            'analysis': analysis,
            'strategy_used': strategy
        }

    def _mean_reversion_strategy(self, df: pd.DataFrame, analysis: Dict, context_trend: str) -> Dict:
        """Estratégia de reversão à média com vela de confirmação e filtro MTA - ✅ VERSÃO DE ELITE"""
        signal = self._get_wait_signal(analysis, 'MEAN_REVERSION')

        # ✅ FILTRO DE ELITE: Evitar reversão contra uma tendência de contexto forte.
        if (context_trend == 'Alta' and analysis['rsi'] > 50) or \
                (context_trend == 'Baixa' and analysis['rsi'] < 50):
            self.logger.debug(f"Reversão bloqueada pela forte tendência de contexto ({context_trend}).")
            return signal

        current = df.iloc[-1]

        # CONDIÇÃO DE SHORT: Sobrecompra + Vela de Confirmação Baixista
        is_overbought = current['close'] >= current['bb_upper'] * 0.995 or current['rsi'] > 72
        is_bearish_confirmation = current['close'] < current['open']

        if is_overbought and is_bearish_confirmation:
            confidence = 0.75 + (min(current['rsi'] - 70, 20) / 100)  # Confiança aumenta com RSI mais extremo
            signal.update({
                'action': 'ENTER',
                'direction': 'SHORT',
                'confidence': min(confidence, 0.95)
            })
            return signal

        # CONDIÇÃO DE LONG: Sobrevenda + Vela de Confirmação Altista
        is_oversold = current['close'] <= current['bb_lower'] * 1.005 or current['rsi'] < 28
        is_bullish_confirmation = current['close'] > current['open']

        if is_oversold and is_bullish_confirmation:
            confidence = 0.75 + (min(30 - current['rsi'], 20) / 100)
            signal.update({
                'action': 'ENTER',
                'direction': 'LONG',
                'confidence': min(confidence, 0.95)
            })
            return signal

        return signal

    def _breakout_watch_strategy(self, df: pd.DataFrame, analysis: Dict) -> Dict:
        """Estratégia completa de rompimento para mercado em SQUEEZE"""

        if len(df) < 10:
            return self._get_breakout_wait_signal(analysis)

        current = df.iloc[-1]

        # CONFIGURAÇÃO INICIAL COM CONFIANÇA BASE
        signal = {
            'action': 'OBSERVE',
            'direction': None,
            'confidence': 0.3,  # CONFIANÇA BASE PARA SQUEEZE - CORREÇÃO AQUI
            'regime': analysis['regime'],
            'analysis': analysis,
            'strategy_used': 'BREAKOUT',
            'breakout_type': 'WATCHING',
            'risk_reward_ratio': 0.0
        }

        # 1. VERIFICAR SE AINDA ESTÁ EM SQUEEZE
        bb_squeeze_threshold = self.analysis_config.get('bb_squeeze_threshold', 0.005)
        if current['bb_width'] >= bb_squeeze_threshold:
            signal['breakout_type'] = 'SQUEEZE_ENDED'
            signal['confidence'] = 0.1  # Baixa confiança quando squeeze termina
            return signal

        # 2. DETECTAR ROMPIMENTO
        breakout_detected, breakout_direction = self._detect_breakout(df)

        if breakout_detected:
            # 3. CONFIRMAR ROMPIMENTO
            if self._confirm_breakout(df, breakout_direction, 2):
                # 4. CALCULAR CONFIANÇA E RISCO/RECOMPENSA
                confidence = self._calculate_breakout_confidence(df, breakout_direction)
                risk_reward = self._calculate_breakout_risk_reward(df, breakout_direction)

                # 5. FILTRO DE RISCO/RECOMPENSA (mínimo 1:1.5)
                if risk_reward >= 1.5 and confidence >= 0.6:
                    signal.update({
                        'action': 'ENTER',
                        'direction': breakout_direction,
                        'confidence': confidence,
                        'breakout_type': 'CONFIRMED_HIGH_CONFIDENCE',
                        'risk_reward_ratio': risk_reward,
                        'stop_loss': self._calculate_breakout_stop_loss(df, breakout_direction),
                        'take_profit': self._calculate_breakout_take_profit(df, breakout_direction)
                    })
                else:
                    signal.update({
                        'breakout_type': 'CONFIRMED_LOW_RR',
                        'risk_reward_ratio': risk_reward,
                        'confidence': max(confidence, 0.4)  # Mínimo 40% de confiança
                    })
            else:
                signal.update({
                    'breakout_type': 'PENDING_CONFIRMATION',
                    'confidence': 0.5  # Confiança moderada aguardando confirmação
                })
        else:
            # SEM ROMPIMENTO - CALCULAR CONFIANÇA BASEADA NA FORÇA DO SQUEEZE
            squeeze_strength = self._calculate_squeeze_strength(df)
            signal['confidence'] = 0.3 * squeeze_strength  # Confiança base no squeeze

        return signal

    def _calculate_squeeze_strength(self, df: pd.DataFrame) -> float:
        """Calcula a força do squeeze baseado em múltiplos fatores"""
        if len(df) < 5:
            return 0.5

        current = df.iloc[-1]

        try:
            factors = []

            # 1. Largura das Bandas de Bollinger (quanto menor, mais forte o squeeze)
            bb_width = current.get('bb_width', 0.02)
            bb_score = max(0, 1.0 - (bb_width / 0.01))  # Normalizado para threshold de 0.01
            factors.append(min(bb_score, 1.0))

            # 2. Volatilidade (ATR) - quanto menor, mais forte o squeeze
            atr_percent = (current.get('atr', 0) / current.get('close', 1)) * 100
            atr_score = max(0, 1.0 - (atr_percent / 0.1))  # Normalizado para threshold de 0.1%
            factors.append(min(atr_score, 1.0))

            # 3. ADX - tendência fraca é melhor para squeeze
            adx = current.get('adx', 25)
            adx_score = max(0, 1.0 - (adx / 20.0))  # ADX baixo é melhor
            factors.append(min(adx_score, 1.0))

            # 4. Volume - volume baixo pode indicar acumulação
            volume_avg = df['volume'].tail(10).mean()
            current_volume = current.get('volume', volume_avg)
            volume_ratio = current_volume / volume_avg if volume_avg > 0 else 1
            volume_score = 1.0 if volume_ratio < 0.8 else 0.7 if volume_ratio < 1.2 else 0.4
            factors.append(volume_score)

            # Média dos fatores
            squeeze_strength = sum(factors) / len(factors)
            return min(squeeze_strength, 1.0)

        except Exception as e:
            self.logger.error(f"Erro cálculo força do squeeze: {e}")
            return 0.5

    def _get_breakout_wait_signal(self, analysis: Dict) -> Dict:
        """Retorna sinal de espera para breakout com confiança mínima"""
        return {
            'action': 'OBSERVE',
            'direction': None,
            'confidence': 0.2,  # CONFIANÇA MÍNIMA - CORREÇÃO AQUI
            'regime': analysis['regime'],
            'analysis': analysis,
            'strategy_used': 'BREAKOUT',
            'breakout_type': 'INSUFFICIENT_DATA'
        }

    def _detect_breakout(self, df: pd.DataFrame) -> tuple:
        """Detecta se houve rompimento das bandas de Bollinger - VERSÃO CORRIGIDA"""
        if len(df) < 3:
            return False, None

        current = df.iloc[-1]
        prev = df.iloc[-2]

        # LIMIARES MAIS FLEXÍVEIS para detecção
        bb_touch_threshold = 0.998  # 99.8% da banda (mais sensível)
        volume_increase = 1.1  # 10% de aumento no volume

        # Rompimento para CIMA - CONDIÇÕES MAIS FLEXÍVEIS
        upside_breakout = (
                current['close'] >= current['bb_upper'] * bb_touch_threshold and  # CORRIGIDO
                current['volume'] > prev['volume'] * volume_increase  # Volume crescente
        )

        # Rompimento para BAIXO - CONDIÇÕES MAIS FLEXÍVEIS
        downside_breakout = (
                current['close'] <= current['bb_lower'] * (2 - bb_touch_threshold) and  # CORRIGIDO
                current['volume'] > prev['volume'] * volume_increase  # Volume crescente
        )

        if upside_breakout:
            self.logger.debug(
                f"Breakout UP detectado - Preço: {current['close']:.5f}, BB Upper: {current['bb_upper']:.5f}")
            return True, 'LONG'
        elif downside_breakout:
            self.logger.debug(
                f"Breakout DOWN detectado - Preço: {current['close']:.5f}, BB Lower: {current['bb_lower']:.5f}")
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