"""
Classificador de Regime de Mercado
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, List, Tuple
from enum import Enum

from bot.utils.logger import setup_logger


class MarketRegime(Enum):
    """Enumeração dos regimes de mercado"""
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    RANGING = "RANGING"
    SQUEEZE = "SQUEEZE"
    CHOPPY = "CHOPPY"


class MarketRegimeClassifier:
    """Classifica o regime atual do mercado baseado em múltiplos indicadores"""

    def __init__(self, config: Dict):
        self.logger = setup_logger()
        self.config = config
        self.analysis_config = config.get('analysis', {})

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula todas as features técnicas - VERSÃO CORRIGIDA"""

        print(f"=== DEBUG CALCULATE_FEATURES ===")
        print(f"DataFrame shape: {df.shape}")

        try:
            # VERIFICAÇÃO INICIAL
            if df.empty or len(df) < 20:
                print("DEBUG: DataFrame vazio ou insuficiente")
                return df

            # Verificar colunas necessárias
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"DEBUG: Colunas faltando: {missing_columns}")
                return df

            # Criar cópia
            df = df.copy()

            # CONVERSÃO NUMÉRICA
            for col in required_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remover NaNs
            df = df.dropna()

            if len(df) < 20:
                print("DEBUG: Dados insuficientes após limpeza")
                return df

            print(f"DEBUG: Dados válidos - shape: {df.shape}")

            # CONFIGURAÇÕES
            bb_period = 20
            ema_fast = 9
            ema_slow = 21
            rsi_period = 14
            atr_period = 14
            adx_period = 14

            # 1. BANDAS DE BOLLINGER - CÁLCULO MANUAL GARANTIDO
            try:
                print("DEBUG: Calculando Bollinger Bands...")
                rolling_mean = df['close'].rolling(window=bb_period).mean()
                rolling_std = df['close'].rolling(window=bb_period).std()
                rolling_std = rolling_std.fillna(0.001)  # Evitar std zero

                df['bb_upper'] = rolling_mean + (rolling_std * 2)
                df['bb_middle'] = rolling_mean
                df['bb_lower'] = rolling_mean - (rolling_std * 2)
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
                print(f"DEBUG: BB calculado - Upper: {df['bb_upper'].iloc[-1]:.5f}")
            except Exception as e:
                print(f"DEBUG: Erro BB: {e}")
                # Fallback
                df['bb_upper'] = df['close'] * 1.02
                df['bb_middle'] = df['close']
                df['bb_lower'] = df['close'] * 0.98
                df['bb_width'] = 0.04

            # 2. EMAs - CÁLCULO MANUAL GARANTIDO
            try:
                print("DEBUG: Calculando EMAs...")
                df['ema_fast'] = df['close'].ewm(span=ema_fast, adjust=False).mean()
                df['ema_slow'] = df['close'].ewm(span=ema_slow, adjust=False).mean()
                print(f"DEBUG: EMA calculado - Fast: {df['ema_fast'].iloc[-1]:.5f}")
            except Exception as e:
                print(f"DEBUG: Erro EMA: {e}")
                df['ema_fast'] = df['close'].rolling(window=ema_fast).mean()
                df['ema_slow'] = df['close'].rolling(window=ema_slow).mean()

            # 3. RSI - CÁLCULO MANUAL GARANTIDO
            try:
                print("DEBUG: Calculando RSI...")
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).fillna(0)
                loss = (-delta.where(delta < 0, 0)).fillna(0)

                avg_gain = gain.rolling(window=rsi_period, min_periods=1).mean()
                avg_loss = loss.rolling(window=rsi_period, min_periods=1).mean()
                avg_loss = avg_loss.replace(0, 0.001)  # Evitar divisão por zero

                rs = avg_gain / avg_loss
                df['rsi'] = 100 - (100 / (1 + rs))
                print(f"DEBUG: RSI calculado: {df['rsi'].iloc[-1]:.2f}")
            except Exception as e:
                print(f"DEBUG: Erro RSI: {e}")
                df['rsi'] = 50.0

            # 4. ATR - CÁLCULO MANUAL GARANTIDO
            try:
                print("DEBUG: Calculando ATR...")
                high_low = df['high'] - df['low']
                high_close_prev = abs(df['high'] - df['close'].shift(1))
                low_close_prev = abs(df['low'] - df['close'].shift(1))

                # Preencher NaN do shift
                high_close_prev = high_close_prev.fillna(high_low)
                low_close_prev = low_close_prev.fillna(high_low)

                true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
                df['atr'] = true_range.rolling(window=atr_period, min_periods=1).mean()
                print(f"DEBUG: ATR calculado: {df['atr'].iloc[-1]:.5f}")
            except Exception as e:
                print(f"DEBUG: Erro ATR: {e}")
                df['atr'] = (df['high'] - df['low']).rolling(window=atr_period).mean()

            # 5. ADX SIMPLIFICADO - CÁLCULO MANUAL GARANTIDO
            try:
                print("DEBUG: Calculando ADX...")
                # Baseado na volatilidade e tendência
                volatility = df['close'].pct_change().abs().rolling(window=5, min_periods=1).mean() * 100
                trend_strength = abs(df['ema_fast'] - df['ema_slow']) / df['close'] * 100

                # Combinar fatores
                df['adx'] = (volatility * 0.6 + trend_strength * 0.4).rolling(window=adx_period, min_periods=1).mean()
                df['adx'] = df['adx'].clip(5, 60)  # Limitar entre 5-60

                # +DI e -DI simplificados
                price_changes = df['close'].diff()
                df['plus_di'] = (price_changes > 0).rolling(window=adx_period, min_periods=1).mean() * 100
                df['minus_di'] = (price_changes < 0).rolling(window=adx_period, min_periods=1).mean() * 100

                print(f"DEBUG: ADX calculado: {df['adx'].iloc[-1]:.2f}")
            except Exception as e:
                print(f"DEBUG: Erro ADX: {e}")
                df['adx'] = 25.0
                df['plus_di'] = 30.0
                df['minus_di'] = 30.0

            # PREENCHIMENTO FINAL
            indicator_cols = ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
                              'ema_fast', 'ema_slow', 'rsi', 'atr', 'adx', 'plus_di', 'minus_di']

            for col in indicator_cols:
                if col in df.columns:
                    df[col] = df[col].ffill().bfill()
                    # Garantir valores padrão se ainda houver NaN
                    if df[col].isna().any() or df[col].isnull().any():
                        if col == 'rsi':
                            df[col] = 50.0
                        elif col == 'adx':
                            df[col] = 25.0
                        elif 'di' in col:
                            df[col] = 30.0
                        elif 'bb' in col:
                            df[col] = df['close']
                        else:
                            df[col] = df['close']

            # DEBUG FINAL
            print("DEBUG: Indicadores calculados:")
            print(f"BB Upper: {df['bb_upper'].iloc[-1]:.5f}")
            print(f"BB Lower: {df['bb_lower'].iloc[-1]:.5f}")
            print(f"ADX: {df['adx'].iloc[-1]:.2f}")
            print(f"RSI: {df['rsi'].iloc[-1]:.2f}")
            print(f"EMA Fast: {df['ema_fast'].iloc[-1]:.5f}")
            print(f"EMA Slow: {df['ema_slow'].iloc[-1]:.5f}")
            print(f"ATR: {df['atr'].iloc[-1]:.5f}")
            print("=============================")

            return df

        except Exception as e:
            print(f"DEBUG: Erro crítico em calculate_features: {e}")
            import traceback
            traceback.print_exc()
            # Retornar fallback mínimo
            for col in ['bb_upper', 'bb_middle', 'bb_lower', 'ema_fast', 'ema_slow']:
                if col not in df.columns:
                    df[col] = df['close']
            if 'rsi' not in df.columns:
                df['rsi'] = 50.0
            if 'adx' not in df.columns:
                df['adx'] = 25.0
            return df

    def classify_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Classifica o regime de mercado - VERSÃO COM LIMIARES CORRETOS"""

        if len(df) < 10:
            return MarketRegime.CHOPPY

        try:
            current = df.iloc[-1]

            # VERIFICAÇÃO ROBUSTA DE VALORES
            required_values = ['adx', 'bb_width', 'ema_fast', 'ema_slow', 'plus_di', 'minus_di']
            for value in required_values:
                if value not in current or pd.isna(current[value]):
                    return MarketRegime.CHOPPY

            # Garantir que valores são numéricos
            adx = float(current['adx'])
            bb_width = float(current['bb_width'])
            ema_fast = float(current['ema_fast'])
            ema_slow = float(current['ema_slow'])
            plus_di = float(current['plus_di'])
            minus_di = float(current['minus_di'])
            atr_percent = (float(current['atr']) / float(current['close'])) * 100

            # 1. Verificar SQUEEZE - COM LIMIARES MAIS REALISTAS
            bb_squeeze_threshold = self.analysis_config.get('bb_squeeze_threshold', 0.005)  # 0.5%
            atr_squeeze_threshold = 0.1  # 0.1% de volatilidade

            squeeze_conditions = (
                    bb_width < bb_squeeze_threshold and
                    atr_percent < atr_squeeze_threshold and
                    adx < 15  # Tendência fraca
            )

            if squeeze_conditions:
                return MarketRegime.SQUEEZE

            # 2. Verificar Tendência FORTE
            adx_threshold = self.analysis_config.get('adx_threshold', 25)
            adx_strong = adx > adx_threshold

            uptrend_conditions = (
                    ema_fast > ema_slow and
                    plus_di > minus_di and
                    current['close'] > current['bb_middle']  # Preço acima da média
            )

            downtrend_conditions = (
                    ema_fast < ema_slow and
                    plus_di < minus_di and
                    current['close'] < current['bb_middle']  # Preço abaixo da média
            )

            if adx_strong and uptrend_conditions:
                return MarketRegime.UPTREND
            elif adx_strong and downtrend_conditions:
                return MarketRegime.DOWNTREND

            # 3. Verificar Tendência FRACA/MODERADA
            adx_moderate = adx > 15

            if adx_moderate and uptrend_conditions:
                return MarketRegime.UPTREND
            elif adx_moderate and downtrend_conditions:
                return MarketRegime.DOWNTREND

            # 4. Mercado Lateral (RANGING)
            ranging_conditions = (
                    adx < 20 and  # Tendência fraca
                    bb_width > 0.01 and  # Bandas com largura razoável
                    abs(current['close'] - current['bb_middle']) / current['bb_middle'] < 0.01  # Preço próximo à média
            )

            if ranging_conditions:
                return MarketRegime.RANGING

            # 5. Mercado CHOPPY (condições perigosas)
            choppy_conditions = (
                    adx < 10 or  # Tendência muito fraca
                    atr_percent > 0.5 or  # Volatilidade excessiva
                    bb_width > 0.03  # Bandas muito largas (mercado instável)
            )

            if choppy_conditions:
                return MarketRegime.CHOPPY

            # 6. Default para RANGING
            return MarketRegime.RANGING

        except Exception as e:
            self.logger.error(f"Erro na classificação: {e}")
            return MarketRegime.CHOPPY

    def get_detailed_analysis(self, df: pd.DataFrame) -> Dict:
        """Retorna análise detalhada do mercado - VERSÃO CORRIGIDA"""

        try:
            df_with_features = self.calculate_features(df)
            regime = self.classify_regime(df_with_features)

            # VERIFICAÇÃO CRÍTICA: garantir que temos dados válidos
            if df_with_features.empty or len(df_with_features) < 2:
                return {
                    'regime': MarketRegime.CHOPPY,
                    'adx': 0.0,
                    'adx_strength': 'Fraca',
                    'bb_width': 0.0,
                    'bb_position': 'Central',
                    'atr': 0.0,
                    'atr_percent': 0.0,
                    'trend_direction': 'Indefinida',
                    'rsi': 50.0,
                    'rsi_signal': 'Neutro'
                }

            current = df_with_features.iloc[-1]

            # Garantir que todos os valores são numéricos
            analysis = {
                'regime': regime,
                'adx': float(current.get('adx', 0.0)),
                'adx_strength': 'Forte' if current.get('adx', 0) > 25 else 'Moderada' if current.get('adx',
                                                                                                     0) > 15 else 'Fraca',
                'bb_width': float(current.get('bb_width', 0.0)),
                'bb_position': self._get_bb_position(current),
                'atr': float(current.get('atr', 0.0)),
                'atr_percent': (float(current.get('atr', 0.0)) / float(current.get('close', 1.0))) * 100,
                'trend_direction': 'Alta' if current.get('ema_fast', 0) > current.get('ema_slow', 0) else 'Baixa',
                'rsi': float(current.get('rsi', 50.0)),
                'rsi_signal': 'Sobrevendido' if current.get('rsi', 50) < 30 else 'Sobrecomprado' if current.get('rsi',
                                                                                                                50) > 70 else 'Neutro'
            }

            return analysis

        except Exception as e:
            self.logger.error(f"Erro em get_detailed_analysis: {e}")
            # Retorno de fallback seguro
            return {
                'regime': MarketRegime.CHOPPY,
                'adx': 0.0,
                'adx_strength': 'Fraca',
                'bb_width': 0.0,
                'bb_position': 'Central',
                'atr': 0.0,
                'atr_percent': 0.0,
                'trend_direction': 'Indefinida',
                'rsi': 50.0,
                'rsi_signal': 'Neutro'
            }

    def _get_bb_position(self, current) -> str:
        """Determina a posição do preço nas Bandas de Bollinger"""
        try:
            close = current.get('close', 0)
            bb_upper = current.get('bb_upper', close * 1.02)
            bb_lower = current.get('bb_lower', close * 0.98)
            bb_middle = current.get('bb_middle', close)

            if close >= bb_upper * 0.99:
                return 'Superior'
            elif close <= bb_lower * 1.01:
                return 'Inferior'
            else:
                return 'Central'
        except:
            return 'Central'