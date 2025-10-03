"""
Classificador de Regime de Mercado
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, List, Tuple
from enum import Enum


class MarketRegime(Enum):
    """Enumeração dos regimes de mercado"""
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    RANGING = "RANGING"
    SQUEEZE = "SQUEEZE"
    CHOPPY = "CHOPPY"


class MarketRegimeClassifier:
    """Classifica o regime atual do mercado baseado em múltiplos indicadores"""

    def __init__(self, config: Dict, logger):
        self.logger = logger
        self.config = config
        self.analysis_config = config.get('analysis', {})

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula todas as features técnicas, normalizando colunas antes de tudo."""

        self.logger.debug(f"Calculando features - DataFrame shape inicial: {df.shape}")

        try:
            # VERIFICAÇÃO INICIAL
            if df.empty or len(df) < 20:
                self.logger.debug("DataFrame vazio ou insuficiente para cálculo de features.")
                return pd.DataFrame()  # Retorna DF vazio para indicar falha

            df = df.copy()

            # ✅ SOLUÇÃO: Normalizar nomes das colunas ANTES de qualquer verificação
            column_mapping = {'min': 'low', 'max': 'high', 'from': 'timestamp'}
            df.rename(columns=column_mapping, inplace=True)

            # Verificar colunas necessárias APÓS a normalização
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.error(f"Mesmo após normalização, colunas essenciais estão faltando: {missing_columns}")
                return pd.DataFrame()  # Retorna DF vazio para indicar falha

            # CONVERSÃO NUMÉRICA
            for col in required_columns + ['volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remover NaNs que possam ter surgido na conversão
            df = df.dropna(subset=required_columns)

            if len(df) < 20:
                self.logger.warning("Dados insuficientes após limpeza e conversão numérica.")
                return pd.DataFrame()

            # --- O resto da função continua como antes ---

            # CONFIGURAÇÕES
            bb_period = self.analysis_config.get('bb_period', 20)
            ema_fast = self.analysis_config.get('ema_fast', 9)
            ema_slow = self.analysis_config.get('ema_slow', 21)
            rsi_period = self.analysis_config.get('rsi_period', 14)
            atr_period = self.analysis_config.get('atr_period', 14)
            adx_period = self.analysis_config.get('adx_period', 14)

            # 1. BANDAS DE BOLLINGER
            try:
                rolling_mean = df['close'].rolling(window=bb_period).mean()
                rolling_std = df['close'].rolling(window=bb_period).std()
                df['bb_upper'] = rolling_mean + (rolling_std * 2)
                df['bb_middle'] = rolling_mean
                df['bb_lower'] = rolling_mean - (rolling_std * 2)
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            except Exception as e:
                self.logger.warning(f"Erro no cálculo BB: {e}")
                df['bb_middle'] = df['close']  # Fallback mínimo

            # 2. EMAs
            df['ema_fast'] = df['close'].ewm(span=ema_fast, adjust=False).mean()
            df['ema_slow'] = df['close'].ewm(span=ema_slow, adjust=False).mean()

            # 3. RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            avg_gain = gain.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
            avg_loss = loss.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
            rs = avg_gain / avg_loss.replace(0, 0.001)
            df['rsi'] = 100 - (100 / (1 + rs))

            # 4. ATR
            high_low = df['high'] - df['low']
            high_close_prev = abs(df['high'] - df['close'].shift(1))
            low_close_prev = abs(df['low'] - df['close'].shift(1))
            true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            df['atr'] = true_range.ewm(alpha=1 / atr_period, adjust=False).mean()

            # 5. ADX
            try:
                import pandas_ta as ta
                adx_data = ta.adx(df['high'], df['low'], df['close'], length=adx_period)
                if adx_data is not None and not adx_data.empty:
                    df['adx'] = adx_data[f'ADX_{adx_period}']
                    df['plus_di'] = adx_data[f'DMP_{adx_period}']
                    df['minus_di'] = adx_data[f'DMN_{adx_period}']
            except Exception as e:
                self.logger.warning(f"Erro no cálculo do ADX: {e}")

            # Preencher quaisquer NaNs restantes no início das séries de indicadores
            df.bfill(inplace=True)

            # Garantir que as colunas existam, mesmo que o cálculo falhe
            indicator_cols = ['adx', 'plus_di', 'minus_di', 'rsi', 'ema_fast', 'ema_slow', 'bb_middle']
            for col in indicator_cols:
                if col not in df.columns:
                    df[col] = 50 if col == 'rsi' else 0 if col == 'adx' else df['close']

            return df

        except Exception as e:
            self.logger.error(f"Erro crítico em calculate_features: {e}", exc_info=True)
            return pd.DataFrame()  # Retorna DF vazio para indicar falha grave

    def classify_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Classifica o regime de mercado - COM DEBUG DETALHADO"""

        if len(df) < 10:
            self.logger.debug("Dados insuficientes para classificação - retornando CHOPPY")
            return MarketRegime.CHOPPY

        try:
            current = df.iloc[-1]

            # VERIFICAÇÃO ROBUSTA DE VALORES
            required_values = ['adx', 'bb_width', 'ema_fast', 'ema_slow', 'plus_di', 'minus_di']
            for value in required_values:
                if value not in current or pd.isna(current[value]):
                    self.logger.debug(f"Valor {value} faltando ou NaN - retornando CHOPPY")
                    return MarketRegime.CHOPPY

            # Garantir que valores são numéricos
            adx = float(current['adx'])
            bb_width = float(current['bb_width'])
            ema_fast = float(current['ema_fast'])
            ema_slow = float(current['ema_slow'])
            plus_di = float(current['plus_di'])
            minus_di = float(current['minus_di'])
            atr_percent = (float(current['atr']) / float(current['close'])) * 100

            # DEBUG DETALHADO - VALORES REAIS
            self.logger.debug(f"=== DEBUG CLASSIFICAÇÃO ===")
            self.logger.debug(f"ADX: {adx:.1f}, BB Width: {bb_width:.4f}, ATR%: {atr_percent:.3f}")
            self.logger.debug(f"EMA Fast: {ema_fast:.5f}, EMA Slow: {ema_slow:.5f}")
            self.logger.debug(f"+DI: {plus_di:.1f}, -DI: {minus_di:.1f}")
            self.logger.debug(f"Close: {current['close']:.5f}, BB Middle: {current['bb_middle']:.5f}")

            # 1. Verificar SQUEEZE - DEBUG DETALHADO
            bb_squeeze_threshold = self.analysis_config.get('bb_squeeze_threshold', 0.02)
            atr_squeeze_threshold = 0.3

            squeeze_conditions = (
                    bb_width < bb_squeeze_threshold and
                    atr_percent < atr_squeeze_threshold and
                    adx < 20
            )

            self.logger.debug(f"CONDIÇÕES SQUEEZE:")
            self.logger.debug(f"  BB Width {bb_width:.4f} < {bb_squeeze_threshold}: {bb_width < bb_squeeze_threshold}")
            self.logger.debug(
                f"  ATR% {atr_percent:.3f} < {atr_squeeze_threshold}: {atr_percent < atr_squeeze_threshold}")
            self.logger.debug(f"  ADX {adx:.1f} < 20: {adx < 20}")
            self.logger.debug(f"  SQUEEZE TOTAL: {squeeze_conditions}")

            if squeeze_conditions:
                self.logger.debug("✅ REGIME: SQUEEZE")
                return MarketRegime.SQUEEZE

            # 2. Verificar Tendência FORTE
            adx_threshold = self.analysis_config.get('adx_threshold', 25)
            adx_strong = adx > adx_threshold

            # Verificar se EMAs estão bem definidas
            ema_diff_percent = abs(ema_fast - ema_slow) / current['close'] * 100
            ema_well_defined = ema_diff_percent > 0.05

            uptrend_conditions = (
                    ema_fast > ema_slow and
                    ema_well_defined and
                    plus_di > minus_di and
                    current['close'] > current['bb_middle']
            )

            downtrend_conditions = (
                    ema_fast < ema_slow and
                    ema_well_defined and
                    plus_di < minus_di and
                    current['close'] < current['bb_middle']
            )

            self.logger.debug(f"CONDIÇÕES UPTREND:")
            self.logger.debug(f"  EMA Fast > Slow: {ema_fast > ema_slow}")
            self.logger.debug(f"  EMA Definida (>0.05%): {ema_well_defined} ({ema_diff_percent:.3f}%)")
            self.logger.debug(f"  +DI > -DI: {plus_di > minus_di}")
            self.logger.debug(f"  Close > BB Middle: {current['close'] > current['bb_middle']}")
            self.logger.debug(f"  ADX Forte (>25): {adx_strong}")
            self.logger.debug(f"  UPTREND TOTAL: {adx_strong and uptrend_conditions}")

            self.logger.debug(f"CONDIÇÕES DOWNTREND:")
            self.logger.debug(f"  EMA Fast < Slow: {ema_fast < ema_slow}")
            self.logger.debug(f"  EMA Definida (>0.05%): {ema_well_defined} ({ema_diff_percent:.3f}%)")
            self.logger.debug(f"  +DI < -DI: {plus_di < minus_di}")
            self.logger.debug(f"  Close < BB Middle: {current['close'] < current['bb_middle']}")
            self.logger.debug(f"  ADX Forte (>25): {adx_strong}")
            self.logger.debug(f"  DOWNTREND TOTAL: {adx_strong and downtrend_conditions}")

            if adx_strong and uptrend_conditions:
                self.logger.debug("✅ REGIME: UPTREND")
                return MarketRegime.UPTREND
            elif adx_strong and downtrend_conditions:
                self.logger.debug("✅ REGIME: DOWNTREND")
                return MarketRegime.DOWNTREND

            # 3. Verificar Tendência FRACA/MODERADA
            adx_moderate = adx > 18

            if adx_moderate and uptrend_conditions:
                self.logger.debug("✅ REGIME: UPTREND (Moderado)")
                return MarketRegime.UPTREND
            elif adx_moderate and downtrend_conditions:
                self.logger.debug("✅ REGIME: DOWNTREND (Moderado)")
                return MarketRegime.DOWNTREND

            # 4. Mercado Lateral (RANGING)
            ranging_conditions = (
                    adx < 25 and
                    bb_width > 0.01 and
                    abs(current['close'] - current['bb_middle']) / current['bb_middle'] < 0.02
            )

            self.logger.debug(f"CONDIÇÕES RANGING:")
            self.logger.debug(f"  ADX < 25: {adx < 25}")
            self.logger.debug(f"  BB Width > 0.01: {bb_width > 0.01}")
            self.logger.debug(
                f"  Preço próximo média (<2%): {abs(current['close'] - current['bb_middle']) / current['bb_middle'] < 0.02}")
            self.logger.debug(f"  RANGING TOTAL: {ranging_conditions}")

            if ranging_conditions:
                self.logger.debug("✅ REGIME: RANGING")
                return MarketRegime.RANGING

            # 5. Mercado CHOPPY
            choppy_conditions = (
                    adx < 12 or
                    atr_percent > 1.0 or
                    bb_width > 0.05
            )

            self.logger.debug(f"CONDIÇÕES CHOPPY:")
            self.logger.debug(f"  ADX < 12: {adx < 12}")
            self.logger.debug(f"  ATR% > 1.0: {atr_percent > 1.0}")
            self.logger.debug(f"  BB Width > 0.05: {bb_width > 0.05}")
            self.logger.debug(f"  CHOPPY TOTAL: {choppy_conditions}")

            if choppy_conditions:
                self.logger.debug("✅ REGIME: CHOPPY")
                return MarketRegime.CHOPPY

            # 6. Default para RANGING
            self.logger.debug("✅ REGIME: RANGING (Padrão)")
            return MarketRegime.RANGING

        except Exception as e:
            self.logger.error(f"Erro na classificação: {e}")
            return MarketRegime.CHOPPY

    def get_detailed_analysis(self, df: pd.DataFrame) -> Dict:
        """Retorna análise detalhada do mercado - VERSÃO CORRIGIDA"""

        try:
            # VERIFICAÇÃO ROBUSTA do DataFrame
            if df is None or df.empty or len(df) < 2:
                self.logger.warning("DataFrame inválido para análise detalhada")
                return self._get_fallback_analysis()

            df_with_features = self.calculate_features(df)

            # ✅ VERIFICAÇÃO MAIS DETALHADA após calculate_features
            if (df_with_features is None or
                    df_with_features.empty or
                    len(df_with_features) < 2 or
                    'adx' not in df_with_features.columns or
                    'bb_width' not in df_with_features.columns):
                self.logger.warning("DataFrame com features inválido ou incompleto")
                self.logger.debug(
                    f"Colunas disponíveis: {df_with_features.columns.tolist() if df_with_features is not None else 'None'}")
                return self._get_fallback_analysis()

            regime = self.classify_regime(df_with_features)
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
            return self._get_fallback_analysis()

    def _get_fallback_analysis(self) -> Dict:
        """Retorna análise de fallback para dados inválidos"""
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