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
        """Calcula todas as features técnicas necessárias - Versão Robusta"""

        try:
            # Garantir que temos dados suficientes
            if len(df) < 50:
                self.logger.warning("Dados insuficientes para análise")
                return df

            # Converter para numérico e limpar dados
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.dropna()

            # Configurações
            adx_period = self.analysis_config.get('adx_period', 14)
            bb_period = self.analysis_config.get('bb_period', 20)
            bb_std = self.analysis_config.get('bb_std', 2)
            atr_period = self.analysis_config.get('atr_period', 14)
            ema_fast = self.analysis_config.get('ema_fast', 9)
            ema_slow = self.analysis_config.get('ema_slow', 21)
            rsi_period = self.analysis_config.get('rsi_period', 14)

            # Calcular cada indicador individualmente com try/except
            indicators = {}

            # 1. Bollinger Bands
            try:
                bb = ta.bbands(df['close'], length=bb_period, std=bb_std)
                indicators['bb_upper'] = bb[f'BBU_{bb_period}_{bb_std}.0']
                indicators['bb_middle'] = bb[f'BBM_{bb_period}_{bb_std}.0']
                indicators['bb_lower'] = bb[f'BBL_{bb_period}_{bb_std}.0']
            except Exception as e:
                self.logger.warning(f"Bollinger Bands falhou: {e}")
                # Fallback manual
                rolling_mean = df['close'].rolling(window=bb_period).mean()
                rolling_std = df['close'].rolling(window=bb_period).std()
                indicators['bb_upper'] = rolling_mean + (rolling_std * bb_std)
                indicators['bb_middle'] = rolling_mean
                indicators['bb_lower'] = rolling_mean - (rolling_std * bb_std)

            # 2. ADX
            try:
                adx_data = ta.adx(df['high'], df['low'], df['close'], length=adx_period)
                indicators['adx'] = adx_data[f'ADX_{adx_period}']
                indicators['plus_di'] = adx_data[f'DMP_{adx_period}']
                indicators['minus_di'] = adx_data[f'DMN_{adx_period}']
            except Exception as e:
                self.logger.warning(f"ADX falhou: {e}")
                indicators['adx'] = 25.0
                indicators['plus_di'] = 25.0
                indicators['minus_di'] = 25.0

            # 3. ATR
            try:
                indicators['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
            except Exception as e:
                self.logger.warning(f"ATR falhou: {e}")
                indicators['atr'] = (df['high'] - df['low']).rolling(window=atr_period).mean()

            # 4. EMAs
            try:
                indicators['ema_fast'] = ta.ema(df['close'], length=ema_fast)
                indicators['ema_slow'] = ta.ema(df['close'], length=ema_slow)
            except Exception as e:
                self.logger.warning(f"EMA falhou: {e}")
                indicators['ema_fast'] = df['close'].ewm(span=ema_fast).mean()
                indicators['ema_slow'] = df['close'].ewm(span=ema_slow).mean()

            # 5. RSI
            try:
                indicators['rsi'] = ta.rsi(df['close'], length=rsi_period)
            except Exception as e:
                self.logger.warning(f"RSI falhou: {e}")
                indicators['rsi'] = 50.0

            # Adicionar todos os indicadores ao DataFrame
            for name, values in indicators.items():
                df[name] = values

            # Calcular largura das Bandas de Bollinger
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']


            df = df.ffill().bfill()

            # Garantir que não há valores infinitos
            df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

            return df

        except Exception as e:
            self.logger.error(f"Erro crítico em calculate_features: {e}")
            # Retornar DataFrame original se tudo falhar
            return df

    def classify_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Classifica o regime de mercado baseado nas features calculadas"""

        if len(df) < 50:  # Dados insuficientes
            return MarketRegime.CHOPPY

        current = df.iloc[-1]
        prev = df.iloc[-2]

        # 1. Verificar Squeeze (compressão de volatilidade)
        bb_squeeze_threshold = self.analysis_config.get('bb_squeeze_threshold', 0.1)
        if current['bb_width'] < bb_squeeze_threshold:
            return MarketRegime.SQUEEZE

        # 2. Verificar Tendência com ADX
        adx_threshold = self.analysis_config.get('adx_threshold', 25)
        adx_strong = current['adx'] > adx_threshold

        # Tendência de Alta
        uptrend_conditions = (
                current['ema_fast'] > current['ema_slow'] and
                current['plus_di'] > current['minus_di'] and
                current['close'] > current['ema_slow']
        )

        # Tendência de Baixa
        downtrend_conditions = (
                current['ema_fast'] < current['ema_slow'] and
                current['plus_di'] < current['minus_di'] and
                current['close'] < current['ema_slow']
        )

        if adx_strong and uptrend_conditions:
            return MarketRegime.UPTREND
        elif adx_strong and downtrend_conditions:
            return MarketRegime.DOWNTREND

        # 3. Verificar Mercado Lateral
        # Preço oscilando entre as bandas de Bollinger
        in_bb_middle = (
                current['close'] > current['bb_lower'] and
                current['close'] < current['bb_upper'] and
                abs(current['close'] - current['bb_middle']) / current['bb_middle'] < 0.01
        )

        # ADX baixo indica falta de tendência
        adx_weak = current['adx'] < 20

        if in_bb_middle and adx_weak:
            return MarketRegime.RANGING

        # 4. Condições para mercado CHOPPY (perigoso)
        choppy_conditions = (
                current['adx'] < 15 or  # Tendência muito fraca
                current['atr'] / current['close'] > 0.02 or  # Volatilidade excessiva
                abs(current['rsi'] - 50) < 10  # RSI neutro demais
        )

        if choppy_conditions:
            return MarketRegime.CHOPPY

        # Default para RANGING se nenhuma condição for atendida
        return MarketRegime.RANGING

    def get_detailed_analysis(self, df: pd.DataFrame) -> Dict:
        """Retorna análise detalhada do mercado"""

        df_with_features = self.calculate_features(df)
        regime = self.classify_regime(df_with_features)

        current = df_with_features.iloc[-1]

        analysis = {
            'regime': regime,
            'adx': current['adx'],
            'adx_strength': 'Forte' if current['adx'] > 25 else 'Moderada' if current['adx'] > 15 else 'Fraca',
            'bb_width': current['bb_width'],
            'bb_position': 'Superior' if current['close'] > current['bb_upper'] * 0.99 else
            'Inferior' if current['close'] < current['bb_lower'] * 1.01 else 'Central',
            'atr': current['atr'],
            'atr_percent': (current['atr'] / current['close']) * 100,
            'trend_direction': 'Alta' if current['ema_fast'] > current['ema_slow'] else 'Baixa',
            'rsi': current['rsi'],
            'rsi_signal': 'Sobrevendido' if current['rsi'] < 30 else
            'Sobrecomprado' if current['rsi'] > 70 else 'Neutro'
        }

        return analysis