# file: bot/analysis/strategies.py
import pandas as pd
import numpy as np


def get_mean_reversion_signals(df: pd.DataFrame) -> pd.Series:
    """
    Gera sinais de Reversão à Média de forma vetorizada.
    Retorna uma Series com sinais 'ENTER_LONG', 'ENTER_SHORT' ou 'WAIT'.
    """
    # Condição 1: Preço toca ou excede a banda
    is_overbought = df['close'] >= df['bb_upper']
    is_oversold = df['close'] <= df['bb_lower']

    # Condição 2: Confirmação de Momentum (vela na direção da reversão)
    bullish_candle = df['close'] > df['open']
    bearish_candle = df['close'] < df['open']

    # Condição 3: Pico/Vale de Momentum (máxima/mínima anterior foi mais forte)
    peak_momentum_lost = df['high'] < df['high'].shift(1)
    valley_momentum_lost = df['low'] > df['low'].shift(1)

    # Combinar condições
    long_signal = is_oversold & bullish_candle & valley_momentum_lost
    short_signal = is_overbought & bearish_candle & peak_momentum_lost

    return np.select(
        [long_signal, short_signal],
        ['ENTER_LONG', 'ENTER_SHORT'],
        default='WAIT'
    )


def get_trend_following_signals(df: pd.DataFrame) -> pd.Series:
    """
    Gera sinais de Seguimento de Tendência de forma vetorizada.
    Retorna uma Series com sinais 'ENTER_LONG', 'ENTER_SHORT' ou 'WAIT'.
    """
    # Condição 1: RSI em zona de pullback (não extremo)
    pullback_zone_long = df['rsi'].between(35, 68)
    pullback_zone_short = df['rsi'].between(32, 65)

    # Condição 2: Confirmação de Momentum (vela na direção da tendência)
    bullish_candle = df['close'] > df['open']
    bearish_candle = df['close'] < df['open']

    # Combinar condições
    long_signal = pullback_zone_long & bullish_candle
    short_signal = pullback_zone_short & bearish_candle

    return np.select(
        [long_signal, short_signal],
        ['ENTER_LONG', 'ENTER_SHORT'],
        default='WAIT'
    )