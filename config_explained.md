# 📊 Configurações do OrionTrader

## TRADING
- `assets`: Pares que o bot irá monitorar e operar
- `timeframe`: 1 = candles de 1 minuto (análise mais rápida)
- `initial_balance`: Saldo inicial para cálculo de position sizing  
- `risk_per_trade`: 2.0% = Risco de 2% do saldo por trade
- `max_open_trades`: 5 trades simultâneas no máximo
- `max_trades_per_asset`: 2 trades por par específico
- `stop_loss_atr_multiplier`: 1.0 = Stop Loss de 1x o ATR
- `take_profit_atr_multiplier`: 1.5 = Take Profit de 1.5x o ATR
- `min_confidence`: 0.4 = Aceita sinais com 40% de confiança

## ANÁLISE (Otimizado para mais entradas)
- `adx_period`: 10 períodos para ADX (mais sensível)
- `adx_threshold`: 20 = ADX mínimo para tendência
- `bb_period`: 14 períodos para Bollinger Bands
- `bb_std`: 1.5 desvios padrão (bandas mais apertadas)
- `bb_squeeze_threshold`: 0.8% de largura para detectar squeeze
- `atr_period`: 10 períodos para ATR
- `ema_fast`: EMA de 7 períodos (rápida)
- `ema_slow`: EMA de 14 períodos (lenta)
- `rsi_period`: 10 períodos para RSI
- `rsi_overbought`: 65 = Limite de sobrecompra
- `rsi_oversold`: 35 = Limite de sobrevenda

## EXECUÇÃO (Mais rápida)
- `check_interval`: 5 segundos entre análises
- `max_retries`: 3 tentativas se ordem falhar
- `retry_delay`: 2 segundos entre tentativas
- `timeout`: 15 segundos máximo para API

## RESULTADO ESPERADO:
- ✅ 3-5x mais trades que a configuração anterior
- ✅ Resposta mais rápida a oportunidades
- ✅ Trades mais curtas (1-5 minutos)
- ✅ Menos filtros, mais sinais aceitos