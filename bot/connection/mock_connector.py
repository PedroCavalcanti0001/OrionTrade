"""
Conector Mock para testes e desenvolvimento
"""

import random
import time
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from bot.utils.logger import setup_logger


class MockConnector:
    """Conector mock que simula dados de mercado"""

    def __init__(self, initial_balance: float = 1000.0):
        self.logger = setup_logger()
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.connected = False
        self.open_orders = {}
        self.order_counter = 0

        # Gerar dados históricos iniciais
        self._generate_historical_data()

    def _generate_historical_data(self):
        """Gera dados históricos realistas para simulação"""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='1min')

        # Preços base com tendência e volatilidade variável
        base_price = 1.1000
        returns = np.random.normal(0, 0.0005, 1000)
        prices = base_price * (1 + np.cumsum(returns))

        # Adicionar alguma sazonalidade
        for i in range(len(prices)):
            if i % 100 < 50:  # Tendência de alta periódica
                prices[i] *= (1 + 0.001 * (i % 100))
            else:  # Tendência de baixa periódica
                prices[i] *= (1 - 0.001 * (i % 50))

        self.historical_data = []
        for i, date in enumerate(dates):
            price = prices[i]
            self.historical_data.append({
                'timestamp': int(date.timestamp()),
                'open': price * (1 + np.random.normal(0, 0.0001)),
                'high': price * (1 + abs(np.random.normal(0, 0.0002))),
                'low': price * (1 - abs(np.random.normal(0, 0.0002))),
                'close': price,
                'volume': random.randint(1000, 10000)
            })

        self.current_index = 500  # Começar no meio dos dados

    def connect(self) -> bool:
        """Simula conexão bem-sucedida"""
        self.logger.info("Conectando ao Mock Connector...")
        time.sleep(1)
        self.connected = True
        self.logger.info("Conexão mock estabelecida com sucesso")
        return True

    def get_candles(self, asset: str, interval: int, count: int) -> List[Dict]:
        """Retorna candles simulados"""
        if not self.connected:
            return []

        candles = []
        for i in range(count):
            idx = (self.current_index + i) % len(self.historical_data)
            candle = self.historical_data[idx].copy()

            # Garantir que todas as colunas necessárias existem
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in candle:
                    # Valores padrão se alguma coluna faltar
                    if col == 'high':
                        candle['high'] = max(candle.get('open', 1.0), candle.get('close', 1.0)) * 1.001
                    elif col == 'low':
                        candle['low'] = min(candle.get('open', 1.0), candle.get('close', 1.0)) * 0.999
                    elif col == 'volume':
                        candle['volume'] = 1000

            candles.append(candle)

        self.current_index = (self.current_index + 1) % len(self.historical_data)
        return candles

    def get_balance(self) -> float:
        """Retorna saldo simulado"""
        return self.balance

    def place_order(self, asset: str, direction: str, amount: float,
                    expiration: int = 1) -> Optional[int]:
        """Simula colocação de ordem registrando o candle de entrada - ✅ CORREÇÃO"""
        if not self.connected:
            return None

        if amount > self.balance:
            self.logger.error("Saldo insuficiente")
            return None

        self.order_counter += 1
        order_id = self.order_counter

        # Armazena o índice do candle em que a ordem foi aberta
        self.open_orders[order_id] = {
            'asset': asset,
            'direction': direction,
            'amount': amount,
            'entry_index': self.current_index,
            'entry_price': self.historical_data[self.current_index]['close'],
            'expiration_candles': expiration  # Expiração em número de candles
        }

        # Não deduzir o saldo ainda, apenas no resultado
        self.logger.info(f"Ordem mock {order_id} colocada: {direction} ${amount} no candle {self.current_index}")
        return order_id

    def check_win(self, order_id: int) -> tuple:
        """Simula verificação de resultado baseado nos dados históricos - ✅ CORREÇÃO"""
        if order_id not in self.open_orders:
            return False, 0  # Trade não encontrada

        order = self.open_orders[order_id]

        entry_index = order['entry_index']
        expiration_candles = order['expiration_candles']

        # O candle de resultado é o candle seguinte à expiração
        result_index = entry_index + expiration_candles

        # Se o candle de resultado ainda não "aconteceu" na simulação, a trade não fechou
        if result_index >= self.current_index:
            return False, 0  # Trade ainda está aberta

        # Se o candle de resultado existe nos dados históricos
        if result_index < len(self.historical_data):
            entry_price = order['entry_price']
            result_price = self.historical_data[result_index]['close']

            payout_multiplier = 0.8  # 80% de payout
            is_win = False

            if order['direction'] == 'call' and result_price > entry_price:
                is_win = True
            elif order['direction'] == 'put' and result_price < entry_price:
                is_win = True

            # Atualizar saldo
            if is_win:
                profit = order['amount'] * payout_multiplier
                self.balance += profit
                del self.open_orders[order_id]
                return True, profit
            else:
                loss = -order['amount']
                self.balance += loss
                del self.open_orders[order_id]
                return True, loss

        # Se não há dados suficientes, considera como perda
        del self.open_orders[order_id]
        return True, -order['amount']  # Trade fechada (sem dados), resultado negativo

    def tick(self):
        """Avança a simulação em um passo (1 candle)."""
        if self.current_index < len(self.historical_data) - 1:
            self.current_index += 1
            return True
        return False # Fim dos dados históricos

    def disconnect(self):
        """Simula desconexão"""
        self.connected = False
        self.logger.info("Desconectado do Mock Connector")