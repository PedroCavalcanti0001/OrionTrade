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
        """Simula colocação de ordem"""
        if not self.connected:
            return None

        if amount > self.balance:
            self.logger.error("Saldo insuficiente")
            return None

        self.order_counter += 1
        order_id = self.order_counter

        # Simular resultado baseado em probabilidade
        win_probability = 0.6  # 60% de chance de ganhar na simulação
        is_win = random.random() < win_probability

        self.open_orders[order_id] = {
            'asset': asset,
            'direction': direction,
            'amount': amount,
            'is_win': is_win,
            'timestamp': time.time()
        }

        self.balance -= amount

        self.logger.info(f"Ordem mock {order_id} colocada: {direction} ${amount}")
        return order_id

    def check_win(self, order_id: int) -> Optional[bool]:
        """Simula verificação de resultado"""
        if order_id not in self.open_orders:
            return None

        order = self.open_orders[order_id]
        is_win = order['is_win']

        # Simular payout
        payout_multiplier = 0.8  # 80% de payout na simulação

        if is_win:
            win_amount = order['amount'] * (1 + payout_multiplier)
            self.balance += win_amount
            self.logger.info(f"Ordem mock {order_id}: GANHOU +${win_amount - order['amount']:.2f}")
        else:
            self.logger.info(f"Ordem mock {order_id}: PERDEU -${order['amount']:.2f}")

        del self.open_orders[order_id]
        return is_win

    def disconnect(self):
        """Simula desconexão"""
        self.connected = False
        self.logger.info("Desconectado do Mock Connector")