"""
Módulo de conexão com a IQ Option
"""

import os
import time
from typing import Dict, List, Optional, Tuple
from iqoptionapi.stable_api import IQ_Option

class IQConnector:
    """Conector para API real da IQ Option"""

    def __init__(self, email: str, password: str, account_type: str = "PRACTICE", logger=None):
        self.logger = logger
        self.email = email
        self.password = password
        self.account_type = account_type
        self.api = None
        self.connected = False

    def connect(self) -> bool:
        """Estabelece conexão com a IQ Option - Versão Corrigida"""
        try:
            self.logger.info("Conectando à IQ Option...")
            self.api = IQ_Option(self.email, self.password)

            # Conectar à API
            check, reason = self.api.connect()

            if not check:
                self.logger.error(f"Falha na conexão: {reason}")
                return False

            # Mudar para conta demo ou real
            if self.account_type == "REAL":
                self.api.change_balance("REAL")
            else:
                self.api.change_balance("PRACTICE")

            # CONFIGURAÇÃO PARA EVITAR ERROS DE DIGITAL OPTIONS ← ADICIONE ESTE BLOCO
            try:
                # Desativar funcionalidades de digital options se não forem necessárias
                self.api.subscribe_strike_list = lambda *args, **kwargs: None
            except:
                pass

            self.connected = True
            self.logger.info("Conexão estabelecida com sucesso")
            return True

        except Exception as e:
            self.logger.error(f"Erro na conexão: {e}")
            return False

    def get_candles(self, asset: str, interval: int, count: int) -> List[Dict]:
        if not self.connected:
            self.logger.error("Não conectado à API")
            return []

        try:
            candles = self.api.get_candles(asset, interval, count, time.time())

            # Verificar e garantir a estrutura dos dados
            if candles:
                for candle in candles:
                    # Garantir que todas as colunas necessárias existem
                    required_fields = ['open', 'high', 'low', 'close', 'volume']
                    for field in required_fields:
                        if field not in candle:
                            candle[field] = candle.get('close', 1.0)  # Valor padrão

            return candles if candles else []
        except Exception as e:
            self.logger.error(f"Erro ao obter candles: {e}")
            return []

    def get_balance(self) -> float:
        """Obtém saldo atual da conta"""
        if not self.connected:
            return 0.0

        try:
            return self.api.get_balance()
        except Exception as e:
            self.logger.error(f"Erro ao obter saldo: {e}")
            return 0.0

    def place_order(self, asset: str, direction: str, amount: float,
                    expiration: int = 1) -> Optional[int]:
        """Coloca uma ordem de trade"""
        if not self.connected:
            self.logger.error("Não conectado à API")
            return None

        try:
            # direction: "call" ou "put"
            result, order_id = self.api.buy(amount, asset, direction, expiration)

            if result:
                self.logger.info(f"Ordem executada: {order_id}")
                return order_id
            else:
                self.logger.error(f"Falha na ordem: {order_id}")
                return None

        except Exception as e:
            self.logger.error(f"Erro ao colocar ordem: {e}")
            return None

    def check_win(self, order_id: int) -> Tuple[bool, float]:
        """Verifica resultado de uma ordem - VERSÃO CORRIGIDA"""
        if not self.connected:
            return False, 0.0

        try:
            # Usar check_win_v4 que retorna (resultado_str, pnl)
            status, pnl = self.api.check_win_v4(order_id)

            # Se status for None, a trade ainda está aberta
            if status is None:
                return False, 0.0

            # Se houver um status ('win', 'loose', 'equal'), a trade está fechada
            return True, pnl

        except Exception as e:
            self.logger.error(f"Erro ao verificar resultado da ordem {order_id}: {e}")
            return False, 0.0

    def disconnect(self):
        """Desconecta da API"""
        if self.connected and self.api:
            self.api.close()
            self.connected = False
            self.logger.info("Desconectado da IQ Option")