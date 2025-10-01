"""
Validação de horários de mercado usando IQ Option API
"""

import time
from typing import Dict, Optional
from datetime import datetime


class MarketHoursValidator:
    """Valida se os pares estão abertos usando a API da IQ Option"""

    def __init__(self, connector, logger=None):
        self.connector = connector
        self.logger = logger
        self._open_assets_cache = {}
        self._cache_timeout = 60  # 1 minuto de cache

    def is_asset_open(self, asset: str) -> bool:
        """
        Verifica se um ativo está aberto para trading
        Usa a API oficial da IQ Option
        """
        try:
            # Verificar cache primeiro
            current_time = time.time()
            if (asset in self._open_assets_cache and
                    current_time - self._open_assets_cache[asset]['timestamp'] < self._cache_timeout):
                return self._open_assets_cache[asset]['is_open']

            # Usar a API para verificar se o ativo está aberto
            if hasattr(self.connector.api, 'get_all_open_time'):
                all_open_times = self.connector.api.get_all_open_time()

                if asset in all_open_times:
                    is_open = all_open_times[asset]['open']

                    # Atualizar cache
                    self._open_assets_cache[asset] = {
                        'is_open': is_open,
                        'timestamp': current_time
                    }

                    if self.logger:
                        status = "ABERTO" if is_open else "FECHADO"
                        self.logger.debug(f"Ativo {asset}: {status}")

                    return is_open

            # Fallback: se não conseguir verificar, assume que está aberto
            if self.logger:
                self.logger.warning(f"Não foi possível verificar status do ativo {asset}, assumindo ABERTO")

            return True

        except Exception as e:
            if self.logger:
                self.logger.error(f"Erro ao verificar status do ativo {asset}: {e}")

            # Em caso de erro, assume que está aberto para não bloquear operações
            return True

    def get_asset_schedule(self, asset: str) -> Optional[Dict]:
        """
        Obtém informações do horário de funcionamento do ativo
        """
        try:
            if hasattr(self.connector.api, 'get_all_open_time'):
                all_open_times = self.connector.api.get_all_open_time()

                if asset in all_open_times:
                    return all_open_times[asset]

            return None

        except Exception as e:
            if self.logger:
                self.logger.error(f"Erro ao obter horário do ativo {asset}: {e}")
            return None

    def clear_cache(self):
        """Limpa o cache de ativos abertos"""
        self._open_assets_cache.clear()

    def get_cached_assets_status(self) -> Dict:
        """Retorna o status atual do cache"""
        return self._open_assets_cache