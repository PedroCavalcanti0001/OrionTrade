# file: bot/connection/mock_connector.py
import os
import time
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

class MockConnector:
    """
    Conector para backtesting que lê dados locais e baixa-os automaticamente se não existirem.
    """

    def __init__(self, config: Dict, logger, api_connector=None):
        self.logger = logger
        self.config = config
        self.trading_config = config.get('trading', {})
        self.backtest_config = config.get('backtest', {})

        # Conector real, usado apenas para baixar dados se necessário
        self.api_connector = api_connector

        self.initial_balance = self.trading_config.get('initial_balance', 1000.0)
        self.balance = self.initial_balance
        self.connected = False
        self.open_orders = {}
        self.order_counter = 0

        self.assets_to_test = self.backtest_config.get('assets', [])
        self.timeframe = self.trading_config.get('timeframe', 1)
        self.historical_data_frames = {}
        self.current_index = 0
        self.max_index = 0
        self.data_dir = 'historical_data'

    def connect(self) -> bool:
        """Verifica, baixa (se necessário) e carrega os dados históricos."""
        self.logger.info("Iniciando conector de backtest com download 'embutido'...")

        if not self.assets_to_test:
            self.logger.error("Nenhum ativo configurado para o backtest em 'config.json'.")
            return False

        try:
            os.makedirs(self.data_dir, exist_ok=True)

            # Para cada ativo, verificar se os dados locais existem. Se não, baixar.
            for asset in self.assets_to_test:
                file_path = os.path.join(self.data_dir, f"{asset}_M{self.timeframe}.csv")

                if not os.path.exists(file_path):
                    self.logger.warning(f"Arquivo de dados '{file_path}' não encontrado.")
                    if not self._fetch_data_for_asset(asset):
                        return False # Falha no download impede o início do backtest

                # Carregar dados do arquivo CSV
                df = pd.read_csv(file_path)
                self.historical_data_frames[asset] = df.to_dict('records')
                self.logger.info(f"Dados para '{asset}' carregados com sucesso ({len(df)} candles).")

            # Sincronizar os dados e preparar para a simulação
            min_length = min(len(data) for data in self.historical_data_frames.values())
            if min_length < 200:
                self.logger.error(f"Dados insuficientes para o backtest (mínimo de 200 candles). Encontrado: {min_length}")
                return False

            self.max_index = min_length - 2 # -2 para garantir que o candle de resultado sempre exista
            self.current_index = 200

            self.connected = True
            self.logger.info(f"Backtest pronto para iniciar. Total de {self.max_index - self.current_index} candles para simulação.")
            return True

        except Exception as e:
            self.logger.error(f"Erro crítico ao preparar dados para o backtest: {e}", exc_info=True)
            return False

    def _fetch_data_for_asset(self, asset: str) -> bool:
        """Usa o conector real para baixar e salvar os dados de um ativo."""
        if not self.api_connector:
            self.logger.error("Conector real da API não fornecido para download automático de dados.")
            return False

        try:
            days_to_download = self.backtest_config.get('days_to_download', 30)
            timeframe_seconds = self.timeframe * 60
            candles_per_call = 1000

            total_candles_needed = (days_to_download * 24 * 60) / self.timeframe
            num_calls = int(total_candles_needed / candles_per_call) + 1

            self.logger.info(f"Iniciando download automático de {days_to_download} dias de dados para {asset}...")

            end_time = time.time()
            all_candles = []

            for i in range(num_calls):
                # ✅ LOG MELHORADO: Alterado de DEBUG para INFO e com mais detalhes
                current_period_str = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d')
                self.logger.info(
                    f"[Download {asset}] Buscando lote {i + 1}/{num_calls}... (Período a partir de ~{current_period_str})")

                candles = self.api_connector.api.get_candles(asset, timeframe_seconds, candles_per_call, end_time)

                if not candles:
                    self.logger.warning(f"Fim dos dados históricos para {asset} alcançado no lote {i + 1}.")
                    break

                new_candles = [c for c in candles if c['from'] not in [ac.get('from') for ac in all_candles]]
                all_candles.extend(new_candles)

                if not new_candles:
                    self.logger.info("Nenhum candle novo encontrado no lote. Encerrando a coleta para este ativo.")
                    break

                end_time = candles[0]['from']
                time.sleep(1.1)  # Pausa para não sobrecarregar a API

            if not all_candles:
                self.logger.error(f"Nenhum dado foi baixado para {asset}.")
                return False

            df = pd.DataFrame(all_candles).drop_duplicates(subset=['from']).sort_values(by='from', ascending=True)
            file_path = os.path.join(self.data_dir, f"{asset}_M{self.timeframe}.csv")
            df.to_csv(file_path, index=False)

            self.logger.info(f"Download para {asset} concluído! {len(df)} candles salvos.")
            return True
        except Exception as e:
            self.logger.error(f"Falha no download automático para {asset}: {e}", exc_info=True)
            return False

    # Os métodos get_candles, check_win, tick, place_order, get_balance, e disconnect
    # permanecem exatamente como na versão anterior, pois a lógica de simulação
    # sobre os dados carregados está correta. Vou incluí-los aqui para o arquivo ficar completo.

    def get_candles(self, asset: str, interval: int, count: int) -> List[Dict]:
        if not self.connected or asset not in self.historical_data_frames:
            return []
        start_index = max(0, self.current_index - count)
        end_index = self.current_index
        return self.historical_data_frames[asset][start_index:end_index]

    def check_win(self, order_id: int) -> tuple:
        if order_id not in self.open_orders:
            return False, 0
        order = self.open_orders[order_id]
        asset = order['asset']
        entry_index = order['entry_index']
        expiration_candles = order['expiration_candles']
        result_index = entry_index + expiration_candles

        if result_index >= self.current_index:
            return False, 0

        if result_index < len(self.historical_data_frames[asset]):
            entry_price = order['entry_price']
            result_price = self.historical_data_frames[asset][result_index]['close']
            payout_multiplier = 0.87
            is_win = False
            if order['direction'] == 'call' and result_price > entry_price:
                is_win = True
            elif order['direction'] == 'put' and result_price < entry_price:
                is_win = True
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
        del self.open_orders[order_id]
        return True, -order['amount']

    def tick(self):
        if self.current_index < self.max_index:
            self.current_index += 1
            return True
        self.logger.info("Fim dos dados históricos para o backtest.")
        return False

    def place_order(self, asset: str, direction: str, amount: float, expiration: int = 1) -> Optional[int]:
        if not self.connected or amount > self.balance:
            return None
        self.order_counter += 1
        order_id = self.order_counter
        self.open_orders[order_id] = {
            'asset': asset, 'direction': direction, 'amount': amount,
            'entry_index': self.current_index,
            'entry_price': self.historical_data_frames[asset][self.current_index]['close'],
            'expiration_candles': expiration,
        }
        self.logger.debug(f"Ordem mock {order_id}: {direction} ${amount} no candle {self.current_index} para {asset}")
        return order_id

    def get_balance(self) -> float:
        return self.balance

    def disconnect(self):
        self.connected = False
        self.logger.info("Conector de Backtest desconectado.")