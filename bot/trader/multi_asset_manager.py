"""
Gerenciador de Múltiplos Ativos
"""

import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

from bot.analysis.regime_classifier import MarketRegime


class MultiAssetManager:
    """Gerencia análise e trading em múltiplos ativos simultaneamente"""

    def __init__(self, connector, strategy_selector, risk_manager, config: Dict, logger):
        self.connector = connector
        self.strategy_selector = strategy_selector
        self.risk_manager = risk_manager
        self.config = config
        self.logger = logger

        self.trading_config = config.get('trading', {})
        self.assets = self.trading_config.get('assets', ['EURUSD'])
        self.max_trades_per_asset = self.trading_config.get('max_trades_per_asset', 1)

        # Estado por ativo
        self.asset_states = {}
        self.initialize_asset_states()

    def initialize_asset_states(self):
        """Inicializa o estado para cada ativo"""
        for asset in self.assets:
            self.asset_states[asset] = {
                'open_trades': 0,
                'last_analysis': None,
                'last_signal': None,
                'consecutive_choppy': 0,
                'priority_score': 0.0
            }

    def get_market_data_for_asset(self, asset: str, count: int = 100) -> pd.DataFrame:
        try:
            timeframe = self.trading_config.get('timeframe', 60)

            candles = self.connector.get_candles(asset, timeframe, count)

            if not candles:
                self.logger.debug(f"Nenhum candle obtido para {asset}")
                return pd.DataFrame()

            # Converter para DataFrame
            df = pd.DataFrame(candles)

            # VERIFICAÇÃO CRÍTICA: garantir colunas necessárias
            required_columns = ['open', 'high', 'low', 'close', 'volume']

            # Verificar se as colunas existem com nomes alternativos
            column_mapping = {
                'min': 'low',
                'max': 'high'
            }

            for old_name, new_name in column_mapping.items():
                if old_name in df.columns and new_name not in df.columns:
                    df[new_name] = df[old_name]

            # Garantir que todas as colunas necessárias existem
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.error(f"Colunas faltando para {asset}: {missing_columns}")
                return pd.DataFrame()

            # Processar timestamp
            if 'timestamp' not in df.columns and 'from' in df.columns:
                df['timestamp'] = pd.to_datetime(df['from'], unit='s')
            elif 'timestamp' not in df.columns:
                # Criar timestamp sequencial se não existir
                df['timestamp'] = pd.date_range(
                    start=datetime.now(),
                    periods=len(df),
                    freq=f'{timeframe}min'
                )

            # Garantir que os dados são numéricos
            for col in required_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remover NaNs
            df = df.dropna()

            if df.empty:
                self.logger.warning(f"Dados insuficientes para {asset} após limpeza")
                return pd.DataFrame()

            df.set_index('timestamp', inplace=True)
            return df

        except Exception as e:
            self.logger.error(f"Erro ao obter dados para {asset}: {e}")
            return pd.DataFrame()

    def analyze_all_assets(self) -> Dict[str, Dict]:
        """Analisa todos os ativos e retorna sinais - VERSÃO CORRIGIDA"""
        all_signals = {}

        for asset in self.assets:
            try:
                # Obter dados do ativo
                df = self.get_market_data_for_asset(asset)

                # ✅ VERIFICAÇÃO ROBUSTA
                if df is None or df.empty:
                    self.logger.debug(f"Dados vazios para {asset}, pulando...")
                    continue

                # Gerar sinal
                signal = self.strategy_selector.get_trading_signal(df)

                # ✅ VERIFICAÇÃO DO SINAL
                if signal is None:
                    self.logger.debug(f"Sinal None para {asset}, pulando...")
                    continue

                signal['asset'] = asset

                # Calcular prioridade
                priority_score = self.calculate_priority_score(signal, asset)
                signal['priority_score'] = priority_score

                # Atualizar estado do ativo
                self.update_asset_state(asset, signal)

                all_signals[asset] = signal

                # ✅ CORREÇÃO: Usar logger.debug diretamente (sem isEnabledFor)
                self.logger.debug(f"Análise {asset}: {signal['regime'].value} - Prioridade: {priority_score:.2f}")

            except Exception as e:
                self.logger.error(f"Erro ao analisar {asset}: {e}")
                continue

        return all_signals

    def calculate_priority_score(self, signal: Dict, asset: str) -> float:
        """Calcula pontuação de prioridade para o ativo - VERSÃO CORRIGIDA"""
        score = 0.0

        # Baseado no regime
        regime_scores = {
            'UPTREND': 0.9,
            'DOWNTREND': 0.9,
            'RANGING': 0.7,
            'SQUEEZE': 0.8,  # SQUEEZE tem boa prioridade
            'CHOPPY': 0.1
        }

        regime = signal['regime'].value
        score += regime_scores.get(regime, 0.5)

        # Baseado na confiança do sinal - USAR A CONFIANÇA REAL AGORA
        confidence = signal.get('confidence', 0.0)
        score += confidence * 0.5  # Peso maior para confiança

        # Baseado no número de trades abertas no ativo
        open_trades = self.asset_states[asset]['open_trades']
        max_per_asset = self.config['trading'].get('max_trades_per_asset', 1)
        if open_trades >= max_per_asset:
            score *= 0.1  # Reduz drasticamente se já tem trades abertas

        # Baseado no RSI (evitar extremos)

        # ✅ CORREÇÃO: Lógica do RSI agora é sensível ao contexto do regime de mercado
        rsi = signal.get('analysis', {}).get('rsi', 50)
        regime = signal.get('regime')

        # Em tendência, um RSI não extremo é preferível para evitar reversões
        if regime in [MarketRegime.UPTREND, MarketRegime.DOWNTREND]:
            if 40 <= rsi <= 60:
                score *= 1.1  # Bonifica RSI neutro na tendência
            elif rsi > 70 or rsi < 30:
                score *= 0.8  # Penaliza levemente RSI extremo

        # Em mercado lateral, um RSI extremo é um sinal de entrada, então é bonificado
        elif regime == MarketRegime.RANGING:
            if rsi > 70 or rsi < 30:
                score *= 1.2  # Bonifica RSI extremo para reversão

        # Baseado na volatilidade (ATR)
        atr_percent = signal.get('analysis', {}).get('atr_percent', 0)
        if 0.05 <= atr_percent <= 0.3:  # Volatilidade ideal
            score *= 1.2
        elif atr_percent > 0.5:  # Muito volátil
            score *= 0.7

        return min(score, 1.0)

    def update_asset_state(self, asset: str, signal: Dict):
        """Atualiza o estado do ativo"""
        state = self.asset_states[asset]
        state['last_analysis'] = datetime.now()
        state['last_signal'] = signal

        # Contar CHOPPY consecutivos
        if signal['regime'].value == 'CHOPPY':
            state['consecutive_choppy'] += 1
        else:
            state['consecutive_choppy'] = 0

        # Atualizar pontuação de prioridade
        state['priority_score'] = signal.get('priority_score', 0.0)

    def get_top_signals(self, all_signals: Dict, limit: int = 3) -> List[Dict]:
        """Retorna os melhores sinais baseado na prioridade"""
        if not all_signals:
            return []

        # Filtrar sinais válidos para entrada
        min_confidence = self.trading_config.get('min_confidence', 0.5)
        valid_signals = []
        for asset, signal in all_signals.items():
            if (signal['action'] == 'ENTER' and
                    self.asset_states[asset]['open_trades'] < self.max_trades_per_asset and
                    signal.get('confidence', 0) >= min_confidence):
                valid_signals.append(signal)

        # Ordenar por prioridade
        valid_signals.sort(key=lambda x: x.get('priority_score', 0), reverse=True)

        return valid_signals[:limit]

    def update_trade_count(self, asset: str, change: int):
        """Atualiza contador de trades para um ativo"""
        if asset in self.asset_states:
            self.asset_states[asset]['open_trades'] += change
            self.asset_states[asset]['open_trades'] = max(0, self.asset_states[asset]['open_trades'])

    def get_asset_summary(self) -> Dict:
        """Retorna resumo do estado de todos os ativos"""
        summary = {}
        for asset, state in self.asset_states.items():
            summary[asset] = {
                'open_trades': state['open_trades'],
                'priority_score': state['priority_score'],
                'consecutive_choppy': state['consecutive_choppy'],
                'last_signal': state['last_signal']['regime'].value if state['last_signal'] else 'N/A'
            }
        return summary

    def should_skip_asset(self, asset: str) -> bool:
        """Verifica se deve pular análise de um ativo"""
        state = self.asset_states.get(asset, {})

        # Pular se já tem muitas trades abertas
        if state.get('open_trades', 0) >= self.max_trades_per_asset:
            return True

        # Pular se está muito tempo em CHOPPY
        if state.get('consecutive_choppy', 0) > 10:  # 10 ciclos consecutivos
            return True

        return False