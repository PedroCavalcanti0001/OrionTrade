# OrionTrader - Sistema Adaptativo de Trading Algorítmico

![OrionTrader](https://img.shields.io/badge/Status-Desenvolvimento-yellow)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Visão Geral

O **OrionTrader** é um sistema de trading algorítmico avançado que implementa uma filosofia adaptativa de "Diagnosticar e Agir". Ao invés de usar estratégias fixas, o bot analisa continuamente o regime de mercado e seleciona dinamicamente a estratégia mais apropriada.

## 🎯 Filosofia: Sistema Adaptativo de Troca de Regime

### Diagnóstico de Mercado
O sistema classifica o mercado em 5 regimes distintos:
- **UPTREND**: Tendência de alta forte
- **DOWNTREND**: Tendência de baixa forte  
- **RANGING**: Mercado lateral
- **SQUEEZE**: Compressão de volatilidade (pré-rompimento)
- **CHOPPY**: Mercado indefinido/perigoso (não opera)

### Arsenal de Estratégias
Cada regime ativa uma estratégia específica:
- **Tendência**: Estratégia de seguir tendência com entrada em pullbacks
- **Lateral**: Estratégia de reversão à média
- **Squeeze**: Modo observação (aguarda rompimento)
- **Choppy**: Filtro de proteção (não opera)

## 🏗️ Arquitetura do Sistema

```mermaid
graph TB
    A[OrionTrader] --> B[Conexão]
    A --> C[Análise]
    A --> D[Risco]
    A --> E[Execução]
    
    B --> B1[IQConnector]
    B --> B2[MockConnector]
    
    C --> C1[MarketRegimeClassifier]
    C --> C2[StrategySelector]
    
    C1 --> C1a[Sensores/Features]
    C1 --> C1b[Classificador]
    
    C2 --> C2a[Trend Following]
    C2 --> C2b[Mean Reversion]
    C2 --> C2c[Breakout Watch]
    
    D --> D1[Position Sizing]
    D --> D2[SL/TP Dinâmico]
    D --> D3[Risk Filters]