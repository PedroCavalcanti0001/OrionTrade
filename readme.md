OrionTrader: Bot Adaptativo de Day Trade para IQ Option
OrionTrader não é um simples bot de trading que segue regras fixas. É um sistema de análise e execução projetado para ser robusto e adaptativo, capaz de interpretar diferentes regimes de mercado e selecionar a estratégia mais apropriada para cada cenário em tempo real.

🧠 Filosofia Principal: Diagnosticar Antes de Agir
A maioria dos bots falha porque aplica a mesma estratégia (ex: seguir tendência) em todos os cenários de mercado. OrionTrader opera com uma filosofia diferente:

Diagnosticar: Primeiro, o bot utiliza um conjunto de "sensores" (indicadores de volatilidade, tendência e momentum) para classificar o estado atual do mercado (Regime).

Agir: Com base no diagnóstico, ele seleciona a ferramenta (estratégia) mais adequada de seu "arsenal". Se o mercado estiver perigoso ou indefinido, a ação mais adequada é não fazer nada.

📊 Diagrama da Arquitetura
Este diagrama ilustra o fluxo de dados e decisões dentro do bot, desde a análise do mercado até a execução da ordem.

Snippet de código

graph TD
    subgraph "Ambiente Externo"
        A[IQ Option API]
    end

    subgraph "Infraestrutura do Bot"
        B(Módulo de Conexão)
        C{{Banco de Dados de Mercado}}
        G((Módulo de Logging))
    end

    subgraph "Cérebro do Bot: Módulo de Análise e Estratégia"
        D1(<b>Camada 1:</b><br>Sensores de Mercado)
        D2{<b>Camada 2:</b><br>Classificador de Regime}
        
        subgraph D3 [<b>Camada 3:</b> Arsenal de Estratégias]
            direction LR
            D3_Trend[Estratégia:<br>Seguir Tendência]
            D3_Range[Estratégia:<br>Reversão à Média]
            D3_Squeeze[Estratégia:<br>Rompimento]
            D3_Filter[Filtro:<br>Não Operar]
        end

        D4(<b>Camada 4:</b><br>Lógica de Execução Dinâmica)
    end

    subgraph "Execução e Risco"
        E(Módulo de Gerenciamento de Risco)
        F(Execução de Ordens)
    end

    %% Conexões
    A -- Dados em Tempo Real --> B
    B -- Dados Brutos --> C & D1
    D1 -- Features (ADX, ATR, etc.) --> D2
    D2 -- "É TENDÊNCIA" --> D3_Trend --> D4
    D2 -- "É RANGE" --> D3_Range --> D4
    D2 -- "É SQUEEZE" --> D3_Squeeze --> D4
    D2 -- "É CHOPPY" --> D3_Filter
    D4 -- Sinal de COMPRA/VENDA --> E
    E -- Parâmetros de Risco --> F
    F -- Ordem Finalizada --> B --> A
    B & D4 & F -- Registros --> G
✨ Features Principais
Análise de Regime de Mercado: Classifica o mercado automaticamente em Tendência, Range, Squeeze ou Perigoso (Choppy).

Seleção de Estratégia Adaptativa: Ativa a estratégia correta para o regime identificado.

Gerenciamento de Risco Dinâmico: Calcula Stop Loss e Take Profit com base na volatilidade atual (ATR), não em valores fixos.

Arquitetura Modular: Código limpo e organizado em módulos independentes (Conexão, Análise, Risco, etc.).

Backtesting Integrado: Permite testar e validar estratégias com dados históricos.

Logging Completo: Registra todas as decisões, operações e possíveis erros para análise posterior.

🛠️ Stack de Tecnologias
Linguagem: Python 3.10+

Análise de Dados: Pandas

Indicadores Técnicos: Pandas TA

Conexão API: Um fork ativo e mantido da iqoptionapi

Banco de Dados: SQLite (padrão) / InfluxDB (opcional)

🚀 Começando
Siga estes passos para configurar e executar o bot em seu ambiente local.

Pré-requisitos
Python 3.10 ou superior

Git

Instalação
Clone o repositório:

Bash

git clone https://github.com/SEU_USUARIO/OrionTrader.git
cd OrionTrader
Crie e ative um ambiente virtual (Recomendado):

Bash

python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
Instale as dependências:

Bash

pip install -r requirements.txt
Configure suas credenciais:

Renomeie o arquivo de configuração de exemplo.

Bash

cp config.json.example config.json
Abra o arquivo config.json e insira seu email e senha da IQ Option. Comece SEMPRE com as credenciais da sua conta DEMO!

JSON

{
  "iq_option": {
    "email": "seu_email_demo@email.com",
    "password": "sua_senha_demo"
  },
  "trading_settings": {
    "asset": "EURUSD-OTC",
    "risk_percent": 1.5,
    "timeframe": 1
  }
}
▶️ Como Usar
O bot pode ser executado em três modos diferentes através da linha de comando.

Para executar em modo DEMONSTRAÇÃO (o padrão e o mais seguro):

Bash

python main.py --mode demo
Para executar em modo REAL (use com extrema cautela):

Bash

python main.py --mode live
Para executar um backtest com dados históricos:

Bash

python main.py --mode backtest --asset EURUSD --start_date 2025-09-01 --end_date 2025-09-30
🛑 AVISO IMPORTANTE E ISENÇÃO DE RESPONSABILIDADE 🛑
ALTO RISCO FINANCEIRO: Day trading é uma atividade de altíssimo risco. Você pode perder todo o seu capital. Este bot não é uma garantia de lucro.

API NÃO OFICIAL: Este software utiliza uma API não oficial, de engenharia reversa. Seu uso pode violar os Termos de Serviço da IQ Option e levar ao bloqueio da sua conta.

USE POR SUA CONTA E RISCO: Os desenvolvedores deste projeto não se responsabilizam por quaisquer perdas financeiras ou problemas com sua conta na corretora.

COMECE NA CONTA DEMO: É obrigatório testar exaustivamente o bot em uma conta de demonstração por um período prolongado antes de considerar o uso com dinheiro real.

🗺️ Roadmap
[ ] Adicionar mais estratégias ao "Arsenal".

[ ] Integrar modelos de Machine Learning para previsão de regime.

[ ] Implementar notificações via Telegram para operações e alertas.

[ ] Desenvolver um painel de controle (Dashboard) web para monitoramento.

🤝 Contribuições
Contribuições são bem-vindas! Sinta-se à vontade para abrir uma Issue para relatar bugs ou sugerir novas features.

📄 Licença
Distribuído sob a Licença MIT. Veja LICENSE para mais informações.