import os
import logging
import time
from iqoptionapi.stable_api import IQ_Option

# Configura o logging para ver as mensagens da API
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def run_bot():
    """
    Função principal que executa o bot de teste.
    """
    # --- 1. Obter Credenciais das Variáveis de Ambiente ---
    email = os.getenv('IQ_EMAIL')
    senha = os.getenv('IQ_PASSWORD')

    if not email or not senha:
        print("Erro: As variáveis de ambiente IQ_EMAIL e IQ_PASSWORD não foram definidas.")
        print("Por favor, configure-as antes de executar o script.")
        return

    print("--- Iniciando Bot de Teste da IQ Option API ---")

    # --- 2. Conectar à API ---
    # Você pode precisar ajustar o host se a conexão falhar. Ex: "wss://ws.eu.iqoption.com/echo/websocket"
    api = IQ_Option(email, senha)
    conectado, razao = api.connect()

    if not conectado:
        print(f"Falha na conexão. Razão: {razao}")
        return

    print("✅ Conectado com sucesso!")

    try:
        # --- 3. Mudar para a Conta de Treinamento (MUITO IMPORTANTE) ---
        print("🔄 Alterando para a conta de TREINAMENTO (PRACTICE)...")
        api.change_balance('PRACTICE')

        # --- 4. Obter e Exibir o Saldo ---
        saldo = api.get_balance()
        print(f"💰 Saldo da conta de treinamento: ${saldo:,.2f}")

        # --- 5. Obter Dados de um Ativo ---
        ativo = "GBPUSD-OTC"
        print(f"📈 Obtendo dados para o ativo: {ativo}...")

        # Obtém as últimas velas para verificar o preço
        velas = api.get_candles(ativo, 60, 5, time.time())
        if velas:
            preco_atual = velas[-1]['close']
            print(f"📊 Preço de fechamento da última vela de 1 min: {preco_atual}")
        else:
            print(f"Não foi possível obter dados para {ativo}. Talvez o mercado esteja fechado.")
            # Para testes fora do horário de mercado, use ativos OTC. Ex: "EURUSD-OTC"
            ativo = "GBPUSD-OTC"
            print(f"Tentando com o ativo {ativo}...")

        # --- 6. Realizar uma Operação de Teste ---
        valor_entrada = 1  # Valor em dólares para a operação
        direcao = "call"  # "call" (sobe) ou "put" (desce)
        expiracao = 1  # Duração em minutos

        print(
            f"🤖 Realizando uma operação de teste: {direcao} de ${valor_entrada} em {ativo} com expiração de {expiracao} minuto(s)...")

        check, order_id = api.buy(valor_entrada, ativo, direcao, expiracao)

        if check:
            print(f"✅ Operação de teste enviada com sucesso! ID do pedido: {order_id}")
            # Espera o resultado (opcional)
            resultado = api.check_win_v4(order_id)
            print(f"🤑 Resultado da operação: Lucro de ${resultado[1]:,.2f}")
        else:
            print("❌ Falha ao enviar a operação de teste.")

    except Exception as e:
        print(f"Ocorreu um erro durante a execução: {e}")

    finally:
        # --- 7. Desconectar ---
        print("--- Desconectando da API ---")
        api.disconnect()


if __name__ == "__main__":
    run_bot()