import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from src.data.data_fetcher import update_all_data, get_mercado_status
from src.features.feature_builder import gerar_features_para_modelo
from src.ui.cli import fazer_perguntas
from src.models.predict import train_model, load_model, predict_scores, BackpropCalibrator
from src.models.optimization import otimizar_escalacao
import pandas as pd

def main():
    """Ponto de entrada principal da aplicação."""
    load_dotenv()

    # Menu inicial para o usuário
    print("Bem-vindo ao Cartolando-AI!")
    print("O que você gostaria de fazer?")
    print("  [1] Somente atualizar todos os dados do banco.")
    print("  [2] Atualizar os dados e iniciar a escalação do time.")
    print("  [3] Escalar time (usando dados existentes).")
    
    escolha = input("Escolha uma opção: ")

    # Passo 1: Atualização dos dados (se aplicável)
    if escolha in ['1', '2']:
        print("\nIniciando atualização dos dados...")
        update_all_data()
        if escolha == '1':
            print("\nDados atualizados com sucesso. Encerrando.")
            return
    elif escolha != '3':
        print("Opção inválida. Encerrando.")
        return

    # Conexão com o banco deve ser feita antes de qualquer fluxo
    try:
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST")
        db_port = os.getenv("DB_PORT")
        db_name = os.getenv("DB_NAME")
        engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')
    except Exception as e:
        print(f"Erro ao conectar ao banco de dados: {e}")
        return

    # Se a escolha for '2' ou '3', continuamos para a escalação
    print("\n--- Início da Escalação ---")
    
    # PASSO 0: Verificar se o mercado está aberto antes de prosseguir
    mercado_status = get_mercado_status()
    if not mercado_status or mercado_status.get('status_mercado') != 1:
        status_id = mercado_status.get('status_mercado', 'desconhecido') if mercado_status else 'desconhecido'
        # Mapear IDs para nomes mais amigáveis (fonte: documentação não oficial da API)
        status_map = {
            1: "Aberto",
            2: "Fechado",
            3: "Em Manutenção",
            4: "Pós-Rodada"
        }
        status_nome = status_map.get(status_id, f"Desconhecido ({status_id})")
        print(f"\nO mercado está {status_nome.upper()}. Não é possível escalar um time agora.")
        print("A escalação só pode ser feita quando o mercado estiver Aberto.")
        return

    restricoes = fazer_perguntas()
    print("\nRestrições definidas:", restricoes)
    
    df_train, df_predict = gerar_features_para_modelo(engine)
    
    if df_train.empty or df_predict.empty:
        print("\nNenhum dado de feature gerado para o modelo. Verifique se há dados históricos e do mercado.")
        return

    # PASSO 2: Previsão de Pontuação/Valorização
    print("\nPASSO 2: Previsão de Pontuação/Valorização")
    
    # Treinar ou carregar os modelos de pontuação
    models = load_model()
    if models is None:
        # Se for o foco em pontuação, treinamos os modelos
        if restricoes['foco'] == 1:
            models = train_model(df_train)
        else:
            print("\nModelo de previsão de valorização ainda não implementado. Treine o modelo de pontuação primeiro.")
            return

    calibrator = None
    if models and restricoes['foco'] == 1 and not df_train.empty:
        calibrator = BackpropCalibrator.from_history(df_train, models)
        if calibrator and not calibrator.bias_by_position.empty:
            calibrator.report()

    if models and restricoes['foco'] == 1:
        df_com_previsao = predict_scores(df_predict.copy(), models, calibrator=calibrator) # .copy() para evitar SettingWithCopyWarning
        print(f"Atletas com pontuação prevista: {len(df_com_previsao)}")
        
        # PASSO 3: Otimização da Escalação
        print("\nPASSO 3: Otimização da Escalação")
        df_time_escalado, formacao_final, pontuacao_final, df_banco = otimizar_escalacao(
            df_jogadores=df_com_previsao,
            cartolas=restricoes['cartoletas'],
            usar_tudo=restricoes['usar_tudo'],
            engine=engine
        )

        if df_time_escalado is None:
            print("Não foi possível escalar um time titular. Verifique as restrições e o pool de jogadores.")


    elif restricoes['foco'] == 2:
        print("\nFoco: Valorização (previsão ainda não implementada)")

    # PASSO 3: Otimização da Escalação
    # print("\nPASSO 3: Otimização da Escalação (ainda não implementado)")

if __name__ == "__main__":
    main()
