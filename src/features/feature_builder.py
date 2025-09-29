import pandas as pd
from sqlalchemy import text

def gerar_features_para_modelo(engine):
    """
    Busca dados brutos do banco e prepara DataFrames com features para treino e previsão.
    """
    print("\\nGerando features para o modelo (via Python/Pandas)...")

    # 1. Buscar dados brutos
    query_historico = "SELECT * FROM cartola_2025.pontuacao_rodada;"
    df_historico = pd.read_sql(query_historico, engine)

    query_partidas = "SELECT * FROM cartola_2025.partidas;"
    df_partidas = pd.read_sql(query_partidas, engine)

    query_mercado = """
    SELECT
        mercado.atleta_id, mercado.rodada_id, mercado.apelido, mercado.clube_id, mercado.posicao_id, mercado.preco_num,
    (mercado.preco_num - mercado.variacao_num) AS preco_anterior,
        CASE WHEN mercado.clube_id = p.clube_casa_id THEN 1 ELSE 0 END AS joga_em_casa,
        CASE WHEN mercado.clube_id = p.clube_casa_id THEN p.clube_visitante_id ELSE p.clube_casa_id END AS adversario_id
FROM cartola_2025.mercado_atletas mercado
    LEFT JOIN cartola_2025.partidas p ON mercado.rodada_id = p.rodada_id AND (mercado.clube_id = p.clube_casa_id OR mercado.clube_id = p.clube_visitante_id)
    WHERE mercado.status_id = 6;
    """
    df_mercado = pd.read_sql(text(query_mercado), engine)

    if df_historico.empty or df_mercado.empty or df_partidas.empty:
        print("Dados históricos, de partidas ou de mercado não encontrados.")
        return pd.DataFrame(), pd.DataFrame()

    # 2. Garantir tipos de dados consistentes para 'atleta_id'
    try:
        df_mercado['atleta_id'] = df_mercado['atleta_id'].astype('int64')
        df_historico['atleta_id'] = pd.to_numeric(df_historico['atleta_id'], errors='coerce')
        df_historico.dropna(subset=['atleta_id'], inplace=True)
        df_historico['atleta_id'] = df_historico['atleta_id'].astype('int64')
    except Exception as e:
        print(f"Erro ao converter 'atleta_id': {e}")
        return pd.DataFrame(), pd.DataFrame()

    # 3. Enriquecer dados históricos com informação de 'joga_em_casa'
    df_casa = df_partidas[['rodada_id', 'clube_casa_id']].rename(columns={'clube_casa_id': 'clube_id'})
    df_casa['joga_em_casa'] = 1
    df_visitante = df_partidas[['rodada_id', 'clube_visitante_id']].rename(columns={'clube_visitante_id': 'clube_id'})
    df_visitante['joga_em_casa'] = 0
    df_partidas_flat = pd.concat([df_casa, df_visitante], ignore_index=True)
    df_historico = pd.merge(df_historico, df_partidas_flat, left_on=['rodada', 'clube_id'], right_on=['rodada_id', 'clube_id'], how='left')

    # 4. Engenharia de Features no histórico
    scouts = ['G','A','FT','FF','FD','FS','PS','I','PP','DS','SG','DD','DP','GS','GC','FC','CA','CV','PI']
    for s in scouts:
        if s not in df_historico.columns:
            df_historico[s] = 0
    df_historico[scouts] = df_historico[scouts].fillna(0)
    df_historico = df_historico.sort_values(by=['atleta_id', 'rodada']).reset_index(drop=True)

    # Calcular métricas combinadas
    df_historico['participacoes_gol'] = df_historico['G'] + df_historico['A']
    df_historico['total_finalizacoes'] = df_historico['G'] + df_historico['FD'] + df_historico['FF'] + df_historico['FT']
    df_historico['finalizacoes_alvo'] = df_historico['G'] + df_historico['FD']
    df_historico['pontaria'] = (df_historico['finalizacoes_alvo'] / df_historico['total_finalizacoes'].replace(0, 1)).fillna(0)
    df_historico['taxa_conversao'] = (df_historico['G'] / df_historico['total_finalizacoes'].replace(0, 1)).fillna(0)
    df_historico['criacao_oportunidades'] = df_historico['A'] + df_historico['FS'] + df_historico['PS']
    df_historico['total_defesas'] = df_historico['DD'] + df_historico['DP']
    g_gs = df_historico['G'] + df_historico['GS']
    df_historico['eficiencia_defensiva'] = (df_historico['total_defesas'] / g_gs.replace(0, 1)).fillna(0)
    df_historico['balanco_duelos'] = df_historico['DS'] - df_historico['FC']
    df_historico['erros_capitais'] = df_historico['PI'] + df_historico['GC'] + df_historico['PP']
    df_historico['indice_indisciplina'] = df_historico['FC'] + (df_historico['CA'] * 2) + (df_historico['CV'] * 5)
    
    metricas_combinadas_cols = ["participacoes_gol", "total_finalizacoes", "finalizacoes_alvo", "pontaria", "taxa_conversao", "criacao_oportunidades", "total_defesas", "eficiencia_defensiva", "balanco_duelos", "erros_capitais", "indice_indisciplina"]
    features_to_avg = scouts + metricas_combinadas_cols + ['pontuacao']
    
    # 5. Preparar DataFrame de Treino
    grouped = df_historico.groupby('atleta_id')
    rolling_avg = grouped[features_to_avg].shift(1).rolling(window=3, min_periods=1).mean()
    rolling_avg.columns = [f"media_{col}_ult3" for col in features_to_avg]
    df_com_features = pd.concat([df_historico, rolling_avg], axis=1)
    df_com_features['pontuacao_rodada_anterior'] = grouped['pontuacao'].shift(1)
    df_train = df_com_features.dropna(subset=['pontuacao_rodada_anterior', 'joga_em_casa']).reset_index(drop=True)
    if 'media_pontuacao_ult3' in df_train.columns:
        df_train = df_train.rename(columns={'media_pontuacao_ult3': 'media_pontos_ult3'})

    # 6. Preparar DataFrame de Previsão
    atletas_mercado = df_mercado['atleta_id'].unique()
    df_historico_mercado = df_historico[df_historico['atleta_id'].isin(atletas_mercado)].copy()
    
    def get_last_3_games_mean(group):
        return group.tail(3).mean()

    last_3_games_avg = df_historico_mercado.groupby('atleta_id')[features_to_avg].apply(get_last_3_games_mean).reset_index()
    last_3_games_avg.columns = ['atleta_id'] + [f"media_{col}_ult3" for col in features_to_avg]

    last_game_score = df_historico_mercado.groupby('atleta_id')['pontuacao'].last().reset_index(name='pontuacao_rodada_anterior')
    
    features_previsao = pd.merge(last_3_games_avg, last_game_score, on='atleta_id', how='left')
    df_predict = pd.merge(df_mercado, features_previsao, on='atleta_id', how='left')
    
    if 'media_pontuacao_ult3' in df_predict.columns:
        df_predict = df_predict.rename(columns={'media_pontuacao_ult3': 'media_pontos_ult3'})
    feature_cols_predict = [col for col in df_predict.columns if 'media_' in col or 'pontuacao_rodada_anterior' in col]
    df_predict[feature_cols_predict] = df_predict[feature_cols_predict].fillna(0)

    print(f"DataFrame de treino criado com {len(df_train)} registros.")
    print(f"DataFrame de previsão criado com {len(df_predict)} registros.")
    
    return df_train, df_predict
