import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib
import os

MODEL_PATH_CASA = './src/models/xgboost_pontuacao_model_casa.joblib'
MODEL_PATH_VISITANTE = './src/models/xgboost_pontuacao_model_visitante.joblib'
VAL_MODEL_PATH_CASA = './src/models/xgboost_valorizacao_model_casa.joblib'
VAL_MODEL_PATH_VISITANTE = './src/models/xgboost_valorizacao_model_visitante.joblib'

def _train_single_model(df: pd.DataFrame, features: list, target: str, model_type: str):
    """Função auxiliar para treinar um único modelo (reutilizável para pontuação e valorização)."""
    if len(df) < 100:
        print(f"Não há dados suficientes para treinar o modelo de {model_type} (registros: {len(df)}).")
        return None
        
    X_split = df[features]
    y_split = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X_split, y_split, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"Modelo para jogos em {model_type} treinado com R^2 no teste: {score:.4f}")
    
    return model

def train_model(df_features: pd.DataFrame):
    """
    Treina e salva dois modelos de XGBoost para previsão de pontuação:
    um para jogos em casa e outro para jogos como visitante.
    """
    print("\\nIniciando treinamento dos modelos de previsão de pontuação (casa e visitante)...")

    # Lista de siglas de scouts para gerar o nome das features
    scouts = [
        'G', 'A', 'FT', 'FF', 'FD', 'FS', 'PS', 'I', 'PP',
        'DS', 'SG', 'DD', 'DP', 'GS', 'GC', 'FC', 'CA', 'CV', 'PI'
    ]
    
    # Novas métricas de scout combinadas
    metricas_combinadas = [
        "participacoes_gol", "total_finalizacoes", "finalizacoes_alvo", "pontaria",
        "taxa_conversao", "criacao_oportunidades", "total_defesas", "balanco_duelos",
        "erros_capitais", "indice_indisciplina"
    ]

    # Features base + posicao_id
    # Preço foi removido pois não está disponível nos dados históricos de treino
    features = [
        'pontuacao_rodada_anterior', 'media_pontos_ult3', 'posicao_id'
    ]
    # Adiciona as features de média de scout bruto
    features += [f'media_{s}_ult3' for s in scouts]
    # Adiciona as features de média de métricas combinadas
    features += [f'media_{m}_ult3' for m in metricas_combinadas]
    
    target = 'pontuacao'

    # Filtrar apenas as features presentes no DataFrame
    features = [f for f in features if f in df_features.columns]
    
    # Preparar os dados
    df_features = df_features.fillna(0)
    X = df_features[features]
    y = df_features[target]

    # Dividir dados em casa e visitante
    df_casa = df_features[df_features['joga_em_casa'] == 1]
    df_visitante = df_features[df_features['joga_em_casa'] == 0]
    
    # Treinar os dois modelos
    model_casa = _train_single_model(df_casa, features, target, "casa")
    model_visitante = _train_single_model(df_visitante, features, target, "visitante")

    # Salvar os modelos
    if model_casa:
        joblib.dump(model_casa, MODEL_PATH_CASA)
        print(f"Modelo 'casa' salvo em {MODEL_PATH_CASA}")
    if model_visitante:
        joblib.dump(model_visitante, MODEL_PATH_VISITANTE)
        print(f"Modelo 'visitante' salvo em {MODEL_PATH_VISITANTE}")
    
    return model_casa, model_visitante

def train_valuation_model(df_features: pd.DataFrame):
    """Treina e salva dois modelos de XGBoost para previsão de VALORIZAÇÃO."""
    print("\\nIniciando treinamento dos modelos de previsão de VALORIZAÇÃO (casa e visitante)...")

    scouts = [
        'G', 'A', 'FT', 'FF', 'FD', 'FS', 'PS', 'I', 'PP',
        'DS', 'SG', 'DD', 'DP', 'GS', 'GC', 'FC', 'CA', 'CV', 'PI'
    ]
    metricas_combinadas = [
        "participacoes_gol", "total_finalizacoes", "finalizacoes_alvo", "pontaria",
        "taxa_conversao", "criacao_oportunidades", "total_defesas", "balanco_duelos",
        "erros_capitais", "indice_indisciplina"
    ]
    features = [
        'pontuacao_rodada_anterior', 'media_pontos_ult3', 'posicao_id'
    ]
    features += [f'media_{s}_ult3' for s in scouts]
    features += [f'media_{m}_ult3' for m in metricas_combinadas]
    target = 'variacao_num' # O alvo agora é a valorização

    features = [f for f in features if f in df_features.columns]
    
    df_features = df_features.fillna(0)

    df_casa = df_features[df_features['joga_em_casa'] == 1]
    df_visitante = df_features[df_features['joga_em_casa'] == 0]
    
    model_casa = _train_single_model(df_casa, features, target, "valorização casa")
    model_visitante = _train_single_model(df_visitante, features, target, "valorização visitante")

    if model_casa:
        joblib.dump(model_casa, VAL_MODEL_PATH_CASA)
        print(f"Modelo 'valorização casa' salvo em {VAL_MODEL_PATH_CASA}")
    if model_visitante:
        joblib.dump(model_visitante, VAL_MODEL_PATH_VISITANTE)
        print(f"Modelo 'valorização visitante' salvo em {VAL_MODEL_PATH_VISITANTE}")
    
    return model_casa, model_visitante


def load_model():
    """Carrega os modelos de PONTUAÇÃO de casa e visitante, se existirem."""
    model_casa = None
    model_visitante = None
    if os.path.exists(MODEL_PATH_CASA):
        print(f"Carregando modelo 'casa' de {MODEL_PATH_CASA}")
        model_casa = joblib.load(MODEL_PATH_CASA)
    if os.path.exists(MODEL_PATH_VISITANTE):
        print(f"Carregando modelo 'visitante' de {MODEL_PATH_VISITANTE}")
        model_visitante = joblib.load(MODEL_PATH_VISITANTE)
    
    if model_casa and model_visitante:
        return model_casa, model_visitante
    return None

def load_valuation_model():
    """Carrega os modelos de VALORIZAÇÃO de casa e visitante, se existirem."""
    model_casa = None
    model_visitante = None
    if os.path.exists(VAL_MODEL_PATH_CASA):
        print(f"Carregando modelo 'valorização casa' de {VAL_MODEL_PATH_CASA}")
        model_casa = joblib.load(VAL_MODEL_PATH_CASA)
    if os.path.exists(VAL_MODEL_PATH_VISITANTE):
        print(f"Carregando modelo 'valorização visitante' de {VAL_MODEL_PATH_VISITANTE}")
        model_visitante = joblib.load(VAL_MODEL_PATH_VISITANTE)
    
    if model_casa and model_visitante:
        return model_casa, model_visitante
    return None

def predict_scores(df_current_round_features: pd.DataFrame, models):
    """
    Faz previsões de pontuação usando o modelo apropriado (casa ou visitante).
    """
    print("\\nFazendo previsões de pontuação...")
    
    model_casa, model_visitante = models

    # Lista de siglas de scouts para gerar o nome das features
    scouts = [
        'G', 'A', 'FT', 'FF', 'FD', 'FS', 'PS', 'I', 'PP',
        'DS', 'SG', 'DD', 'DP', 'GS', 'GC', 'FC', 'CA', 'CV', 'PI'
    ]
    
    # Novas métricas de scout combinadas
    metricas_combinadas = [
        "participacoes_gol", "total_finalizacoes", "finalizacoes_alvo", "pontaria",
        "taxa_conversao", "criacao_oportunidades", "total_defesas", "balanco_duelos",
        "erros_capitais", "indice_indisciplina"
    ]

    # Preço foi removido pois não está disponível nos dados históricos de treino
    features = [
        'pontuacao_rodada_anterior', 'media_pontos_ult3', 'posicao_id'
    ]
    features += [f'media_{s}_ult3' for s in scouts]
    features += [f'media_{m}_ult3' for m in metricas_combinadas]
    
    features = [f for f in features if f in df_current_round_features.columns]

    df_current_round_features['pontuacao_prevista'] = 0.0
    
    # Separar atletas que jogam em casa e fora
    idx_casa = df_current_round_features['joga_em_casa'] == 1
    idx_visitante = df_current_round_features['joga_em_casa'] == 0
    
    df_casa = df_current_round_features[idx_casa]
    df_visitante = df_current_round_features[idx_visitante]

    if not df_casa.empty and model_casa:
        X_predict_casa = df_casa[features].fillna(0)
        df_current_round_features.loc[idx_casa, 'pontuacao_prevista'] = model_casa.predict(X_predict_casa)
        print(f"Previsões feitas para {len(df_casa)} atletas jogando em casa.")

    if not df_visitante.empty and model_visitante:
        X_predict_visitante = df_visitante[features].fillna(0)
        df_current_round_features.loc[idx_visitante, 'pontuacao_prevista'] = model_visitante.predict(X_predict_visitante)
        print(f"Previsões feitas para {len(df_visitante)} atletas jogando como visitante.")

    print("Previsões concluídas.")
    return df_current_round_features

def predict_valuation(df_current_round_features: pd.DataFrame, models):
    """Faz previsões de VALORIZAÇÃO para os jogadores da rodada atual."""
    print("\\nFazendo previsões de valorização...")
    
    model_casa, model_visitante = models

    scouts = [
        'G', 'A', 'FT', 'FF', 'FD', 'FS', 'PS', 'I', 'PP',
        'DS', 'SG', 'DD', 'DP', 'GS', 'GC', 'FC', 'CA', 'CV', 'PI'
    ]
    metricas_combinadas = [
        "participacoes_gol", "total_finalizacoes", "finalizacoes_alvo", "pontaria",
        "taxa_conversao", "criacao_oportunidades", "total_defesas", "balanco_duelos",
        "erros_capitais", "indice_indisciplina"
    ]
    features = [
        'pontuacao_rodada_anterior', 'media_pontos_ult3', 'posicao_id'
    ]
    features += [f'media_{s}_ult3' for s in scouts]
    features += [f'media_{m}_ult3' for m in metricas_combinadas]
    
    features = [f for f in features if f in df_current_round_features.columns]

    df_current_round_features['valorizacao_prevista'] = 0.0
    
    idx_casa = df_current_round_features['joga_em_casa'] == 1
    idx_visitante = df_current_round_features['joga_em_casa'] == 0
    
    df_casa = df_current_round_features[idx_casa]
    df_visitante = df_current_round_features[idx_visitante]

    if not df_casa.empty and model_casa:
        X_predict_casa = df_casa[features].fillna(0)
        df_current_round_features.loc[idx_casa, 'valorizacao_prevista'] = model_casa.predict(X_predict_casa)
        print(f"Previsões de valorização feitas para {len(df_casa)} atletas jogando em casa.")

    if not df_visitante.empty and model_visitante:
        X_predict_visitante = df_visitante[features].fillna(0)
        df_current_round_features.loc[idx_visitante, 'valorizacao_prevista'] = model_visitante.predict(X_predict_visitante)
        print(f"Previsões de valorização feitas para {len(df_visitante)} atletas jogando como visitante.")

    print("Previsões de valorização concluídas.")
    return df_current_round_features
