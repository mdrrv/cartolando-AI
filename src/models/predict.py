import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib
import os
from typing import Dict, Iterable, List, Optional
from sklearn.linear_model import Ridge


def _build_feature_list(df_columns: Iterable[str]) -> List[str]:
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
    return [f for f in features if f in df_columns]


def _run_predictions_for_subset(df_subset: pd.DataFrame, model, feature_cols: List[str]) -> np.ndarray:
    if df_subset.empty or model is None:
        return np.zeros(len(df_subset))
    X = df_subset[feature_cols].fillna(0)
    return model.predict(X)

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

class BackpropCalibrator:
    """Calibra previsões combinando viés e regressões por posição e rodada."""

    def __init__(
        self,
        bias_by_position: Optional[pd.DataFrame] = None,
        history: Optional[pd.DataFrame] = None,
        models_by_position: Optional[Dict[int, Ridge]] = None,
        feature_columns: Optional[Dict[int, List[str]]] = None,
        calibration_dataset: Optional[pd.DataFrame] = None,
        alpha: float = 1.0,
    ):
        self.bias_by_position = bias_by_position.reset_index(drop=True) if bias_by_position is not None else pd.DataFrame()
        self.history = history if history is not None else pd.DataFrame()
        self.models_by_position = models_by_position or {}
        self.feature_columns = feature_columns or {}
        self.calibration_dataset = calibration_dataset if calibration_dataset is not None else pd.DataFrame()
        self.alpha = alpha

    @classmethod
    def from_history(
        cls,
        df_history: pd.DataFrame,
        models,
        smoothing_factor: float = 0.3,
        min_samples_per_position: int = 25,
        alpha: float = 1.0,
        verbose: bool = True,
    ) -> "BackpropCalibrator":
        required_cols = {'rodada', 'posicao_id', 'joga_em_casa', 'pontuacao'}
        if df_history.empty or not required_cols.issubset(df_history.columns):
            if verbose:
                print("Histórico insuficiente para gerar calibração.")
            return cls(alpha=alpha)

        feature_cols = _build_feature_list(df_history.columns)
        if not feature_cols:
            if verbose:
                print("Não há features compatíveis para executar a calibração.")
            return cls(alpha=alpha)

        df_hist = df_history.dropna(subset=['rodada']).copy()
        df_hist['rodada'] = df_hist['rodada'].astype(int)
        df_hist = df_hist.sort_values(['rodada', 'atleta_id']).reset_index(drop=True)

        model_casa, model_visitante = models

        running_bias: Dict[int, float] = {}
        samples_counter: Dict[int, int] = {}
        history_rows: List[Dict[str, float]] = []
        calibration_records: List[Dict[str, float]] = []

        for rodada in sorted(df_hist['rodada'].unique()):
            df_round = df_hist[df_hist['rodada'] == rodada].copy()
            df_round['pontuacao_prevista_model'] = 0.0

            idx_casa = df_round['joga_em_casa'] == 1
            idx_visitante = df_round['joga_em_casa'] == 0

            if not df_round[idx_casa].empty and model_casa is not None:
                preds_casa = _run_predictions_for_subset(df_round[idx_casa], model_casa, feature_cols)
                df_round.loc[idx_casa, 'pontuacao_prevista_model'] = preds_casa

            if not df_round[idx_visitante].empty and model_visitante is not None:
                preds_visitante = _run_predictions_for_subset(df_round[idx_visitante], model_visitante, feature_cols)
                df_round.loc[idx_visitante, 'pontuacao_prevista_model'] = preds_visitante

            df_round['erro'] = df_round['pontuacao'] - df_round['pontuacao_prevista_model']

            for posicao_id, grupo in df_round.groupby('posicao_id'):
                erro_medio = grupo['erro'].mean()
                prev_bias = running_bias.get(posicao_id, 0.0)
                novo_bias = (1 - smoothing_factor) * prev_bias + smoothing_factor * erro_medio
                running_bias[posicao_id] = novo_bias
                samples_counter[posicao_id] = samples_counter.get(posicao_id, 0) + len(grupo)

                history_rows.append({
                    'rodada': rodada,
                    'posicao_id': posicao_id,
                    'erro_medio': erro_medio,
                    'bias': novo_bias,
                    'amostras_acumuladas': samples_counter[posicao_id],
                })

            for _, row in df_round.iterrows():
                record = {
                    'rodada': rodada,
                    'posicao_id': int(row['posicao_id']),
                    'erro': float(row['erro'])
                }
                feature_vals = row.reindex(feature_cols).fillna(0.0).to_dict()
                record.update(feature_vals)
                calibration_records.append(record)

        bias_rows = [
            {
                'posicao_id': pos,
                'bias': running_bias.get(pos, 0.0),
                'amostras': samples_counter.get(pos, 0),
            }
            for pos in running_bias.keys()
            if samples_counter.get(pos, 0) >= min_samples_per_position
        ]

        bias_df = pd.DataFrame(bias_rows)
        history_df = pd.DataFrame(history_rows)
        calibration_df = pd.DataFrame(calibration_records)

        models_by_position: Dict[int, Ridge] = {}
        feature_map: Dict[int, List[str]] = {}

        if not calibration_df.empty:
            calibration_df[feature_cols] = calibration_df[feature_cols].fillna(0.0)
            for posicao_id, grupo in calibration_df.groupby('posicao_id'):
                if len(grupo) < min_samples_per_position:
                    continue
                X_pos = grupo[feature_cols]
                y_pos = grupo['erro']
                model = Ridge(alpha=alpha, fit_intercept=True)
                model.fit(X_pos, y_pos)
                models_by_position[int(posicao_id)] = model
                feature_map[int(posicao_id)] = feature_cols

        if verbose:
            if not bias_df.empty:
                print("\nCalibração (viés por posição) usando média móvel de erros:")
                for _, row in bias_df.sort_values('posicao_id').iterrows():
                    print(
                        f" - Posição {int(row['posicao_id'])}: bias {row['bias']:.3f} (amostras={int(row['amostras'])})"
                    )
            else:
                print("Calibração por viés não gerou ajustes significativos (poucas amostras por posição).")

            if models_by_position:
                print("\nModelos de regressão por posição treinados com scouts como features.")
            else:
                print("Nenhum modelo de regressão por posição pôde ser treinado (dados insuficientes).")

        return cls(
            bias_by_position=bias_df,
            history=history_df,
            models_by_position=models_by_position,
            feature_columns=feature_map,
            calibration_dataset=calibration_df,
            alpha=alpha,
        )

    def apply(self, df_round: pd.DataFrame) -> pd.DataFrame:
        if df_round is None:
            return df_round

        df_round = df_round.copy()
        df_round['pontuacao_prevista'] = df_round['pontuacao_prevista_model']
        df_round['calibration_bias'] = 0.0
        df_round['calibration_model_delta'] = 0.0

        if not self.bias_by_position.empty:
            bias_map = self.bias_by_position.set_index('posicao_id')['bias']
            df_round['calibration_bias'] = df_round['posicao_id'].map(bias_map).fillna(0.0)
            df_round['pontuacao_prevista'] += df_round['calibration_bias']

        for posicao_id, model in self.models_by_position.items():
            feature_cols = self.feature_columns.get(posicao_id, [])
            if not feature_cols:
                continue
            idx_pos = df_round['posicao_id'] == posicao_id
            if not idx_pos.any():
                continue
            available = [col for col in feature_cols if col in df_round.columns]
            X_pos = df_round.loc[idx_pos, available].copy()
            for missing_col in feature_cols:
                if missing_col not in available:
                    X_pos[missing_col] = 0.0
            X_pos = X_pos[feature_cols].fillna(0.0)
            if X_pos.empty:
                continue
            delta = model.predict(X_pos)
            df_round.loc[idx_pos, 'calibration_model_delta'] = delta
            df_round.loc[idx_pos, 'pontuacao_prevista'] += delta

        df_round['calibration_total_delta'] = df_round['calibration_bias'] + df_round['calibration_model_delta']
        return df_round

    def report(self) -> None:
        if self.bias_by_position.empty and not self.models_by_position:
            print("Calibração: nenhum ajuste aplicado.")
            return

        if not self.bias_by_position.empty:
            print("\nResumo dos ajustes de viés por posição:")
            print(
                self.bias_by_position.sort_values('posicao_id')
                .assign(bias=lambda df: df['bias'].round(3))
                .to_string(index=False)
            )

        if self.models_by_position:
            print("\nResumo dos pesos por scout (top 3 por posição):")
            for posicao_id in sorted(self.models_by_position.keys()):
                model = self.models_by_position[posicao_id]
                feature_cols = self.feature_columns.get(posicao_id, [])
                if not feature_cols:
                    continue
                coef_series = pd.Series(model.coef_, index=feature_cols)
                top_features = coef_series.abs().sort_values(ascending=False).head(3).index
                top_str = ", ".join(f"{feat}:{coef_series[feat]:.3f}" for feat in top_features)
                print(
                    f" - Posição {posicao_id}: intercept {model.intercept_:.3f} | {top_str}"
                )


def predict_scores(
    df_current_round_features: pd.DataFrame,
    models,
    calibrator: Optional["BackpropCalibrator"] = None,
    apply_calibration: bool = True,
):
    """Faz previsões de pontuação usando o modelo apropriado e aplica calibração opcional."""

    print("\nFazendo previsões de pontuação...")

    model_casa, model_visitante = models
    feature_cols = _build_feature_list(df_current_round_features.columns)

    df_current_round_features['pontuacao_prevista'] = 0.0
    
    idx_casa = df_current_round_features['joga_em_casa'] == 1
    idx_visitante = df_current_round_features['joga_em_casa'] == 0
    
    df_casa = df_current_round_features[idx_casa]
    df_visitante = df_current_round_features[idx_visitante]

    if not df_casa.empty and model_casa:
        preds_casa = _run_predictions_for_subset(df_casa, model_casa, feature_cols)
        df_current_round_features.loc[idx_casa, 'pontuacao_prevista'] = preds_casa
        print(f"Previsões feitas para {len(df_casa)} atletas jogando em casa.")

    if not df_visitante.empty and model_visitante:
        preds_visitante = _run_predictions_for_subset(df_visitante, model_visitante, feature_cols)
        df_current_round_features.loc[idx_visitante, 'pontuacao_prevista'] = preds_visitante
        print(f"Previsões feitas para {len(df_visitante)} atletas jogando como visitante.")

    df_current_round_features['pontuacao_prevista_model'] = df_current_round_features['pontuacao_prevista']

    if apply_calibration and calibrator is not None:
        df_current_round_features = calibrator.apply(df_current_round_features)

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
