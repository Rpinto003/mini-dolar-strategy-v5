# src/analysis/technical/enhanced_strategy.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from .strategy import TechnicalStrategy  # Ajuste o caminho conforme necessário
import logging

# Configuração do logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTechnicalStrategy(TechnicalStrategy):
    """Estratégia Técnica Aprimorada com Machine Learning."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = RandomForestClassifier(n_estimators=200, random_state=42)
        logger.info("EnhancedTechnicalStrategy inicializada.")

    def prepare_ml_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Seleciona as features para o modelo de ML."""
        feature_columns = [
            'rsi', 'ma_fast', 'ma_slow', 'adx', 'plus_di', 'minus_di',
            'volume_ratio', 'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower',
            'atr', 'dynamic_stop_loss', 'dynamic_take_profit'
        ]
        # Assegure-se de que todas as colunas existem no DataFrame
        missing_cols = set(feature_columns) - set(data.columns)
        if missing_cols:
            logger.error(f"Missing columns for ML features: {missing_cols}")
            raise KeyError(f"Missing columns for ML features: {missing_cols}")
        return data[feature_columns].dropna()

    def create_labels(self, data: pd.DataFrame, future_steps: int = 1) -> pd.Series:
        """Cria labels para classificação baseada na variação futura do preço."""
        data['future_price'] = data['close'].shift(-future_steps)
        data['price_diff'] = data['future_price'] - data['close']
        data['label'] = 0
        data.loc[data['price_diff'] > 0, 'label'] = 1
        data.loc[data['price_diff'] < 0, 'label'] = -1
        return data['label']

    def train_ml_model(self, data: pd.DataFrame):
        """Treina o modelo de ML com os dados fornecidos."""
        df = data.dropna()
        X = self.prepare_ml_features(df)
        y = self.create_labels(df)
        
        # Divisão temporal para evitar data leakage
        split_index = int(0.8 * len(df))
        X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_val = y.iloc[:split_index], y.iloc[split_index:]
        
        # Treinamento
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_val)
        report = classification_report(y_val, y_pred)
        logger.info("Modelo de ML treinado.")
        logger.info(f"Relatório de Classificação:\n{report}")

    def generate_ml_signals(self, data: pd.DataFrame) -> pd.Series:
        """Gera sinais de trading utilizando o modelo de ML treinado."""
        features = self.prepare_ml_features(data)
        ml_signals = self.model.predict(features)
        return pd.Series(ml_signals, index=features.index)

    def run_enhanced_strategy(self, data: pd.DataFrame):
        """Executa a estratégia completa incluindo ML."""
        # Calcular indicadores técnicos
        analysis = self.calculate_indicators(data)
        
        # Gerar sinais de análise técnica
        analysis = self.generate_signals(analysis)
        
        # Treinar o modelo de ML
        self.train_ml_model(analysis)
        
        # Gerar sinais de ML
        analysis['ml_signal'] = self.generate_ml_signals(analysis)
        
        # Combinar sinais de análise técnica com ML
        analysis['final_signal_ml'] = analysis['signal'] * analysis['ml_signal']
        
        return analysis