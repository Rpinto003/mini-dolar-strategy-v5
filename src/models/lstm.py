from typing import Tuple, List
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from loguru import logger

class LSTMPredictor:
    """LSTM model for price prediction."""
    
    def __init__(self,
                 window_size: int = 60,
                 feature_columns: List[str] = None,
                 units: int = 50):
        """
        Initialize LSTM predictor.
        
        Args:
            window_size: Number of time steps to look back
            feature_columns: List of feature column names
            units: Number of LSTM units
        """
        self.window_size = window_size
        self.feature_columns = feature_columns or ['close', 'volume', 'rsi', 'macd']
        self.units = units
        self.model = None
        self.scaler = MinMaxScaler()
        
        logger.info(f"Initialized LSTMPredictor with window={window_size}, features={feature_columns}")
    
    def prepare_data(self, 
                    data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM model.
        
        Args:
            data: DataFrame with features
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        # Scale features
        scaled_data = self.scaler.fit_transform(data[self.feature_columns])
        
        X, y = [], []
        for i in range(self.window_size, len(scaled_data)):
            X.append(scaled_data[i-self.window_size:i])
            y.append(scaled_data[i, 0])  # Predict next close price
        
        return np.array(X), np.array(y)
    
    def build_model(self):
        """Build LSTM model architecture."""
        self.model = Sequential([
            LSTM(self.units, return_sequences=True, 
                 input_shape=(self.window_size, len(self.feature_columns))),
            Dropout(0.2),
            LSTM(self.units//2),
            Dropout(0.2),
            Dense(1)
        ])
        
        self.model.compile(optimizer=Adam(), loss='mse')
        logger.info("Built LSTM model")
    
    def train(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             epochs: int = 50,
             batch_size: int = 32,
             validation_split: float = 0.2):
        """
        Train LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Validation data fraction
        """
        if self.model is None:
            self.build_model()
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        logger.info("Completed model training")
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with trained model.
        
        Args:
            X: Input features
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X)
        # Inverse transform predictions
        pred_transformed = np.zeros((len(predictions), len(self.feature_columns)))
        pred_transformed[:, 0] = predictions.flatten()
        predictions = self.scaler.inverse_transform(pred_transformed)[:, 0]
        
        return predictions