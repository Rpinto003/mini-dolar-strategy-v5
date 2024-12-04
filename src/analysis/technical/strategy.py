from typing import Dict, Optional
import pandas as pd
import numpy as np
from loguru import logger
import talib

class TechnicalStrategy:
    """Technical analysis strategy for Mini Dollar trading."""
    
    def __init__(self,
                 rsi_period: int = 14,
                 ma_fast: int = 9,
                 ma_slow: int = 21):
        """
        Initialize technical strategy parameters.
        
        Args:
            rsi_period: Period for RSI calculation
            ma_fast: Fast moving average period
            ma_slow: Slow moving average period
        """
        self.rsi_period = rsi_period
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        logger.info(f"Initialized TechnicalStrategy with RSI={rsi_period}, "
                   f"MA_Fast={ma_fast}, MA_Slow={ma_slow}")
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        if data.empty:
            raise ValueError("Data cannot be empty")
            
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise KeyError(f"Data must contain columns: {required_columns}")
            
        df = data.copy()
        
        # Calcular RSI com tratamento de NaN
        try:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            
            avg_gain = gain.rolling(window=self.rsi_period).mean()
            avg_loss = loss.rolling(window=self.rsi_period).mean()
            
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            df['rsi'] = df['rsi'].fillna(50)  # Preencher NaN iniciais com valor neutro
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            df['rsi'] = 50  # Valor neutro em caso de erro
        
        # Moving Averages
        df['ma_fast'] = df['close'].rolling(window=self.ma_fast).mean()
        df['ma_slow'] = df['close'].rolling(window=self.ma_slow).mean()
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (std * 2)
        df['bb_lower'] = df['bb_middle'] - (std * 2)
        
        # Preencher NaN com forward fill e depois backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        logger.info("Calculated technical indicators")
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on technical indicators.
        
        Args:
            data: DataFrame with technical indicators
            
        Returns:
            DataFrame with added trading signals
        """
        df = data.copy()
        
        # Initialize signals column
        df['signal'] = 0
        
        # Generate signals based on RSI
        df.loc[df['rsi'] < 30, 'signal'] = 1  # Oversold
        df.loc[df['rsi'] > 70, 'signal'] = -1  # Overbought
        
        # Moving Average Crossover
        df['ma_cross'] = np.where(df['ma_fast'] > df['ma_slow'], 1, -1)
        
        # Combine signals
        df['final_signal'] = df['signal'] * (df['ma_cross'] == df['signal']).astype(int)
        
        logger.info("Generated trading signals")
        return df