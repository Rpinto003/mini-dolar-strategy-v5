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
            data: OHLCV DataFrame with market data
            
        Returns:
            DataFrame with added technical indicators
        """
        df = data.copy()
        
        # Calculate RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.rsi_period)
        
        # Calculate Moving Averages
        df['ma_fast'] = talib.SMA(df['close'], timeperiod=self.ma_fast)
        df['ma_slow'] = talib.SMA(df['close'], timeperiod=self.ma_slow)
        
        # Calculate MACD
        macd, signal, hist = talib.MACD(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        # Calculate Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['close'])
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        
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