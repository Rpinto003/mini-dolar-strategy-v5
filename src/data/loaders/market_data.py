from typing import Optional, Dict
import pandas as pd
from datetime import datetime
from loguru import logger
from sqlalchemy import create_engine

class MarketDataLoader:
    """Market data loader for Mini Dollar futures."""
    
    def __init__(self, db_path: str = "src/data/database/candles.db"):
        """
        Initialize the market data loader.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.engine = create_engine(f"sqlite:///{db_path}")
        logger.info(f"Initialized MarketDataLoader with database: {db_path}")
        
    def load_data(self, 
                 start_date: str, 
                 end_date: str,
                 timeframe: str = "5T") -> pd.DataFrame:
        """
        Load market data for the specified period.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Data timeframe ('1T', '5T', '15T', etc)
            
        Returns:
            DataFrame with OHLCV data
        """
        query = f"""
        SELECT timestamp, open, high, low, close, volume 
        FROM candles 
        WHERE timestamp BETWEEN '{start_date}' AND '{end_date}'
        AND timeframe = '{timeframe}'
        ORDER BY timestamp
        """
        
        data = pd.read_sql(query, self.engine)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        
        logger.info(f"Loaded {len(data)} records from {start_date} to {end_date}")
        return data

    def get_latest_data(self, 
                       lookback: int = 100,
                       timeframe: str = "5T") -> pd.DataFrame:
        """
        Get the most recent market data.
        
        Args:
            lookback: Number of candles to retrieve
            timeframe: Data timeframe
            
        Returns:
            DataFrame with recent OHLCV data
        """
        query = f"""
        SELECT timestamp, open, high, low, close, volume 
        FROM candles 
        WHERE timeframe = '{timeframe}'
        ORDER BY timestamp DESC
        LIMIT {lookback}
        """
        
        data = pd.read_sql(query, self.engine)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        data.sort_index(inplace=True)
        
        return data