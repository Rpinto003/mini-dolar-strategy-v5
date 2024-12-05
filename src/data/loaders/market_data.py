import pandas as pd
import sqlite3
from loguru import logger

class MarketDataLoader:
    def __init__(self, db_path):
        self.db_path = db_path
        logger.info(f"Using database: {db_path}")
        
    def get_minute_data(self, interval=5, start_date=None, end_date=None):
        """Load minute candles from database"""
        logger.info(f"Attempting to load data from {self.db_path}")
        
        query = """
        SELECT * FROM candles
        WHERE time >= ?
        AND time <= ?
        ORDER BY time ASC
        """
        
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(
                query,
                conn,
                params=[start_date, end_date],
                parse_dates=['time']
            )
            conn.close()
            
            logger.info(f"Data loaded: {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise