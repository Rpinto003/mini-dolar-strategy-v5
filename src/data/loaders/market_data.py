import pandas as pd
import sqlite3
from loguru import logger

class MarketDataLoader:
    def __init__(self, db_path, table_name='candles'):
        self.db_path = db_path
        self.table_name = table_name
        logger.info(f"Using database: {db_path}, table: {table_name}")
        
    def get_minute_data(self, interval=5, start_date=None, end_date=None):
        """Load minute candles from database and aggregate by interval minutes"""
        logger.info(f"Attempting to load data from {self.db_path}, table: candles")
        
        query = f"""
        SELECT * FROM candles
        WHERE time >= ?
        AND time <= ?
        ORDER BY time ASC
        """
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(
                    query,
                    conn,
                    params=[start_date, end_date],
                    parse_dates=['time']
                )
            
            logger.info(f"Data loaded: {len(df)} records")
            
            # Verificar se 'time' é o índice
            if 'time' not in df.columns:
                raise ValueError("'time' column is missing from the data")
            
            df.set_index('time', inplace=True)
            
            # Resampling com 'min' em vez de 'T' para evitar o FutureWarning
            df = df.resample(f'{interval}min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            logger.info(f"Data resampled to {interval}-minute intervals: {len(df)} records")
            
            # Resetar o índice para manter 'time' como uma coluna
            df = df.reset_index()
            
            logger.info(f"'time' column reset to ensure its presence: {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise