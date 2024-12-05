import sqlite3
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path

def init_database():
    db_path = Path('src/data/database/candles.db')
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS candles (
            time TIMESTAMP PRIMARY KEY,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT,
            volume INTEGER
        )
    ''')
    conn.commit()
    logger.info(f"Initialized database at {db_path}")
    return conn

def load_sample_data():
    """Load sample data for testing"""
    # Sample data for last 6 months of 5-minute candles
    dates = pd.date_range(start='2024-06-01', end='2024-12-05', freq='5min')
    n = len(dates)
    
    data = pd.DataFrame({
        'time': dates,
        'open': [5000 + i*0.1 + np.random.normal() for i in range(n)],
        'high': [5000 + i*0.1 + abs(np.random.normal(0, 0.5)) for i in range(n)],
        'low': [5000 + i*0.1 - abs(np.random.normal(0, 0.5)) for i in range(n)],
        'close': [5000 + i*0.1 + np.random.normal() for i in range(n)],
        'volume': np.random.randint(100, 1000, n)
    })
    
    conn = init_database()
    data.to_sql('candles', conn, if_exists='replace', index=False)
    logger.info(f"Loaded {len(data)} sample candles")
    conn.close()

if __name__ == '__main__':
    logger.add("logs/init_db_{time}.log")
    load_sample_data()