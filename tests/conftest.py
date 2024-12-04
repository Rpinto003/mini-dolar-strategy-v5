import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture(scope="session")
def test_dates():
    """Generate test dates that can be used across all test modules."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    return start_date, end_date

@pytest.fixture(scope="session")
def sample_ohlcv_data():
    """Generate sample OHLCV data that can be used across all test modules."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='5min')
    data = pd.DataFrame({
        'open': np.random.randn(len(dates)).cumsum() + 1000,
        'high': np.random.randn(len(dates)).cumsum() + 1000,
        'low': np.random.randn(len(dates)).cumsum() + 1000,
        'close': np.random.randn(len(dates)).cumsum() + 1000,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Ensure OHLC integrity
    data['high'] = data[['open', 'close']].max(axis=1) + abs(np.random.randn(len(dates)))
    data['low'] = data[['open', 'close']].min(axis=1) - abs(np.random.randn(len(dates)))
    
    return data