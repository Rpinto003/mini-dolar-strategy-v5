import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.data.loaders import MarketDataLoader

@pytest.fixture
def loader():
    """Create a MarketDataLoader instance for testing."""
    return MarketDataLoader()

@pytest.fixture
def sample_dates():
    """Generate sample dates for testing."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    return start_date, end_date

def test_load_data(loader, sample_dates):
    """Test basic data loading functionality."""
    start_date, end_date = sample_dates
    
    data = loader.get_minute_data(
        interval=5,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    assert data.index.is_monotonic_increasing

def test_data_integrity(loader, sample_dates):
    """Test data integrity constraints."""
    start_date, end_date = sample_dates
    
    data = loader.get_minute_data(
        interval=5,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    # Price integrity
    assert all(data['high'] >= data['low'])
    assert all(data['high'] >= data['close'])
    assert all(data['high'] >= data['open'])
    assert all(data['low'] <= data['close'])
    assert all(data['low'] <= data['open'])
    
    # Volume integrity
    assert all(data['volume'] >= 0)