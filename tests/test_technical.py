import pytest
import pandas as pd
import numpy as np
from src.analysis.technical import TechnicalStrategy

@pytest.fixture
def strategy():
    """Create a TechnicalStrategy instance for testing."""
    return TechnicalStrategy()

@pytest.fixture
def sample_data():
    """Generate sample price data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='5T')
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

def test_indicators_calculation(strategy, sample_data):
    """Test calculation of technical indicators."""
    results = strategy.calculate_indicators(sample_data)
    
    # Check if all expected indicators are present
    expected_indicators = ['rsi', 'ma_fast', 'ma_slow', 'macd', 'macd_signal', 
                         'macd_hist', 'bb_upper', 'bb_middle', 'bb_lower']
    assert all(indicator in results.columns for indicator in expected_indicators)
    
    # Check indicators values
    assert all(results['rsi'].between(0, 100))
    assert all(results['bb_upper'] >= results['bb_middle'])
    assert all(results['bb_lower'] <= results['bb_middle'])

def test_signal_generation(strategy, sample_data):
    """Test trading signal generation."""
    data = strategy.calculate_indicators(sample_data)
    signals = strategy.generate_signals(data)
    
    # Check signal values
    assert all(signals['signal'].isin([-1, 0, 1]))
    assert all(signals['final_signal'].isin([-1, 0, 1]))
    
    # Check signal logic
    assert all(signals.loc[signals['rsi'] > 70, 'signal'] <= 0)  # Oversold
    assert all(signals.loc[signals['rsi'] < 30, 'signal'] >= 0)  # Overbought

def test_strategy_parameters():
    """Test strategy initialization with different parameters."""
    strategy = TechnicalStrategy(rsi_period=21, ma_fast=5, ma_slow=13)
    assert strategy.rsi_period == 21
    assert strategy.ma_fast == 5
    assert strategy.ma_slow == 13

def test_error_handling(strategy):
    """Test error handling for invalid inputs."""
    # Test with empty DataFrame
    with pytest.raises(ValueError):
        strategy.calculate_indicators(pd.DataFrame())
    
    # Test with missing columns
    invalid_data = pd.DataFrame({
        'close': [100, 101, 102],
        'volume': [1000, 1000, 1000]
    })
    with pytest.raises(KeyError):
        strategy.calculate_indicators(invalid_data)