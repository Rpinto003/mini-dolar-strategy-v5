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
    # Criar dados mais realistas para o teste
    dates = pd.date_range(start='2023-01-01', periods=100, freq='5min')
    
    # Gerar preços que seguem um padrão mais realista
    base_price = 100
    random_walk = np.random.normal(0, 0.1, len(dates)).cumsum()
    prices = base_price + random_walk
    
    data = pd.DataFrame({
        'open': prices + np.random.normal(0, 0.05, len(dates)),
        'high': prices + abs(np.random.normal(0, 0.1, len(dates))),
        'low': prices - abs(np.random.normal(0, 0.1, len(dates))),
        'close': prices + np.random.normal(0, 0.05, len(dates)),
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Garantir que high é sempre o maior e low o menor
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    return data

def test_indicators_calculation(strategy, sample_data):
    """Test calculation of technical indicators."""
    results = strategy.calculate_indicators(sample_data.copy())
    
    # Verificar se todos os indicadores esperados estão presentes
    expected_indicators = ['rsi', 'ma_fast', 'ma_slow', 'macd', 'macd_signal', 
                         'macd_hist', 'bb_upper', 'bb_middle', 'bb_lower']
    
    for indicator in expected_indicators:
        assert indicator in results.columns, f"Indicator {indicator} not found in results"
    
    # Verificar ranges dos indicadores
    assert results['rsi'].between(0, 100).all(), "RSI values out of range"
    assert (results['bb_upper'] >= results['bb_middle']).all(), "Bollinger Bands upper < middle"
    assert (results['bb_lower'] <= results['bb_middle']).all(), "Bollinger Bands lower > middle"

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
    # Teste com DataFrame vazio
    with pytest.raises((ValueError, KeyError)):
        strategy.calculate_indicators(pd.DataFrame())
    
    # Teste com dados faltando colunas necessárias
    invalid_data = pd.DataFrame({
        'close': [100, 101, 102],
        'volume': [1000, 1000, 1000]
    })
    with pytest.raises((ValueError, KeyError)):
        strategy.calculate_indicators(invalid_data)