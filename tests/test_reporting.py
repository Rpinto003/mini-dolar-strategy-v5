import pytest
import pandas as pd
import numpy as np
from src.reporting import PerformanceAnalyzer

@pytest.fixture
def analyzer():
    """Create a PerformanceAnalyzer instance for testing."""
    return PerformanceAnalyzer()

@pytest.fixture
def sample_results():
    """Generate sample trading results for testing."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='5min')
    
    # Criar um DataFrame com resultados de trading mais realistas
    initial_balance = 100000
    trades = np.random.choice([0, 1], size=len(dates), p=[0.9, 0.1])  # 10% são trades
    profits = np.random.normal(100, 50, len(dates)) * trades
    
    # Calcular balance acumulativo
    balance = initial_balance + np.cumsum(profits)
    
    results = pd.DataFrame({
        'close': np.random.normal(100, 1, len(dates)),
        'trade_executed': trades.astype(bool),
        'profit': profits,
        'balance': balance
    }, index=dates)
    
    return results

def test_metrics_calculation(analyzer: PerformanceAnalyzer, sample_results: pd.DataFrame):
    """Test calculation of performance metrics."""
    metrics = analyzer.calculate_metrics(sample_results)
    
    # Check if all expected metrics are present
    expected_metrics = ['total_trades', 'winning_trades', 'losing_trades',
                       'win_rate', 'total_profit', 'average_profit',
                       'profit_std', 'max_profit', 'max_loss',
                       'sharpe_ratio', 'max_drawdown']
    assert all(metric in metrics for metric in expected_metrics)
    
    # Validate metric values
    assert metrics['total_trades'] >= 0
    assert metrics['win_rate'] >= 0 and metrics['win_rate'] <= 1
    assert metrics['max_drawdown'] <= 0

def test_report_generation(analyzer: PerformanceAnalyzer, sample_results: pd.DataFrame):
    """Test performance report generation."""
    report = analyzer.generate_report(sample_results)
    
    assert isinstance(report, str)
    assert "Trading Strategy Performance Report" in report
    assert "Performance Metrics" in report
    assert "Trade Statistics" in report

def test_drawdown_calculation(analyzer, sample_results):
    """Test drawdown calculations."""
    drawdown = analyzer._calculate_drawdown_series(sample_results)
    
    assert isinstance(drawdown, pd.Series), "Drawdown should be a Series"
    assert len(drawdown) == len(sample_results), "Drawdown length mismatch"
    assert (drawdown <= 0).all(), "Drawdown values should be negative or zero"
    assert not drawdown.isna().any(), "Drawdown should not contain NaN values"

def test_sharpe_ratio(analyzer: PerformanceAnalyzer, sample_results: pd.DataFrame):
    """Test Sharpe ratio calculation."""
    sharpe = analyzer._calculate_sharpe(sample_results)
    
    assert isinstance(sharpe, float)
    assert not np.isnan(sharpe)

def test_error_handling(analyzer):
    """Test error handling for invalid inputs."""
    # DataFrame vazio
    with pytest.raises(ValueError):
        analyzer.calculate_metrics(pd.DataFrame())
    
    # DataFrame com dados faltando
    invalid_data = pd.DataFrame({
        'close': [100, 101, 102],
        'balance': [100000, 100100, 100200]
    })
    
    with pytest.raises((ValueError, KeyError)):
        analyzer.calculate_metrics(invalid_data)