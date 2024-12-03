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
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='5T')
    
    # Create sample trading results
    results = pd.DataFrame({
        'close': np.random.randn(len(dates)).cumsum() + 1000,
        'trade_executed': np.random.choice([True, False], len(dates), p=[0.1, 0.9]),
        'profit': np.random.normal(10, 50, len(dates)),
        'balance': np.random.randn(len(dates)).cumsum() + 100000
    }, index=dates)
    
    # Ensure realistic balance progression
    results['balance'] = results['balance'].rolling(window=10).mean() + 100000
    
    return results

def test_metrics_calculation(analyzer, sample_results):
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

def test_report_generation(analyzer, sample_results):
    """Test performance report generation."""
    report = analyzer.generate_report(sample_results)
    
    assert isinstance(report, str)
    assert "Trading Strategy Performance Report" in report
    assert "Performance Metrics" in report
    assert "Trade Statistics" in report

def test_drawdown_calculation(analyzer, sample_results):
    """Test drawdown calculations."""
    drawdown = analyzer._calculate_drawdown_series(sample_results)
    
    assert isinstance(drawdown, pd.Series)
    assert all(drawdown <= 0)  # Drawdowns should be negative or zero
    assert len(drawdown) == len(sample_results)

def test_sharpe_ratio(analyzer, sample_results):
    """Test Sharpe ratio calculation."""
    sharpe = analyzer._calculate_sharpe(sample_results)
    
    assert isinstance(sharpe, float)
    assert not np.isnan(sharpe)

def test_error_handling(analyzer):
    """Test error handling for invalid inputs."""
    # Test with empty DataFrame
    with pytest.raises(ValueError):
        analyzer.calculate_metrics(pd.DataFrame())
    
    # Test with missing columns
    invalid_data = pd.DataFrame({
        'close': [100, 101, 102],
        'profit': [1, -1, 1]
    })
    with pytest.raises(KeyError):
        analyzer.calculate_metrics(invalid_data)