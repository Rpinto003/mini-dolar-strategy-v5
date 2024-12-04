import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.agents import StrategyCoordinator, MarketAgent

@pytest.fixture
def coordinator():
    """Create a StrategyCoordinator instance for testing."""
    return StrategyCoordinator(
        initial_balance=100000,
        max_position=1,
        stop_loss=100,
        take_profit=200
    )

@pytest.fixture
def market_agent():
    """Create a MarketAgent instance for testing."""
    return MarketAgent(
        initial_balance=100000,
        max_position=1,
        stop_loss=100,
        take_profit=200
    )

@pytest.fixture
def sample_data():
    """Generate sample market data with signals."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='5min')
    data = pd.DataFrame({
        'open': np.random.randn(len(dates)).cumsum() + 1000,
        'high': np.random.randn(len(dates)).cumsum() + 1000,
        'low': np.random.randn(len(dates)).cumsum() + 1000,
        'close': np.random.randn(len(dates)).cumsum() + 1000,
        'volume': np.random.randint(1000, 10000, len(dates)),
        'final_signal': np.random.choice([-1, 0, 1], len(dates))
    }, index=dates)
    
    return data

def test_coordinator_initialization(coordinator):
    """Test StrategyCoordinator initialization."""
    assert coordinator.market.initial_balance == 100000
    assert coordinator.market.max_position == 1
    assert coordinator.market.stop_loss == 100
    assert coordinator.market.take_profit == 200

def test_market_agent_trade_execution(market_agent, sample_data):
    """Test MarketAgent trade execution."""
    results = market_agent.execute_trades(sample_data)
    
    assert 'trade_executed' in results.columns
    assert 'profit' in results.columns
    assert 'balance' in results.columns
    
    # Check balance updates
    assert all(results['balance'] > 0)
    
    # Check position constraints
    positions = abs(results[results['trade_executed']]['final_signal'])
    assert all(positions <= market_agent.max_position)

def test_stop_loss_take_profit(market_agent):
    """Test stop loss and take profit functionality."""
    # Create sample data with a profitable trade
    data = pd.DataFrame({
        'close': [1000, 1100, 1200, 1300],  # Strong uptrend
        'final_signal': [1, 0, 0, -1]  # Buy and sell signals
    })
    
    results = market_agent.execute_trades(data)
    profits = results[results['trade_executed']]['profit']
    
    # Check if profits are within SL/TP limits
    assert all(abs(profits) <= market_agent.take_profit)

def test_backtest_functionality(coordinator):
    """Test backtesting functionality."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    results = coordinator.backtest(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        interval=5  # 5-minute data
    )
    
    assert isinstance(results, pd.DataFrame)
    assert 'trade_executed' in results.columns
    assert 'profit' in results.columns
    assert 'balance' in results.columns

def test_performance_metrics(coordinator, sample_data):
    """Test performance metrics calculation."""
    results = coordinator.market.execute_trades(sample_data)
    metrics = coordinator.get_performance_metrics(results)
    
    expected_metrics = ['total_trades', 'win_rate', 'total_profit', 
                       'max_drawdown', 'sharpe_ratio']
    assert all(metric in metrics for metric in expected_metrics)
    assert metrics['win_rate'] >= 0 and metrics['win_rate'] <= 1