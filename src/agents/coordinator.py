from typing import Optional
import pandas as pd
from datetime import datetime
from loguru import logger
from ..data.loaders.market_data import MarketDataLoader
from ..analysis.technical.strategy import TechnicalStrategy
from .market import MarketAgent

class StrategyCoordinator:
    """Coordinates the trading strategy execution."""
    
    def __init__(self,
                initial_balance: float = 100000,
                max_position: int = 1,
                stop_loss: float = 100,
                take_profit: float = 200,
                db_path: Optional[str] = None):
        """
        Initialize the strategy coordinator.
        
        Args:
            initial_balance: Starting balance for trading
            max_position: Maximum number of contracts per position
            stop_loss: Stop loss in points
            take_profit: Take profit in points
            db_path: Optional path to the database file
        """
        if db_path is None:
            raise ValueError("Database path must be provided.")
        
        self.data_loader = MarketDataLoader(db_path)
        self.strategy = TechnicalStrategy()
        self.market = MarketAgent(
            initial_balance=initial_balance,
            max_position=max_position,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        logger.info("Initialized StrategyCoordinator")
    
    def process_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process market data and generate trading signals.
        
        Args:
            data: Raw market data DataFrame
            
        Returns:
            DataFrame with indicators and signals
        """
        # Calculate technical indicators
        data = self.strategy.calculate_indicators(data)
        
        # Generate trading signals
        data = self.strategy.generate_signals(data)
        
        return data
    
    def backtest(self,
                start_date: str,
                end_date: str,
                interval: int = 5) -> pd.DataFrame:
        """
        Run strategy backtest.
        
        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            interval: Data interval in minutes (default: 5)
            
        Returns:
            DataFrame with backtest results
        """
        # Load historical data
        data = self.data_loader.get_minute_data(
            interval=interval,
            start_date=start_date,
            end_date=end_date
        )
        
        # Process data and generate signals
        data = self.process_market_data(data)
        
        # Execute trades
        results = self.market.execute_trades(data)
        
        logger.info(f"Completed backtest from {start_date} to {end_date}")
        return results
    
    def get_performance_metrics(self, results: pd.DataFrame) -> dict:
        """
        Calculate strategy performance metrics.
        
        Args:
            results: DataFrame with backtest results
            
        Returns:
            Dictionary with performance metrics
        """
        metrics = {
            'total_trades': len(results[results['trade_executed']]),
            'win_rate': (results['profit'] > 0).mean(),
            'total_profit': results['profit'].sum(),
            'max_drawdown': self.market.calculate_max_drawdown(results),
            'sharpe_ratio': self.market.calculate_sharpe_ratio(results)
        }
        
        logger.info(f"Calculated performance metrics: {metrics}")
        return metrics