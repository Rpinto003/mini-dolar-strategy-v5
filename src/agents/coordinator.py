from typing import Optional, Dict
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
                 db_path: Optional[str] = None,
                 strategy_params: Optional[Dict] = None):
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
        self.strategy_params = strategy_params or {}
        self.strategy = TechnicalStrategy(**self.strategy_params)
        self.market = MarketAgent(
            initial_balance=initial_balance,
            max_position=max_position,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr_multiplier=self.strategy.atr_multiplier
        )
      
        logger.info("Initialized StrategyCoordinator with parameters:", self.strategy_params)
    
    def process_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process market data and generate trading signals."""
        try:
            # Verificar dados necessários
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                logger.error(f"Missing columns in data: {missing_columns}")
                raise ValueError(f"Required columns missing: {missing_columns}")
            
            # Criar cópia dos dados para evitar modificações indesejadas
            df = data.copy()
            
            # Calcular indicadores
            df = self.strategy.calculate_indicators(df)
            
            # Gerar sinais
            df = self.strategy.generate_signals(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}")
            raise
    
    def backtest(self,
                start_date: str,
                end_date: str,
                interval: int = 5) -> pd.DataFrame:
        """Run strategy backtest."""
        try:
            # Load historical data
            data = self.data_loader.get_minute_data(
                interval=interval,
                start_date=start_date,
                end_date=end_date
            )
            
            # Verificar se os dados foram carregados corretamente
            if data.empty:
                raise ValueError("No data loaded for the specified period")
                
            logger.info(f"Loaded {len(data)} candles for backtest")
            
            # Verificar colunas antes do processamento
            print("Columns before processing:", data.columns.tolist())
            
            # Process data and generate signals
            results = self.process_market_data(data)
            
            # Verificar colunas após o processamento
            print("Columns after processing:", results.columns.tolist())
            
            # Execute trades
            results = self.market.execute_trades(results)
            
            logger.info(f"Completed backtest from {start_date} to {end_date}")
            return results
            
        except Exception as e:
            logger.error(f"Error during backtest: {str(e)}")
            raise
    
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