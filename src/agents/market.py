from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger

class MarketAgent:
    """Handles market interactions and trade execution."""
    
    def __init__(self,
                 initial_balance: float = 100000,
                 max_position: int = 1,
                 stop_loss: float = 100,
                 take_profit: float = 200):
        """
        Initialize the market agent.
        
        Args:
            initial_balance: Starting balance
            max_position: Maximum position size
            stop_loss: Stop loss in points
            take_profit: Take profit in points
        """
        self.initial_balance = initial_balance #add RLCP [GPT]
        self.balance = initial_balance
        self.max_position = max_position
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position = 0
        self.entry_price = 0
        
        logger.info(f"Initialized MarketAgent with balance={initial_balance}")
    
    def execute_trades(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Execute trades based on signals.
        
        Args:
            data: DataFrame with trading signals
            
        Returns:
            DataFrame with trade executions
        """
        df = data.copy()
        df['trade_executed'] = False
        df['profit'] = 0.0
        df['balance'] = self.initial_balance
        
        for i in range(1, len(df)):
            current_signal = df.iloc[i]['final_signal']
            current_price = df.iloc[i]['close']
            
            # Check for stop loss or take profit
            if self.position != 0:
                profit = self._calculate_profit(current_price)
                if abs(profit) >= self.take_profit or abs(profit) >= self.stop_loss:
                    self._close_position(df, i, current_price)
            
            # Execute new trades based on signals
            if current_signal != 0 and self.position == 0:
                self._open_position(df, i, current_price, current_signal)
        
        return df
    
    def _calculate_profit(self, current_price: float) -> float:
        """Calculate current profit/loss."""
        if self.position == 0:
            return 0
        
        points = (current_price - self.entry_price) * self.position
        return points
    
    def _open_position(self, 
                      df: pd.DataFrame, 
                      index: int, 
                      price: float, 
                      direction: int):
        """Open a new trading position."""
        self.position = self.max_position * direction
        self.entry_price = price
        df.loc[df.index[index], 'trade_executed'] = True
        
        logger.info(f"Opened position: Direction={direction}, Price={price}")
    
    def _close_position(self, 
                       df: pd.DataFrame, 
                       index: int, 
                       price: float):
        """Close current trading position."""
        profit = self._calculate_profit(price)
        df.loc[df.index[index], 'trade_executed'] = True
        df.loc[df.index[index], 'profit'] = profit
        self.balance += profit
        df.loc[df.index[index], 'balance'] = float(self.balance)
        
        self.position = 0
        self.entry_price = 0
        
        logger.info(f"Closed position: Profit={profit}, Balance={self.balance}")
    
    def calculate_max_drawdown(self, results: pd.DataFrame) -> float:
        """Calculate maximum drawdown from results."""
        balance = results['balance']
        rolling_max = balance.expanding().max()
        drawdowns = (balance - rolling_max) / rolling_max
        return drawdowns.min() * 100
    
    def calculate_sharpe_ratio(self, 
                             results: pd.DataFrame,
                             risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from results."""
        returns = results['balance'].pct_change().dropna()
        excess_returns = returns - risk_free_rate/252
        sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
        return sharpe