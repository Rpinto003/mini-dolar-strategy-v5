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
                 take_profit: float = 200,
                 atr_multiplier: float = 2.0):
        """
        Initialize the market agent.
        
        Args:
            initial_balance: Starting balance
            max_position: Maximum position size
            stop_loss: Stop loss in points
            take_profit: Take profit in points
            atr_multiplier: Multiplier for ATR to calculate stops
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_position = max_position
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.atr_multiplier = atr_multiplier
        self.position = 0
        self.entry_price = 0
        
        logger.info(f"Initialized MarketAgent with balance={initial_balance}")
    
    def execute_trades(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        position = 0
        entry_price = 0.0
        total_profit = 0.0
        balance = self.initial_balance
        current_tp_index = 0
        last_cum_return = 1.0

        # Inicializar colunas
        df['trade_executed'] = False
        df['profit'] = 0.0
        df['balance'] = balance
        df['position'] = 0
        df['returns'] = 0.0
        df['cumulative_returns'] = 1.0

        for i, (index, row) in enumerate(df.iterrows()):
            signal = row['final_signal']
            profit = 0.0
            trade_executed = False

            # Abrir posição
            if position == 0 and signal != 0:
                position = signal * self.max_position
                entry_price = row['close']
                stop_loss = entry_price - row['dynamic_stop_loss'] if position > 0 else entry_price + row['dynamic_stop_loss']
                take_profit_levels = [row['take_profit_level1'], row['take_profit_level2'], row['take_profit_level3']]
                current_tp_index = 0
                trailing_stop = stop_loss
                trade_executed = True
                logger.info(f"Opened position at {entry_price}, position size: {position}")
            
            elif position != 0:
                new_trailing_stop = row['close'] - row['atr'] * self.atr_multiplier if position > 0 else row['close'] + row['atr'] * self.atr_multiplier
                trailing_stop = max(trailing_stop, new_trailing_stop) if position > 0 else min(trailing_stop, new_trailing_stop)
                
                # Take profit parcial
                if current_tp_index < len(take_profit_levels) and (
                    (position > 0 and row['close'] >= take_profit_levels[current_tp_index]) or
                    (position < 0 and row['close'] <= take_profit_levels[current_tp_index])
                ):
                    partial_position = self.max_position / len(take_profit_levels)
                    partial_profit = (take_profit_levels[current_tp_index] - entry_price) * partial_position * np.sign(position)
                    total_profit += partial_profit
                    balance += partial_profit
                    profit += partial_profit
                    position -= partial_position * np.sign(position)
                    current_tp_index += 1
                    trade_executed = True
                
                # Stop loss
                elif (position > 0 and row['close'] <= trailing_stop) or (position < 0 and row['close'] >= trailing_stop):
                    profit = (row['close'] - entry_price) * position
                    total_profit += profit
                    balance += profit
                    position = 0
                    trade_executed = True

            # Atualizar DataFrame
            df.at[index, 'trade_executed'] = trade_executed
            df.at[index, 'position'] = position
            df.at[index, 'profit'] = profit
            df.at[index, 'balance'] = balance
            
            returns = profit / balance if balance > 0 else 0
            df.at[index, 'returns'] = returns
            
            if i > 0:
                last_cum_return = df.iloc[i-1]['cumulative_returns']
            df.at[index, 'cumulative_returns'] = last_cum_return * (1 + returns)

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