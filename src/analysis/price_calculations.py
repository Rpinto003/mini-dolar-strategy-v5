from typing import Dict, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class TransactionPrices:
    entry_price: float
    exit_price: float
    slippage: float
    commission: float
    
class PriceCalculator:
    def __init__(self, slippage_factor: float = 0.0002, commission_rate: float = 0.0003):
        self.slippage_factor = slippage_factor
        self.commission_rate = commission_rate
        
    def calculate_entry_price(self, row: pd.Series, signal: int) -> float:
        """
        Calculate entry price considering market impact and slippage
        
        Args:
            row: DataFrame row with OHLCV data
            signal: 1 for long, -1 for short
        """
        base_price = row['close']
        volume_impact = min(0.0001, (row['volume'] / row['volume_ma']) * 0.0001)
        slippage = base_price * self.slippage_factor * (1 + volume_impact)
        
        # Add slippage for longs, subtract for shorts
        return base_price + (signal * slippage)
    
    def calculate_exit_price(self, 
                           row: pd.Series,
                           entry_price: float,
                           signal: int,
                           exit_type: str) -> Tuple[float, Dict]:
        """
        Calculate exit price based on exit type and market conditions
        
        Args:
            row: DataFrame row with OHLCV data
            entry_price: Original entry price
            signal: 1 for long, -1 for short
            exit_type: 'stop_loss', 'take_profit', or 'market'
        """
        base_price = row['close']
        
        # Adjust slippage based on exit type
        if exit_type == 'stop_loss':
            slippage_mult = 2.0  # Higher slippage for stops
        elif exit_type == 'take_profit':
            slippage_mult = 0.8  # Lower slippage for take profits
        else:
            slippage_mult = 1.0  # Normal slippage for market exits
            
        volume_impact = min(0.0001, (row['volume'] / row['volume_ma']) * 0.0001)
        slippage = base_price * self.slippage_factor * slippage_mult * (1 + volume_impact)
        
        # Calculate commission
        commission = base_price * self.commission_rate
        
        # Calculate exit price
        exit_price = base_price - (signal * slippage)
        
        metrics = {
            'slippage': slippage,
            'commission': commission,
            'volume_impact': volume_impact
        }
        
        return exit_price, metrics
    
    def calculate_position_size(self,
                              capital: float,
                              risk_per_trade: float,
                              stop_loss: float,
                              entry_price: float) -> float:
        """
        Calculate position size based on risk management parameters
        
        Args:
            capital: Available capital
            risk_per_trade: Maximum risk per trade as percentage
            stop_loss: Stop loss price
            entry_price: Entry price
        """
        risk_amount = capital * risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        
        # Include commission in position sizing
        total_commission = entry_price * self.commission_rate * 2  # Entry + exit
        position_size = (risk_amount - total_commission) / price_risk
        
        return round(position_size, 2)  # Round to 2 decimals for mini-dollar