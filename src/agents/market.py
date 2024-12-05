import pandas as pd
import numpy as np
from loguru import logger

class MarketAgent:
    def __init__(self, 
                 initial_balance=100000,
                 max_position=1,
                 stop_loss=100,
                 take_profit=200,
                 atr_multiplier=2.0):
        
        self.balance = initial_balance
        self.max_position = max_position
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.atr_multiplier = atr_multiplier
        self.position = 0
        self.entry_price = 0
        
    def execute_trades(self, data):
        df = data.copy()
        df['position'] = 0
        df['trade_executed'] = False
        df['profit'] = 0.0
        
        for i in range(1, len(df)):
            current_bar = df.iloc[i]
            prev_bar = df.iloc[i-1]
            
            # Check stop loss and take profit
            if self.position != 0:
                if self.check_exit_conditions(current_bar):
                    df.loc[df.index[i], 'position'] = 0
                    df.loc[df.index[i], 'trade_executed'] = True
                    self.position = 0
                    continue
            
            # Dynamic position sizing based on volatility
            position_size = self.calculate_position_size(current_bar)
            
            # Entry signals
            if self.position == 0:
                if current_bar['final_signal'] == 1 and self.validate_long_entry(current_bar):
                    self.position = position_size
                    self.entry_price = current_bar['close']
                    df.loc[df.index[i], 'position'] = position_size
                    df.loc[df.index[i], 'trade_executed'] = True
                    
                elif current_bar['final_signal'] == -1 and self.validate_short_entry(current_bar):
                    self.position = -position_size
                    self.entry_price = current_bar['close']
                    df.loc[df.index[i], 'position'] = -position_size
                    df.loc[df.index[i], 'trade_executed'] = True
            
            # Calculate trade profit/loss
            if df.loc[df.index[i], 'trade_executed']:
                df.loc[df.index[i], 'profit'] = self.calculate_profit(
                    current_bar['close'],
                    prev_bar['close'],
                    self.position
                )
        
        return df
    
    def validate_long_entry(self, bar):
        return (
            bar['session_active'] and
            bar['volume_ratio'] > 1.0 and
            bar['pressure_ratio'] > 1.2
        )
    
    def validate_short_entry(self, bar):
        return (
            bar['session_active'] and
            bar['volume_ratio'] > 1.0 and
            bar['pressure_ratio'] < 0.8
        )
    
    def check_exit_conditions(self, bar):
        if self.position > 0:
            # Long position
            stop_hit = bar['low'] <= bar['stop_loss']
            take_profit_hit = bar['high'] >= bar['take_profit_1']
            
            if bar['high'] >= bar['breakeven_level']:
                # Move stop to breakeven
                bar['stop_loss'] = self.entry_price
            
            return stop_hit or take_profit_hit
            
        elif self.position < 0:
            # Short position
            stop_hit = bar['high'] >= bar['stop_loss']
            take_profit_hit = bar['low'] <= bar['take_profit_1']
            
            if bar['low'] <= bar['breakeven_level']:
                # Move stop to breakeven
                bar['stop_loss'] = self.entry_price
                
            return stop_hit or take_profit_hit
            
        return False
    
    def calculate_position_size(self, bar):
        base_size = self.max_position
        
        # Adjust size based on volatility
        volatility_factor = min(1.0, 1.0 / (bar['atr'] / bar['close'] * 100))
        
        # Adjust size based on market regime
        regime_factor = 1.0 if bar['regime'] == 'trending' else 0.7
        
        # Adjust size based on signal strength
        signal_factor = min(1.0, abs(bar['ml_prob'] - 0.5) * 2)
        
        return base_size * volatility_factor * regime_factor * signal_factor
    
    def calculate_profit(self, current_price, prev_price, position):
        if position == 0:
            return 0.0
            
        price_change = current_price - prev_price
        return position * price_change
    
    def calculate_max_drawdown(self, data):
        cumulative_returns = data['profit'].cumsum()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns - rolling_max
        return drawdowns.min()
    
    def calculate_sharpe_ratio(self, data, risk_free_rate=0.02):
        returns = data['profit'] / self.initial_balance
        excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / returns.std()
