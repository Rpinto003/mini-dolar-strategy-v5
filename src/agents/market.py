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
        
        self.initial_balance = initial_balance
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
        df['cumulative_profit'] = 0.0
        df['drawdown'] = 0.0
        
        for i in range(1, len(df)):
            current_bar = df.iloc[i]
            prev_bar = df.iloc[i-1]
            
            # Check stop loss and take profit
            if self.position != 0:
                if self.check_exit_conditions(current_bar):
                    exit_price = self.calculate_exit_price(current_bar)
                    trade_profit = self.calculate_trade_profit(exit_price)
                    
                    df.loc[df.index[i], 'position'] = 0
                    df.loc[df.index[i], 'trade_executed'] = True
                    df.loc[df.index[i], 'profit'] = trade_profit
                    self.balance += trade_profit
                    self.position = 0
                    continue
            
            # Dynamic position sizing based on volatility and account balance
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
            
            # Update cumulative metrics
            df.loc[df.index[i], 'cumulative_profit'] = self.balance - self.initial_balance
            df.loc[df.index[i], 'drawdown'] = self.calculate_drawdown(df.loc[:df.index[i], 'cumulative_profit'])
        
        return df
    
    def validate_long_entry(self, bar):
        return (
            bar['session_active'] and
            bar['volume_ratio'] > 1.0 and
            bar['pressure_ratio'] > 1.2 and
            bar['regime'] == 'trending'
        )
    
    def validate_short_entry(self, bar):
        return (
            bar['session_active'] and
            bar['volume_ratio'] > 1.0 and
            bar['pressure_ratio'] < 0.8 and
            bar['regime'] == 'trending'
        )
    
    def check_exit_conditions(self, bar):
        if self.position > 0:  # Long position
            stop_hit = bar['low'] <= bar['stop_loss']
            tp1_hit = bar['high'] >= bar['take_profit_1']
            tp2_hit = bar['high'] >= bar['take_profit_2']
            tp3_hit = bar['high'] >= bar['take_profit_3']
            
            if bar['high'] >= bar['breakeven_level']:
                bar['stop_loss'] = max(self.entry_price, bar['stop_loss'])
            
            return stop_hit or tp1_hit or tp2_hit or tp3_hit
            
        elif self.position < 0:  # Short position
            stop_hit = bar['high'] >= bar['stop_loss']
            tp1_hit = bar['low'] <= bar['take_profit_1']
            tp2_hit = bar['low'] <= bar['take_profit_2']
            tp3_hit = bar['low'] <= bar['take_profit_3']
            
            if bar['low'] <= bar['breakeven_level']:
                bar['stop_loss'] = min(self.entry_price, bar['stop_loss'])
                
            return stop_hit or tp1_hit or tp2_hit or tp3_hit
            
        return False
    
    def calculate_position_size(self, bar):
        base_size = self.max_position * (self.balance / self.initial_balance)
        
        # Adjust size based on volatility
        volatility_factor = min(1.0, 1.0 / (bar['atr'] / bar['close'] * 100))
        
        # Adjust size based on market regime
        regime_factor = 1.0 if bar['regime'] == 'trending' else 0.7
        
        # Adjust size based on signal strength
        signal_factor = min(1.0, abs(bar['ml_prob'] - 0.5) * 2)
        
        # Risk-based position sizing
        risk_factor = min(1.0, self.stop_loss / (bar['atr'] * self.atr_multiplier))
        
        return base_size * volatility_factor * regime_factor * signal_factor * risk_factor
    
    def calculate_exit_price(self, bar):
        if self.position > 0:  # Long position
            if bar['low'] <= bar['stop_loss']:
                return bar['stop_loss']
            elif bar['high'] >= bar['take_profit_3']:
                return bar['take_profit_3']
            elif bar['high'] >= bar['take_profit_2']:
                return bar['take_profit_2']
            elif bar['high'] >= bar['take_profit_1']:
                return bar['take_profit_1']
        
        elif self.position < 0:  # Short position
            if bar['high'] >= bar['stop_loss']:
                return bar['stop_loss']
            elif bar['low'] <= bar['take_profit_3']:
                return bar['take_profit_3']
            elif bar['low'] <= bar['take_profit_2']:
                return bar['take_profit_2']
            elif bar['low'] <= bar['take_profit_1']:
                return bar['take_profit_1']
        
        return bar['close']
    
    def calculate_trade_profit(self, exit_price):
        return self.position * (exit_price - self.entry_price)
    
    def calculate_drawdown(self, cumulative_profits):
        peak = cumulative_profits.expanding().max()
        drawdown = (cumulative_profits - peak) / peak * 100
        return drawdown.iloc[-1]
    
    def calculate_max_drawdown(self, data):
        cumulative_returns = data['profit'].cumsum()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max * 100
        return drawdowns.min()
    
    def calculate_sharpe_ratio(self, data, risk_free_rate=0.02):
        daily_returns = data['profit'] / self.initial_balance
        excess_returns = daily_returns - risk_free_rate/252
        return np.sqrt(252) * excess_returns.mean() / daily_returns.std()
    
    def calculate_sortino_ratio(self, data, risk_free_rate=0.02):
        daily_returns = data['profit'] / self.initial_balance
        excess_returns = daily_returns - risk_free_rate/252
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = np.sqrt(np.mean(downside_returns**2))
        return np.sqrt(252) * excess_returns.mean() / downside_std if len(downside_returns) > 0 else np.inf