import numpy as np
import pandas as pd
from typing import Dict, List
from scipy import stats
import empyrical as ep

class PerformanceMetrics:
    def __init__(self, risk_free_rate: float = 0.0559):  # Current SELIC rate
        self.risk_free_rate = risk_free_rate
        
    def calculate_returns(self, trades_df: pd.DataFrame) -> pd.Series:
        """Calculate returns series from trades dataframe"""
        returns = pd.Series(index=trades_df.index)
        
        for idx, row in trades_df.iterrows():
            if row['signal'] == 1:  # long
                returns[idx] = (row['exit_price'] - row['entry_price']) / row['entry_price']
            else:  # short
                returns[idx] = (row['entry_price'] - row['exit_price']) / row['entry_price']
                
            # Subtract transaction costs
            returns[idx] -= (row['commission'] + row['slippage']) / row['entry_price']
            
        return returns
    
    def calculate_metrics(self, returns: pd.Series) -> Dict:
        """Calculate comprehensive performance metrics"""
        daily_returns = returns.resample('D').sum()
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        cagr = (1 + total_return) ** (252 / len(daily_returns)) - 1
        
        # Risk metrics
        daily_std = daily_returns.std() * np.sqrt(252)
        downside_returns = daily_returns[daily_returns < 0]
        sortino_denominator = np.sqrt(np.sum(downside_returns**2) / len(downside_returns)) * np.sqrt(252)
        
        # Maximum drawdown calculation
        cum_returns = (1 + daily_returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Win rate and profit metrics
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        
        win_rate = len(winning_trades) / len(returns) if len(returns) > 0 else 0
        profit_factor = abs(winning_trades.sum() / losing_trades.sum()) if len(losing_trades) > 0 and losing_trades.sum() != 0 else np.inf
        
        # Advanced metrics
        metrics = {
            'total_return': total_return,
            'cagr': cagr,
            'sharpe_ratio': ep.sharpe_ratio(daily_returns, risk_free=self.risk_free_rate),
            'sortino_ratio': (cagr - self.risk_free_rate) / sortino_denominator if sortino_denominator != 0 else np.nan,
            'calmar_ratio': -cagr / max_drawdown if max_drawdown != 0 else np.nan,
            'max_drawdown': max_drawdown,
            'volatility': daily_std,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': winning_trades.mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades.mean() if len(losing_trades) > 0 else 0,
            'largest_win': winning_trades.max() if len(winning_trades) > 0 else 0,
            'largest_loss': losing_trades.min() if len(losing_trades) > 0 else 0,
            'avg_trade': returns.mean() if len(returns) > 0 else 0,
            'trades_count': len(returns),
            'skew': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'var_95': returns.quantile(0.05),  # 95% VaR
            'cvar_95': returns[returns <= returns.quantile(0.05)].mean()  # 95% CVaR
        }
        
        return metrics
    
    def calculate_trade_quality(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate quality metrics for individual trades"""
        trades_df = trades_df.copy()
        
        # Calculate MAE and MFE
        trades_df['mae'] = np.nan  # Maximum Adverse Excursion
        trades_df['mfe'] = np.nan  # Maximum Favorable Excursion
        
        for idx, trade in trades_df.iterrows():
            if 'price_series' in trade and trade['price_series'] is not None:
                price_data = trade['price_series']
                
                if trade['signal'] == 1:  # long
                    trades_df.loc[idx, 'mae'] = (price_data.min() - trade['entry_price']) / trade['entry_price']
                    trades_df.loc[idx, 'mfe'] = (price_data.max() - trade['entry_price']) / trade['entry_price']
                else:  # short
                    trades_df.loc[idx, 'mae'] = (trade['entry_price'] - price_data.max()) / trade['entry_price']
                    trades_df.loc[idx, 'mfe'] = (trade['entry_price'] - price_data.min()) / trade['entry_price']
        
        # Calculate trade efficiency
        trades_df['efficiency'] = trades_df.apply(
            lambda x: x['return'] / x['mfe'] if x['mfe'] > 0 else 0, axis=1
        )
        
        return trades_df