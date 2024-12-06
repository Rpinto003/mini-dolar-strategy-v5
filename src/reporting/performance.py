import pandas as pd
import numpy as np
from loguru import logger

class PerformanceAnalyzer:
    def __init__(self):
       pass
       
    def calculate_metrics(self, results):
       results = results.copy()
       if 'time' in results.columns:
           results.set_index(pd.to_datetime(results['time']), inplace=True)
           
       metrics = {
           'total_trades': len(results[results['trade_executed']]),
           'win_rate': (results['profit'] > 0).mean() * 100,
           'profit_factor': abs(results[results['profit'] > 0]['profit'].sum() / 
                              results[results['profit'] < 0]['profit'].sum()),
           'average_win': results[results['profit'] > 0]['profit'].mean(),
           'average_loss': abs(results[results['profit'] < 0]['profit'].mean()),
           'largest_win': results['profit'].max(),
           'largest_loss': results['profit'].min(),
           'max_drawdown': results['drawdown'].min(),
           'recovery_factor': abs(results['cumulative_profit'].iloc[-1] / 
                                results['drawdown'].min()) if results['drawdown'].min() < 0 else np.inf,
           'profit_per_trade': results['profit'].mean(),
           'sharpe_ratio': self.calculate_sharpe_ratio(results),
           'sortino_ratio': self.calculate_sortino_ratio(results),
           'calmar_ratio': self.calculate_calmar_ratio(results),
           'total_return': results['cumulative_profit'].iloc[-1] / 100000 * 100,
           'win_loss_ratio': results[results['profit'] > 0]['profit'].mean() / 
                            abs(results[results['profit'] < 0]['profit'].mean())
       }
       return metrics
   
    def calculate_sharpe_ratio(self, results, risk_free_rate=0.02):
       daily_returns = results['profit'].resample('D').sum() / 100000
       excess_returns = daily_returns - risk_free_rate/252
       return np.sqrt(252) * excess_returns.mean() / daily_returns.std()
   
    def calculate_sortino_ratio(self, results, risk_free_rate=0.02):
       daily_returns = results['profit'].resample('D').sum() / 100000
       excess_returns = daily_returns - risk_free_rate/252
       downside_returns = daily_returns[daily_returns < 0]
       downside_std = np.sqrt(np.mean(downside_returns**2))
       return np.sqrt(252) * excess_returns.mean() / downside_std if len(downside_returns) > 0 else np.inf
   
    def calculate_calmar_ratio(self, results):
       annual_return = (results['cumulative_profit'].iloc[-1] / 100000) * (252 / len(results))
       max_dd = abs(results['drawdown'].min()) / 100
       return annual_return / max_dd if max_dd > 0 else np.inf
   
    def analyze_best_conditions(self, results):
       results = results.copy()
       if 'time' in results.columns:
           results.set_index(pd.to_datetime(results['time']), inplace=True)
           
       profitable_trades = results[results['profit'] > 0]
       best_regime = profitable_trades.groupby('regime')['profit'].mean().idxmax()
       best_session = profitable_trades.index.hour.value_counts().idxmax()
       
       return f"Best Market Regime: {best_regime}\nBest Trading Hour: {best_session:02d}:00"
   
    def analyze_trade_distribution(self, results):
        results = results.copy()
        if 'time' in results.columns:
            results.set_index(pd.to_datetime(results['time']), inplace=True)
            
        # Filter for executed trades and get hourly distribution
        executed_trades = results[results['trade_executed']]
        trade_counts = pd.crosstab(
            executed_trades.index.hour,
            executed_trades['regime'],
            margins=True
        )
        
        return f"\nHourly Trade Distribution:\n{trade_counts.to_string()}"

    def generate_report(self, results):
        metrics = self.calculate_metrics(results)
        
        report = [
            "Performance Report",
            "===================\n",
            f"Total Trades: {metrics['total_trades']}",
            f"Win Rate: {metrics['win_rate']:.2f}%",
            f"Profit Factor: {metrics['profit_factor']:.2f}",
            f"Total Return: {metrics['total_return']:.2f}%\n",
            "Trade Statistics",
            "----------------",
            f"Average Win: ${metrics['average_win']:.2f}",
            f"Average Loss: ${metrics['average_loss']:.2f}",
            f"Largest Win: ${metrics['largest_win']:.2f}",
            f"Largest Loss: ${metrics['largest_loss']:.2f}",
            f"Win/Loss Ratio: {metrics['win_loss_ratio']:.2f}\n",
            "Risk Metrics",
            "------------",
            f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%",
            f"Recovery Factor: {metrics['recovery_factor']:.2f}",
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}",
            f"Sortino Ratio: {metrics['sortino_ratio']:.2f}",
            f"Calmar Ratio: {metrics['calmar_ratio']:.2f}\n",
            "Trade Analysis",
            "-------------",
            self.analyze_best_conditions(results),
            "\nTrade Distribution",
            "-----------------",
            self.analyze_trade_distribution(results)
        ]
        
        return '\n'.join(report)
     