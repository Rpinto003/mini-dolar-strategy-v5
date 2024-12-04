from typing import Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

class PerformanceAnalyzer:
    """Analyzes and reports trading strategy performance."""
    
    def __init__(self):
        """Initialize performance analyzer."""
        pass
    
    def calculate_metrics(self, results: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            results: DataFrame with trading results
            
        Returns:
            Dictionary of performance metrics
        """
        if results.empty:
            raise ValueError("Results DataFrame cannot be empty")
            
        required_columns = ['trade_executed', 'profit', 'balance']
        if not all(col in results.columns for col in required_columns):
            raise KeyError(f"Results must contain columns: {required_columns}")
        
        trades = results[results['trade_executed']]
        profits = trades['profit']
        
        metrics = {
            'total_trades': len(trades),
            'winning_trades': len(profits[profits > 0]),
            'losing_trades': len(profits[profits < 0]),
            'win_rate': len(profits[profits > 0]) / len(profits) if len(profits) > 0 else 0,
            'total_profit': profits.sum(),
            'average_profit': profits.mean() if len(profits) > 0 else 0,
            'profit_std': profits.std() if len(profits) > 0 else 0,
            'max_profit': profits.max() if len(profits) > 0 else 0,
            'max_loss': profits.min() if len(profits) > 0 else 0,
            'sharpe_ratio': self._calculate_sharpe(results),
            'max_drawdown': self._calculate_max_drawdown(results)
        }
        
        return metrics
    
    def generate_report(self,
                       results: pd.DataFrame,
                       output_path: Optional[str] = None) -> str:
        """
        Generate detailed performance report.
        
        Args:
            results: DataFrame with trading results
            output_path: Optional path to save report
            
        Returns:
            Report text content
        """
        metrics = self.calculate_metrics(results)
        
        report = [
            "Trading Strategy Performance Report",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "Performance Metrics:",
            f"Total Trades: {metrics['total_trades']}",
            f"Win Rate: {metrics['win_rate']:.2%}",
            f"Total Profit: {metrics['total_profit']:.2f}",
            f"Average Profit per Trade: {metrics['average_profit']:.2f}",
            f"Maximum Drawdown: {metrics['max_drawdown']:.2%}",
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n",
            "Trade Statistics:",
            f"Winning Trades: {metrics['winning_trades']}",
            f"Losing Trades: {metrics['losing_trades']}",
            f"Best Trade: {metrics['max_profit']:.2f}",
            f"Worst Trade: {metrics['max_loss']:.2f}"
        ]
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Saved performance report to {output_path}")
        
        return report_text
    
    def plot_equity_curve(self, results: pd.DataFrame):
        """Plot equity curve with drawdowns."""
        plt.figure(figsize=(15, 7))
        
        # Plot equity curve
        plt.subplot(2, 1, 1)
        plt.plot(results.index, results['balance'])
        plt.title('Equity Curve')
        plt.grid(True)
        
        # Plot drawdowns
        plt.subplot(2, 1, 2)
        drawdowns = self._calculate_drawdown_series(results)
        plt.plot(results.index, drawdowns)
        plt.title('Drawdowns')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def _calculate_sharpe(self, 
                         results: pd.DataFrame, 
                         risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        returns = results['balance'].pct_change().dropna()
        excess_returns = returns - risk_free_rate/252
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    def _calculate_max_drawdown(self, results: pd.DataFrame) -> float:
        """Calculate maximum drawdown percentage."""
        return self._calculate_drawdown_series(results).min()
    
    def _calculate_drawdown_series(self, results: pd.DataFrame) -> pd.Series:
        """Calculate drawdown series."""
        balance = results['balance']
        running_max = balance.expanding().max()
        drawdowns = (balance - running_max) / running_max
        return drawdowns