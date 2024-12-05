# API Reference

## StrategyCoordinator

```python
class StrategyCoordinator:
    def __init__(self,
                 initial_balance: float = 100000,
                 max_position: int = 1,
                 stop_loss: float = 100,
                 take_profit: float = 200,
                 db_path: Optional[str] = None,
                 strategy_params: Optional[Dict] = None)
```

### Methods

#### backtest
```python
def backtest(self,
            start_date: str,
            end_date: str,
            interval: int = 5) -> pd.DataFrame
```

#### process_market_data
```python
def process_market_data(self,
                      data: pd.DataFrame) -> pd.DataFrame
```

## MarketAgent

```python
class MarketAgent:
    def __init__(self,
                 initial_balance: float = 100000,
                 max_position: int = 1,
                 stop_loss: float = 100,
                 take_profit: float = 200,
                 atr_multiplier: float = 2.0)
```

### Methods

#### execute_trades
```python
def execute_trades(self,
                   data: pd.DataFrame) -> pd.DataFrame
```

#### calculate_metrics
```python
def calculate_metrics(self,
                      results: pd.DataFrame) -> Dict
```