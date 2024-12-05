# Mini Dollar Strategy v4

Advanced trading strategy for Mini Dollar futures (WDO) combining technical analysis, machine learning, and automated execution.

## Implementation Steps

### 1. Market Analysis
- Market regime detection using ADX and Bollinger Bands
- Volume profile analysis with dynamic price bins
- Advanced orderflow indicators:
  - VWAP calculation
  - Cumulative delta
  - Pressure ratios

### 2. Machine Learning
- Random Forest model:
  - 1000 estimators
  - Max depth: 6
  - Min samples split: 40
  - Min samples leaf: 20
- Feature engineering:
  - Technical indicators
  - Momentum measures
  - Volume analysis
  - Market regime features

### 3. Risk Management
- Dynamic position sizing based on:
  - Volatility adjustment
  - Market regime factor
  - Signal strength scaling
  - Risk-based limits
- Multi-level take-profits:
  - TP1: 2.0 × ATR
  - TP2: 3.0 × ATR
  - TP3: 4.0 × ATR
- Adaptive stop-loss:
  - Initial: 2.5 × ATR (trending)
  - Initial: 1.8 × ATR (ranging)
  - Breakeven adjustment

### 4. Performance Analytics
- Trade metrics:
  - Win rate
  - Profit factor
  - Average win/loss
- Risk metrics:
  - Sharpe ratio
  - Sortino ratio
  - Maximum drawdown
- Market analysis:
  - Regime performance
  - Session analysis
  - Volume profile impact

## Project Structure

```
src/
├── agents/            # Trading agents and coordination
│   ├── coordinator.py  # Strategy coordinator
│   └── market.py      # Market execution
├── analysis/          # Technical analysis
│   └── technical/     # Trading strategies
├── data/             # Data management
│   ├── database/     # Market data storage
│   └── loaders/      # Data loading
├── models/           # Machine learning
│   └── ml/          # ML models
└── reporting/        # Analytics
    └── performance.py # Performance metrics
```

## Usage Example

```python
# Initialize strategy
coordinator = StrategyCoordinator(
    initial_balance=100000,
    max_position=1,
    stop_loss=100,
    take_profit=200
)

# Run backtest
results = coordinator.backtest(
    start_date='2024-01-01',
    end_date='2024-12-01',
    interval=5
)

# Analyze performance
analyzer = PerformanceAnalyzer()
metrics = analyzer.calculate_metrics(results)
report = analyzer.generate_report(results)
```

## Installation

1. Clone repository:
```bash
git clone https://github.com/Rpinto003/mini-dolar-strategy-v4.git
cd mini-dolar-strategy-v4
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Configure settings:
```bash
cp .env.example .env
# Edit .env with your parameters
```

## Documentation

Detailed documentation in `docs/`:
- Strategy guide
- API reference
- Configuration options
- Performance metrics

## Testing

```bash
python -m pytest tests/
```

## License

MIT License
