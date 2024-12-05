# Mini Dollar Strategy v4

Advanced trading strategy for Mini Dollar futures (WDO) combining technical analysis, machine learning, and automated execution.

## Strategy Overview

The strategy combines multiple analysis approaches:

### Market Analysis
- Market regime detection using ADX and Bollinger Bands
- Volume profile analysis with dynamic price bins
- VWAP and cumulative delta calculations
- Enhanced orderflow indicators

### Signal Generation
- Random Forest model with optimized parameters
- Feature engineering for price action patterns
- Risk-adjusted ML labels
- Multi-timeframe analysis

### Risk Management
- Dynamic position sizing based on:
  - Market volatility
  - Account balance
  - Signal strength
  - Market regime
- Multiple take-profit levels
- Adaptive stop-loss with breakeven rules
- Maximum drawdown controls

## Project Structure

```
src/
├── agents/            # Trading agents and coordination
│   ├── coordinator.py # Strategy coordinator
│   └── market.py     # Market execution agent
├── analysis/         # Technical analysis
│   └── technical/    # Trading strategies
│       └── enhanced_strategy_v2.py  # Main strategy
├── data/            # Data management
│   ├── database/    # Market data storage
│   └── loaders/     # Data loading utilities
├── models/          # Machine learning
│   └── ml/         # ML models and training
└── reporting/       # Analytics
    └── performance.py # Performance metrics
```

## Key Features

### Technical Analysis
- Advanced market regime detection
- Volume profile analysis
- Dynamic support/resistance levels
- Multi-timeframe momentum indicators

### Machine Learning
- Random Forest model with 1000 estimators
- Feature importance analysis
- Enhanced feature engineering
- Risk-adjusted labeling

### Risk Management
- Volatility-based position sizing
- Multi-level take-profit strategy
- Dynamic stop-loss adjustment
- Maximum drawdown control

### Performance Analytics
- Comprehensive metrics calculation
- Risk-adjusted return analysis
- Trade distribution analysis
- Market regime performance tracking

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Rpinto003/mini-dolar-strategy-v4.git
cd mini-dolar-strategy-v4
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

## Usage

### Configuration
Set your trading parameters in `config.yaml`:
- Initial balance
- Risk limits
- Trading hours
- Take-profit levels

### Backtesting
```python
from src.agents.coordinator import StrategyCoordinator

coordinator = StrategyCoordinator(
    initial_balance=100000,
    max_position=1,
    stop_loss=100,
    take_profit=200
)

results = coordinator.backtest(
    start_date='2024-01-01',
    end_date='2024-12-01',
    interval=5
)
```

### Performance Analysis
```python
from src.reporting.performance import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()
metrics = analyzer.calculate_metrics(results)
report = analyzer.generate_report(results)
```

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Performance Monitoring

Monitor key metrics:
- Win rate
- Risk-adjusted returns
- Maximum drawdown
- Sharpe/Sortino ratios

## Documentation

See `docs/` for detailed documentation:
- Strategy guide
- API reference
- Configuration options
- Performance metrics

## License

MIT License