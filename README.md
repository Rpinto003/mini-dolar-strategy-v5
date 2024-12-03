# Mini Dollar Strategy v4

Advanced trading strategy for Mini Dollar futures (WDO) combining technical analysis, machine learning, and automated execution.

## Project Structure

```
src/
├── agents/            # Trading agents and coordination
│   ├── coordinator.py  # Main strategy coordinator
│   └── market.py      # Market interaction agent
├── data/              # Data management
│   ├── database/      # SQLite database
│   └── loaders/       # Data loading utilities
├── analysis/          # Analysis modules
│   └── technical/     # Technical analysis strategies
├── models/            # ML models
│   └── lstm.py       # LSTM prediction model
└── reporting/         # Performance reporting
    └── performance.py # Performance metrics
```

## Features

- Advanced technical analysis with multiple indicators
- LSTM-based price prediction
- Real-time market data processing
- Automated trading execution
- Performance monitoring and reporting

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

3. Configure the environment:
```bash
cp .env.example .env
# Edit .env with your configurations
```

## Usage

Check the notebooks directory for usage examples:
- `1_data_analysis.ipynb`: Data loading and analysis
- `2_strategy_testing.ipynb`: Strategy backtesting
- `3_live_trading.ipynb`: Live trading setup

## Documentation

Detailed documentation for each module is available in the `docs/` directory.

## Testing

Run tests with pytest:
```bash
python -m pytest tests/
```

## License

MIT License