# Configuration Guide

## Environment Variables
```env
DB_PATH=src/data/database/candles.db
API_KEY=your_broker_api_key
API_SECRET=your_broker_api_secret
```

## Strategy Parameters
```python
strategy_params = {
    'session_times': {
        'morning_start': '09:00',
        'morning_end': '11:00',
        'afternoon_start': '14:00',
        'afternoon_end': '16:00'
    },
    'gap_threshold': 0.2,
    'rsi_period': 14,
    'ma_fast': 9,
    'ma_slow': 21,
    'volume_profile_periods': 20
}
```

## Risk Management
```python
risk_params = {
    'max_position': 1,
    'stop_loss': 100,
    'take_profit': 200,
    'atr_multiplier': 2.0
}
```

## Machine Learning
```python
ml_params = {
    'n_estimators': 1000,
    'max_depth': 6,
    'min_samples_split': 40,
    'min_samples_leaf': 20
}
```