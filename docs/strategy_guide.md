# Mini Dollar Strategy Guide

## Overview
This strategy combines technical analysis, machine learning, and risk management for trading Mini Dollar futures (WDO).

## Components

### Market Analysis
- ADX and Bollinger Bands for regime detection
- Dynamic volume profile analysis
- Orderflow indicators including VWAP and delta

### Signal Generation
- Random Forest model (1000 estimators)
- Feature engineering with technical indicators
- Risk-adjusted ML labels

### Risk Management
- Dynamic position sizing
- Multi-level take-profits
- Adaptive stop-loss rules

## Configuration

### Session Times
```python
session_times = {
    'morning_start': '09:00',
    'morning_end': '11:00',
    'afternoon_start': '14:00',
    'afternoon_end': '16:00'
}
```

### Risk Parameters
- Initial stop-loss: 2.5 × ATR (trending)
- Initial stop-loss: 1.8 × ATR (ranging)
- Take-profit levels: 2.0/3.0/4.0 × ATR

## Performance Metrics
- Win rate
- Risk-adjusted returns
- Drawdown analysis
- Regime performance