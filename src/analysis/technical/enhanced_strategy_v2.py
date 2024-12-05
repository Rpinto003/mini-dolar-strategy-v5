import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import talib

class EnhancedTechnicalStrategyV2:
    def __init__(self, rsi_period=14, ma_fast=9, ma_slow=21):
        self.rsi_period = rsi_period
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.model = RandomForestClassifier(
            n_estimators=1000,
            max_depth=6,
            min_samples_split=40,
            min_samples_leaf=20,
            random_state=42
        )
        self.scaler = StandardScaler()

    def detect_market_regime(self, data):
        df = data.copy()
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['bbands_upper'], df['bbands_middle'], df['bbands_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        
        df['regime'] = 'ranging'
        df.loc[df['adx'] > 25, 'regime'] = 'trending'
        
        return df

    def prepare_features(self, data):
        df = data.copy()
        
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.rsi_period)
        df['macd'], df['macd_signal'], _ = talib.MACD(df['close'])
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['ma_fast'] = talib.EMA(df['close'], timeperiod=self.ma_fast)
        df['ma_slow'] = talib.EMA(df['close'], timeperiod=self.ma_slow)
        
        df = self.detect_market_regime(df)
        
        feature_cols = ['rsi', 'macd', 'atr', 'adx']
        X = df[feature_cols].copy()
        X = X.fillna(method='ffill').fillna(method='bfill')
        X = self.scaler.fit_transform(X)
        
        return X, df

    def generate_signals(self, data):
        X, df = self.prepare_features(data)
        
        df['ml_prob'] = 0.5
        df['signal'] = 0
        df['final_signal'] = 0
        
        long_conditions = (
            (df['regime'] == 'trending') &
            (df['rsi'] < 70) &
            (df['macd'] > df['macd_signal'])
        )
        
        short_conditions = (
            (df['regime'] == 'trending') &
            (df['rsi'] > 30) &
            (df['macd'] < df['macd_signal'])
        )
        
        df.loc[long_conditions, 'signal'] = 1
        df.loc[short_conditions, 'signal'] = -1
        df['final_signal'] = df['signal']
        
        return df

    def add_risk_management(self, data):
        df = data.copy()
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        multiplier = df['regime'].map({'trending': 2.5, 'ranging': 1.8})
        df['dynamic_stop_loss'] = df['atr'] * multiplier
        
        df['stop_loss'] = np.where(
            df['signal'] == 1,
            df['close'] - df['dynamic_stop_loss'],
            df['close'] + df['dynamic_stop_loss']
        )
        
        df['take_profit_1'] = df['close'] + df['atr'] * 2.0
        df['take_profit_2'] = df['close'] + df['atr'] * 3.0
        df['take_profit_3'] = df['close'] + df['atr'] * 4.0
        
        df['breakeven_level'] = df['close'] + df['atr']
        
        return df