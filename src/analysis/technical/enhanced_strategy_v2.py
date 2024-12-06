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

        df = self.check_session_time(df)
        df = self.detect_market_regime(df)
        df['trend_strength'] = df['adx']

        # Add more predictive features
        df['trend_strength'] = df['adx']
        df['price_momentum'] = df['close'].pct_change(5)
        df['volume_trend'] = df['volume'].pct_change(5)
        
        # Technical indicators
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.rsi_period)
        df['macd'], df['macd_signal'], _ = talib.MACD(df['close'])
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Additional features
        df['ma_fast'] = talib.EMA(df['close'], timeperiod=self.ma_fast)
        df['ma_slow'] = talib.EMA(df['close'], timeperiod=self.ma_slow)
        # Handle missing 'bb_middle' column if it does not exist
        if 'bb_middle' not in df.columns:
            df['bb_middle'] = talib.SMA(df['close'], timeperiod=20)
        df['bollinger_distance'] = (df['close'] - df['bb_middle']) / df['atr']  # Normalized BB position

        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['pressure_ratio'] = (df['high'] - df['close']) / (df['close'] - df['low'])

        # Ensure correct dtypes to avoid warnings
        df['volume_ma'] = df['volume_ma'].astype(float)
        df['bollinger_distance'] = df['bollinger_distance'].astype(float)
        df['volume_ratio'] = df['volume_ratio'].astype(float)
        df['pressure_ratio'] = df['pressure_ratio'].astype(float)

        # Session and regime
        df = self.check_session_time(df)
        df = self.detect_market_regime(df)

        # Define features for modeling
        feature_cols = [
            'rsi', 'macd', 'atr', 'adx', 'trend_strength', 'price_momentum', 'volume_trend',
            'ma_fast', 'ma_slow', 'bollinger_distance', 'volume_ratio', 'pressure_ratio'
        ]
        
        # Handle missing values with forward and backward fill
        df[feature_cols] = df[feature_cols].ffill().bfill()

        # Scale the features for the model
        X = self.scaler.fit_transform(df[feature_cols])
        
        return X, df

    def generate_signals(self, data):
        X, df = self.prepare_features(data)
        
        # Train the model first
        df['target'] = (df['close'].shift(-10) > df['close']).astype(int)
        
        # Convert X to numpy array if not already
        X_array = np.array(X)
        
        # Remove last 10 rows that have NaN target
        train_idx = df.index[:-10]
        train_data = X_array[:len(train_idx)]
        train_target = df['target'].iloc[:-10]
        
        self.model.fit(train_data, train_target)
        
        # Generate predictions
        df['ml_prob'] = self.model.predict_proba(X_array)[:, 1]
        df['signal'] = 0
        df['final_signal'] = 0

        # Add trend confirmation
        df['trend_strength'] = df['adx'] > 30  # Increase from 25
        df['trend_direction'] = (df['ma_fast'] > df['ma_slow']) & (df['close'] > df['ma_fast'])        

        # Rest of your existing code...
        long_conditions = (
            (df['regime'] == 'trending') &
            (df['rsi'] < 60) &
            (df['trend_strength']) &
            (df['trend_direction']) &
            (df['macd'] > df['macd_signal']) &
            (df['volume_ratio'] > 1.2)  # Stronger volume confirmation
        )
        
        short_conditions = (
            (df['regime'] == 'trending') &
            (df['rsi'] > 40) &  # Changed from 30 for symmetry
            (df['trend_strength']) &
            (~df['trend_direction']) &  # Opposite of long trend
            (df['macd'] < df['macd_signal']) &
            (df['volume_ratio'] > 1.2)  # Same volume confirmation
        )
        
        df.loc[long_conditions, 'signal'] = 1
        df.loc[short_conditions, 'signal'] = -1
        df['final_signal'] = df['signal']
        
        return df

    def add_risk_management(self, data):
        df = data.copy()
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Initialize columns
        for col in ['stop_loss', 'take_profit_1', 'take_profit_2', 'take_profit_3', 'breakeven_level']:
            df[col] = df['close']
        
        # Long positions setup
        long_mask = df['signal'] == 1
        df.loc[long_mask, 'stop_loss'] = df['close'] - df['atr'] * 1.5
        df.loc[long_mask, 'take_profit_1'] = df['close'] + df['atr'] * 2.0
        df.loc[long_mask, 'take_profit_2'] = df['close'] + df['atr'] * 3.0
        df.loc[long_mask, 'take_profit_3'] = df['close'] + df['atr'] * 4.0
        
        # Short positions setup
        short_mask = df['signal'] == -1
        df.loc[short_mask, 'stop_loss'] = df['close'] + df['atr'] * 1.5
        df.loc[short_mask, 'take_profit_1'] = df['close'] - df['atr'] * 2.0
        df.loc[short_mask, 'take_profit_2'] = df['close'] - df['atr'] * 3.0
        df.loc[short_mask, 'take_profit_3'] = df['close'] - df['atr'] * 4.0
        
        return df

    def detect_market_regime(self, data):
        df = data.copy()
        # Calculate ADX
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Calculate Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        
        # Classify market regime
        df['regime'] = 'ranging'
        df.loc[df['adx'] > 25, 'regime'] = 'trending'
        
        return df
    
    def check_session_time(self, data):
        df = data.copy()
        df['time'] = pd.to_datetime(df['time'])
        df['hour'] = df['time'].dt.hour
        df['minute'] = df['time'].dt.minute
        df['time_str'] = df['time'].dt.strftime('%H:%M')
        
        morning_mask = (df['time_str'] >= '09:00') & (df['time_str'] <= '10:30')
        afternoon_mask = (df['time_str'] >= '14:00') & (df['time_str'] <= '15:30')
        df['session_active'] = morning_mask | afternoon_mask
        
        return df