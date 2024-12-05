from typing import Dict, Optional
import pandas as pd
import numpy as np
from loguru import logger
import talib

class TechnicalStrategy:
    """Technical analysis strategy for Mini Dollar trading."""

    def __init__(self,
                 # Parâmetros RSI
                 rsi_period: int = 14,
                 rsi_upper: float = 70,
                 rsi_lower: float = 30,
                 
                 # Parâmetros Médias Móveis
                 ma_fast: int = 9,
                 ma_slow: int = 21,
                 ma_type: str = 'EMA',  # 'EMA' ou 'SMA'
                 
                 # Parâmetros ADX
                 adx_period: int = 14,
                 adx_threshold: int = 25,
                 
                 # Parâmetros ATR e Stops
                 atr_period: int = 14,
                 atr_multiplier: float = 2.0,
                 take_profit_multiplier: float = 3.0,
                 trailing_stop_multiplier: float = 1.5,
                 use_trailing_stop: bool = True,
                 use_break_even: bool = True,
                 break_even_threshold: float = 1.0,
                 
                 # Parâmetros Volume
                 volume_ma_period: int = 20,
                 volume_threshold: float = 1.2,
                 
                 # Parâmetros MACD
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 
                 # Parâmetros Bandas de Bollinger
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 
                 # Parâmetros de Score
                 min_score_entry: float = 4.0,
                 weight_rsi: float = 1.0,
                 weight_ma: float = 1.0,
                 weight_adx: float = 1.0,
                 weight_volume: float = 0.5,
                 weight_macd: float = 1.0,
                 weight_bb: float = 0.5):
        
        # RSI
        self.rsi_period = rsi_period
        self.rsi_upper = rsi_upper
        self.rsi_lower = rsi_lower
        
        # Médias Móveis
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.ma_type = ma_type
        
        # ADX
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        
        # ATR e Stops
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.take_profit_multiplier = take_profit_multiplier
        self.trailing_stop_multiplier = trailing_stop_multiplier
        self.use_trailing_stop = use_trailing_stop
        self.use_break_even = use_break_even
        self.break_even_threshold = break_even_threshold
        
        # Volume
        self.volume_ma_period = volume_ma_period
        self.volume_threshold = volume_threshold
        
        # MACD
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        
        # Bollinger
        self.bb_period = bb_period
        self.bb_std = bb_std
        
        # Scores
        self.min_score_entry = min_score_entry
        self.weights = {
            'rsi': weight_rsi,
            'ma': weight_ma,
            'adx': weight_adx,
            'volume': weight_volume,
            'macd': weight_macd,
            'bb': weight_bb
        }
        
        logger.info(f"Initialized Enhanced TechnicalStrategy with {ma_type} moving averages")

    def calculate_ma(self, data: pd.Series, period: int) -> pd.Series:
        """Calcula média móvel conforme tipo especificado"""
        if self.ma_type == 'EMA':
            return talib.EMA(data, timeperiod=period)
        return talib.SMA(data, timeperiod=period)

    def calculate_atr_based_stops(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate dynamic stops based on ATR"""
        df = data.copy()
        
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.atr_period)
        
        # Stops dinâmicos
        df['dynamic_stop_loss'] = df['atr'] * self.atr_multiplier
        df['dynamic_take_profit'] = df['atr'] * self.take_profit_multiplier
        
        # Trailing stop e níveis de take profit
        df['trailing_stop'] = df['close'] - (df['atr'] * self.trailing_stop_multiplier)
        df['take_profit_level1'] = df['close'] + (df['atr'] * self.take_profit_multiplier)
        df['take_profit_level2'] = df['close'] + (df['atr'] * self.take_profit_multiplier * 1.5)
        df['take_profit_level3'] = df['close'] + (df['atr'] * self.take_profit_multiplier * 2)
        
        # Break even
        if self.use_break_even:
            df['break_even_level'] = df['close'] + (df['atr'] * self.break_even_threshold)
        
        return df

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.rsi_period)
        
        # Médias Móveis
        df['ma_fast'] = self.calculate_ma(df['close'], self.ma_fast)
        df['ma_slow'] = self.calculate_ma(df['close'], self.ma_slow)
        
        # ADX e DI
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=self.adx_period)
        df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=self.adx_period)
        df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=self.adx_period)
        
        # Volume
        df['volume_ma'] = talib.SMA(df['volume'], timeperiod=self.volume_ma_period)
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['obv'] = talib.OBV(df['close'], df['volume'])
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'], 
            fastperiod=self.macd_fast, 
            slowperiod=self.macd_slow, 
            signalperiod=self.macd_signal
        )
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'], 
            timeperiod=self.bb_period, 
            nbdevup=self.bb_std, 
            nbdevdn=self.bb_std
        )
        
        # Stops dinâmicos
        df = self.calculate_atr_based_stops(df)
        
        return df.ffill().bfill()

    def calculate_signal_score(self, row) -> float:
        """Calcula score para sinais de compra"""
        score = 0
        
        # RSI
        if row['rsi'] < self.rsi_lower:
            score += self.weights['rsi']
            
        # DI/ADX
        if row['plus_di'] > row['minus_di'] and row['adx'] > self.adx_threshold:
            score += self.weights['adx']
            
        # Volume
        if row['volume_ratio'] > self.volume_threshold:
            score += self.weights['volume']
            
        # MACD
        if row['macd'] > row['macd_signal']:
            score += self.weights['macd']
            
        # Médias Móveis
        if row['ma_fast'] > row['ma_slow']:
            score += self.weights['ma']
            
        # Bollinger Bands
        if row['close'] < row['bb_lower']:
            score += self.weights['bb']
            
        return score

    def calculate_signal_score_sell(self, row) -> float:
        """Calcula score para sinais de venda"""
        score = 0
        
        # RSI
        if row['rsi'] > self.rsi_upper:
            score += self.weights['rsi']
            
        # DI/ADX
        if row['minus_di'] > row['plus_di'] and row['adx'] > self.adx_threshold:
            score += self.weights['adx']
            
        # Volume
        if row['volume_ratio'] > self.volume_threshold:
            score += self.weights['volume']
            
        # MACD
        if row['macd'] < row['macd_signal']:
            score += self.weights['macd']
            
        # Médias Móveis
        if row['ma_fast'] < row['ma_slow']:
            score += self.weights['ma']
            
        # Bollinger Bands
        if row['close'] > row['bb_upper']:
            score += self.weights['bb']
            
        return score

    def generate_signals(self, data: pd.DataFrame, 
                        multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """Generate trading signals"""
        df = data.copy()
        
        # Inicialização
        df['signal'] = 0
        df['buy_score'] = df.apply(self.calculate_signal_score, axis=1)
        df['sell_score'] = df.apply(self.calculate_signal_score_sell, axis=1)
        
        # Gerar sinais baseados nos scores
        df['signal'] = np.where(df['buy_score'] >= self.min_score_entry, 1,
                              np.where(df['sell_score'] >= self.min_score_entry, -1, 0))
        
        # Confirmação Multi-Timeframe
        if multi_tf_data:
            higher_tf = list(multi_tf_data.values())[0]
            higher_tf_trend = np.where(higher_tf['ma_fast'] > higher_tf['ma_slow'], 1, -1)
            df['signal'] = df['signal'] * (df['signal'] == higher_tf_trend).astype(int)
        
        # Remover sinais consecutivos duplicados
        df['final_signal'] = df['signal'].diff().ffill()
        df['final_signal'] = df['final_signal'].replace(0, np.nan).ffill().fillna(0)
        df['final_signal'] = df['final_signal'].apply(lambda x: x if x in [1, -1] else 0)
        
        # Definir stops
        df['stop_loss'] = df['dynamic_stop_loss']
        df['take_profit'] = df['dynamic_take_profit']
        
        if self.use_trailing_stop:
            df['trailing_stop'] = df['trailing_stop']
        
        logger.info("Generated trading signals with complete strategy")
        return df