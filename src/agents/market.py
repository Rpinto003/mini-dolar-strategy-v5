import pandas as pd
import numpy as np
from loguru import logger

class MarketAgent:
    def __init__(self, 
                 initial_balance=100000,
                 max_position=1,
                 stop_loss=100,
                 take_profit=200,
                 atr_multiplier=2.0,
                 slippage=0.01,  # 1% slippage
                 fee=5.0):        # $5 por trade
        
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_position = max_position
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.atr_multiplier = atr_multiplier
        self.slippage = slippage
        self.fee = fee
        self.current_balance = initial_balance
        self.positions = []  # Lista para armazenar múltiplas posições
        
    def execute_trades(self, data: pd.DataFrame) -> pd.DataFrame:
        # Assegurar que a coluna 'position' está como float para evitar FutureWarning
        if 'position' not in data.columns:
            data['position'] = 0.0
        else:
            data['position'] = data['position'].astype(float)
        
        for i, row in data.iterrows():
            current_bar = row
            # Lógica para abrir novas posições
            if current_bar['signal'] == 1 and len(self.positions) < self.max_position:
                # Abrir posição longa
                pos = {
                    'type': 'long',
                    'entry_price': current_bar['close'],
                    'stop_loss': current_bar['close'] - self.stop_loss,
                    'take_profit_1': current_bar['close'] + self.take_profit,
                    'take_profit_2': current_bar['close'] + 2 * self.take_profit,
                    'take_profit_3': current_bar['close'] + 3 * self.take_profit,
                    'breakeven_level': current_bar['close'] + self.take_profit  # Ajuste conforme necessário
                }
                self.positions.append(pos)
                logger.info(f"Abrindo posição longa em {current_bar.name} com preço de entrada {current_bar['close']}")
                data.at[i, 'position'] += 1.0  # Atualizar posição
                
            elif current_bar['signal'] == -1 and len(self.positions) < self.max_position:
                # Abrir posição curta
                pos = {
                    'type': 'short',
                    'entry_price': current_bar['close'],
                    'stop_loss': current_bar['close'] + self.stop_loss,
                    'take_profit_1': current_bar['close'] - self.take_profit,
                    'take_profit_2': current_bar['close'] - 2 * self.take_profit,
                    'take_profit_3': current_bar['close'] - 3 * self.take_profit,
                    'breakeven_level': current_bar['close'] - self.take_profit  # Ajuste conforme necessário
                }
                self.positions.append(pos)
                logger.info(f"Abrindo posição curta em {current_bar.name} com preço de entrada {current_bar['close']}")
                data.at[i, 'position'] -= 1.0  # Atualizar posição

            # Verificar condições de saída para posições abertas
            for pos in self.positions[:]:  # Iterar sobre uma cópia da lista
                if self.check_exit_conditions(current_bar=current_bar, pos=pos):
                    # Fechar posição
                    if pos['type'] == 'long':
                        profit = current_bar['close'] - pos['entry_price']
                        data.at[i, 'position'] -= 1.0
                    elif pos['type'] == 'short':
                        profit = pos['entry_price'] - current_bar['close']
                        data.at[i, 'position'] += 1.0
                    self.current_balance += profit
                    logger.info(f"Fechando posição {pos['type']} em {current_bar.name} com lucro/prejuízo de {profit}")
                    self.positions.remove(pos)
                    # Atualizar DataFrame com resultados da operação
                    data.at[i, 'trade_executed'] = True
                    data.at[i, 'profit'] = profit
            
            # Atualizar o saldo atual
            data.at[i, 'current_balance'] = self.current_balance
        
        return data

    def validate_long_entry(self, bar):
        return (
            bar['session_active'] and
            bar['volume_ratio'] > 1.0 and
            bar['pressure_ratio'] > 1.2 and
            bar['regime'] == 'trending'
        )
    
    def validate_short_entry(self, bar):
        return (
            bar['session_active'] and
            bar['volume_ratio'] > 1.0 and
            bar['pressure_ratio'] < 0.8 and
            bar['regime'] == 'trending'
        )
    
    def check_exit_conditions(self, current_bar, pos):
        """
        Verifica se as condições de saída para uma posição estão atendidas.

        Args:
            current_bar (pd.Series): Dados do candle atual.
            pos (dict): Informações sobre a posição atual.

        Returns:
            bool: True se as condições de saída forem atendidas, False caso contrário.
        """
        exit_price = self.calculate_exit_price(current_bar, pos)
        logger.info(f"Calculando condições de saída para posição {pos['type']} com preço de saída {exit_price} em {current_bar.name}")

        if pos['type'] == 'long':
            if current_bar['close'] >= exit_price:
                logger.info(f"Take profit ou stop loss atingido para posição longa em {current_bar.name}")
                return True
        elif pos['type'] == 'short':
            if current_bar['close'] <= exit_price:
                logger.info(f"Take profit ou stop loss atingido para posição curta em {current_bar.name}")
                return True

        return False

    def calculate_position_size(self, bar):
        base_size = self.max_position * 0.5
        
        # Adjust size based on volatility
        volatility_factor = min(0.8, 1.0 / (bar['atr'] / bar['close'] * 100))
        
        # Adjust size based on market regime
        regime_factor = 1.0 if bar['regime'] == 'trending' else 0.5
        
        # Adjust size based on signal strength
        signal_factor = min(1.0, abs(bar['ml_prob'] - 0.5) * 2)
        
        # Risk-based position sizing
        risk_factor = min(1.0, self.stop_loss / (bar['atr'] * self.atr_multiplier))

        # Add time-based scaling
        time_factor = 1.0 if bar['session_active'] else 0.5
        
        return base_size * volatility_factor * regime_factor * signal_factor * risk_factor * time_factor
    
    def calculate_exit_price(self, current_bar, pos):
        """
        Calcula o preço de saída para uma posição com base no candle atual e na posição.

        Args:
            current_bar (pd.Series): Dados do candle atual.
            pos (dict): Informações sobre a posição atual.

        Returns:
            float: Preço de saída calculado.
        """
        if pos['type'] == 'long':
            if current_bar['low'] <= pos['stop_loss']:
                return pos['stop_loss']
            elif current_bar['high'] >= pos['take_profit_3']:
                return pos['take_profit_3']
            elif current_bar['high'] >= pos['take_profit_2']:
                return pos['take_profit_2']
            elif current_bar['high'] >= pos['take_profit_1']:
                return pos['take_profit_1']

        elif pos['type'] == 'short':
            if current_bar['high'] >= pos['stop_loss']:
                return pos['stop_loss']
            elif current_bar['low'] <= pos['take_profit_3']:
                return pos['take_profit_3']
            elif current_bar['low'] <= pos['take_profit_2']:
                return pos['take_profit_2']
            elif current_bar['low'] <= pos['take_profit_1']:
                return pos['take_profit_1']

        return current_bar['close']
    
    def calculate_trade_profit(self, exit_price, pos):
        # Aplicar slippage
        if pos['type'] == 'long':
            exit_price -= exit_price * self.slippage
        elif pos['type'] == 'short':
            exit_price += exit_price * self.slippage
        
        profit = pos['size'] * (exit_price - pos['entry_price'])
        
        # Aplicar taxa
        profit -= self.fee
        
        return profit
    
    def calculate_drawdown(self, cumulative_profits):
        peak = cumulative_profits.expanding().max()
        drawdown = (cumulative_profits - peak) / peak * 100
        return drawdown.iloc[-1]
    
    def calculate_max_drawdown(self, data):
        cumulative_returns = data['profit'].cumsum()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max * 100
        return drawdowns.min()
    
    def calculate_sortino_ratio(self, results, risk_free_rate=0.02):
        results = results.copy()
        if 'time' in results.columns:
            results.set_index('time', inplace=True)
        
        daily_returns = results['profit'].groupby(results.index.date).sum() / 100000
        excess_returns = daily_returns - risk_free_rate/252
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = np.sqrt(np.mean(downside_returns**2))
        return np.sqrt(252) * excess_returns.mean() / downside_std if len(downside_returns) > 0 else np.inf

    def calculate_sharpe_ratio(self, results, risk_free_rate=0.02):
        results = results.copy()
        if 'time' in results.columns:
            results.set_index('time', inplace=True)
            
        daily_returns = results['profit'].groupby(results.index.date).sum() / 100000
        excess_returns = daily_returns - risk_free_rate/252
        return np.sqrt(252) * excess_returns.mean() / daily_returns.std()