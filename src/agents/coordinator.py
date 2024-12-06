from typing import Optional, Dict
import pandas as pd
from datetime import datetime
from loguru import logger
from src.data.loaders.market_data import MarketDataLoader
from src.analysis.technical.enhanced_strategy_v2 import EnhancedTechnicalStrategyV2
from src.agents.market import MarketAgent

class StrategyCoordinator:
    def __init__(self,
                 initial_balance: float = 100000,
                 max_position: int = 1,
                 stop_loss: float = 100,
                 take_profit: float = 200,
                 db_path: Optional[str] = None,
                 strategy_params: Optional[Dict] = None):
        
        if db_path is None:
            raise ValueError("Database path must be provided.")
        
        self.data_loader = MarketDataLoader(db_path)
        self.strategy_params = strategy_params or {
            'session_times': {
                'morning_start': '09:00',
                'morning_end': '11:00',
                'afternoon_start': '14:00',
                'afternoon_end': '16:00'
            },
            'gap_threshold': 0.2
        }
        
        self.strategy = EnhancedTechnicalStrategyV2(**self.strategy_params)
        self.market = MarketAgent(
            initial_balance=initial_balance,
            max_position=max_position,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr_multiplier=2.0
        )
      
        logger.info("Initialized StrategyCoordinator with parameters:", self.strategy_params)
    
    def process_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                raise ValueError(f"Required columns missing: {missing_columns}")
            
            df = data.copy()
            
            # First generate signals - this adds regime detection
            df = self.strategy.generate_signals(df)
            
            # Then add risk management
            df = self.strategy.add_risk_management(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}")
            raise

    def main():
        """Main execution function"""
        # Configurar logging
        logger.add("logs/strategy_{time}.log", rotation="1 day")
        logger.info("Starting Mini Dollar Strategy")

        try:
            # Inicializar componentes
            coordinator = StrategyCoordinator(
                initial_balance=100000,
                max_position=1,
                stop_loss=100,
                take_profit=200,
                db_path="src/data/database/candles.db"
            )
            analyzer = PerformanceAnalyzer()

            # Definir período de análise
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)  # 6 meses de dados

            logger.info(f"Running backtest from {start_date} to {end_date}")

            # Carregar dados para treinamento
            training_data = coordinator.data_loader.get_minute_data(
                interval=5,
                start_date=(start_date - timedelta(days=30)).strftime('%Y-%m-%d'),  # 30 dias antes para treinamento
                end_date=start_date.strftime('%Y-%m-%d')
            )

            # Treinar o modelo
            coordinator.strategy.train_model(training_data)

            # Executar backtest
            results = coordinator.backtest(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                interval=5
            )

            # Calcular métricas de performance
            metrics = analyzer.calculate_metrics(results)
            
            # Gerar relatório
            report = analyzer.generate_report(results)
            
            # Salvar resultados
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            
            results.to_csv(f"{output_dir}/backtest_results.csv")
            with open(f"{output_dir}/performance_report.txt", "w") as f:
                f.write(report)

            # Exibir métricas principais
            print("\nPerformance Summary:")
            print("-" * 40)
            for metric, value in metrics.items():
                print(f"{metric}: {value:.2f}")

            logger.info("Strategy execution completed successfully")

        except Exception as e:
            logger.error(f"Error executing strategy: {str(e)}")
            raise

    if __name__ == "__main__":
        main()

    def backtest(self,
                start_date: str,
                end_date: str,
                interval: int = 5) -> pd.DataFrame:
        try:
            data = self.data_loader.get_minute_data(
                interval=interval,
                start_date=start_date,
                end_date=end_date
            )
            
            if data.empty:
                raise ValueError("No data loaded for the specified period")
                
            logger.info(f"Loaded {len(data)} candles for backtest")
            
            results = self.process_market_data(data)
            logger.info("Processed market data with technical indicators and ML signals")
            
            results = self.market.execute_trades(results)
            logger.info("Completed trade execution simulation")
            
            logger.info(f"Completed backtest from {start_date} to {end_date}")
            return results
            
        except Exception as e:
            logger.error(f"Error during backtest: {str(e)}")
            raise
    
    def get_performance_metrics(self, results: pd.DataFrame) -> dict:
        metrics = {
            'total_trades': len(results[results['trade_executed']]),
            'win_rate': (results['profit'] > 0).mean(),
            'total_profit': results['profit'].sum(),
            'max_drawdown': self.market.calculate_max_drawdown(results),
            'sharpe_ratio': self.market.calculate_sharpe_ratio(results)
        }
        
        logger.info(f"Calculated performance metrics: {metrics}")
        return metrics