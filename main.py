import sys
import os
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd

from src.agents.coordinator import StrategyCoordinator
from src.reporting.performance import PerformanceAnalyzer

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
        start_date = '2024-06-08'
        end_date = '2024-12-05'
        interval = 5  # Intervalo em minutos
        
        logger.info(f"Running backtest from {start_date} to {end_date}")

        # Executar backtest
        results = coordinator.backtest(
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )
        
        # Inspecionar os dados de resultado
        print(results[['open', 'high', 'low', 'close', 'volume']].head())
        print(results[['open', 'high', 'low', 'close', 'volume']].describe())

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