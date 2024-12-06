# train_model.py
import pandas as pd
import joblib
from src.analysis.technical.enhanced_strategy_v2 import EnhancedTechnicalStrategyV2
from src.data.loaders.market_data import MarketDataLoader
from loguru import logger

def main():
    # Defina os parâmetros do backtest
    db_path = 'src/data/database/candles.db'  # Atualize o caminho se necessário
    table_name = 'candles'
    start_date = '2024-06-08'
    end_date = '2024-07-05'
    interval = 1  # Intervalo em minutos
    model_path = 'model.pkl'  # Caminho para salvar o modelo

    # Inicializar o DataLoader
    data_loader = MarketDataLoader(db_path, table_name=table_name)

    # Carregar os dados
    data = data_loader.get_minute_data(interval=interval, start_date=start_date, end_date=end_date)

    if data.empty:
        logger.error("Nenhum dado carregado para o período especificado.")
        return

    logger.info(f"Dados carregados para treinamento: {len(data)} registros.")

    # Inicializar a estratégia
    strategy = EnhancedTechnicalStrategyV2(model_path=model_path)

    # Treinar o modelo
    try:
        strategy.train_model(data, future_periods=10)
        logger.info(f"Modelo treinado e salvo em '{model_path}'.")
    except Exception as e:
        logger.error(f"Erro durante o treinamento do modelo: {str(e)}")

if __name__ == "__main__":
    main()
