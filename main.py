
from src.data_preprocessing import load_data
from src.logger import logger
from src.model_training import train_model, save_model
from src.evaluation import evaluate_model
import config

def main():
    # 1. Load
    df = load_data(config.PATH)
    if df is None: return

    # 2. Train (Note: getting all 3 returns)
    X_test, y_test, model = train_model(df)
    logger.info("Model training complete.")

    # 3. Evaluate
    evaluate_model(model, X_test, y_test)

    # 4. Save
    save_model(model)
    logger.info("Model training and evaluation complete. Model saved to disk.")

if __name__ == '__main__':
    main()