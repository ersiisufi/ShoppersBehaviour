
from src.data_preprocessing import load_data
from src.feature_engineering import engineer_behavioral_features
from src.model_training import train_model, save_model
from src.evaluation import evaluate_model
import config

def main():
    # 1. Load
    df = load_data(config.PATH)
    if df is None: return

    # 2. Engineer
    df = engineer_behavioral_features(df)

    # 3. Train (Note: getting all 3 returns)
    X_test, y_test, model = train_model(df)

    # 4. Evaluate
    evaluate_model(model, X_test, y_test)

    # 5. Save
    save_model(model)
    print("Project workflow complete!")

if __name__ == '__main__':
    main()