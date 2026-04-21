from sklearn.model_selection import train_test_split
from src.pipeline import create_pipeline
import pandas as pd
import config
from joblib import dump


def train_model(df: pd.DataFrame ):
    X = df.drop(config.TARGET, axis=1)
    y = df[config.TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X,y, config.TEST_SIZE, config.RANDOM_STATE)

    model = create_pipeline()

    model.fit(X_train, y_train)

    return X_test, y_test, model


def save_model (model, path = "models/model.joblib"):
    dump(model, path)