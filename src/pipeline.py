from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from src.feature_engineering import engineer_behavioral_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import config

def create_pipeline():
    engineering_step = FunctionTransformer(engineer_behavioral_features)
    preprocessor = ColumnTransformer(
    transformers = [
            ('num', StandardScaler(), config.NUM_FEATURES),
            ('cat',OneHotEncoder(handle_unknown = 'ignore'), config.CAT_FEATURES),
            ('bin', 'passthrough', config.BIN_FEATURES)
        ]
    )
    fullPipeline =  Pipeline(
        [
            ("Engineering", engineering_step),
            ("Preprocessing", preprocessor),
            ("Clasiffier", RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42)),
        ]
    )
    return fullPipeline