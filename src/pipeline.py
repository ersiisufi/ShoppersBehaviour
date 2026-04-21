from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import config

def create_pipeline():
    preprocessor = ColumnTransformer(
    transformers = [
            ('num', StandardScaler(), config.NUM_FEATURES),
            ('cat',OneHotEncoder(handle_unknown = 'ignore'), config.CAT_FEATURES),
            ('bin', 'passthrough', config.BIN_FEATURES)
        ]
    )
    model = Pipeline(
        [
            ("Preprocessing", preprocessor),
            ("Clasiffier", RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42))
        ]
    )