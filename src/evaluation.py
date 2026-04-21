# src/evaluation.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import config

def evaluate_model(model, X_test, y_test):
    """
    Generates metrics and visualizations for model performance.
    """
    # 1. Predictions
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    # 2. Print Text Reports
    print("\n" + "="*30)
    print("MODEL EVALUATION RESULTS")
    print("="*30)
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_probs):.4f}")

    # 3. Feature Importance Visualization
    # Accessing the classifier inside the pipeline
    classifier = model.named_steps['Clasiffier']
    
    # We need to get the feature names from the preprocessor
    # Note: This part is tricky because OHE creates new columns
    # For now, let's keep it simple or use the preprocessor's get_feature_names_out
    try:
        preprocessor = model.named_steps['Preprocessing']
        ohe_names = preprocessor.transformers_[1][1].get_feature_names_out(config.CAT_FEATURES)
        feature_names = config.NUM_FEATURES + list(ohe_names) + config.BIN_FEATURES
        
        importances = pd.Series(classifier.feature_importances_, index=feature_names)
        importances.sort_values(ascending=False).head(10).plot(kind='barh', figsize=(10,6))
        plt.title("Top 10 Drivers of Purchase Intent")
        plt.gca().invert_yaxis()
        plt.show()
    except Exception as e:
        print(f"Could not generate feature importance plot: {e}")

    # 4. Confusion Matrix Plot
    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()