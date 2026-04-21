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
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_probs):.4f} \n")

    y_pred_new = (y_probs >= 0.3).astype(int)

    print("--- Revised Report (Threshold = 0.3) ---")
    print(classification_report(y_test, y_pred_new))

