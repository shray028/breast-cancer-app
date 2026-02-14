import os
import pickle
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score
)

from data_preprocessing import load_and_preprocess

from logistic_regression import train_model as train_lr
from decision_tree import train_model as train_dt
from knn import train_model as train_knn
from naive_bayes import train_model as train_nb
from random_forest import train_model as train_rf
from xgboost_model import train_model as train_xgb


# Create trained_models folder
os.makedirs("model/trained_models", exist_ok=True)


# Load data
X_train, X_test, y_train, y_test = load_and_preprocess()


models = {
    "Logistic Regression": ("logistic.pkl", train_lr),
    "Decision Tree": ("decision_tree.pkl", train_dt),
    "KNN": ("knn.pkl", train_knn),
    "Naive Bayes": ("naive_bayes.pkl", train_nb),
    "Random Forest": ("random_forest.pkl", train_rf),
    "XGBoost": ("xgboost.pkl", train_xgb),
}


results = []

print("\nTraining All Models")
print("=" * 50)


for name, (filename, trainer) in models.items():

    print(f"\nTraining {name}...")

    model = trainer(X_train, y_train)

    # Predictions on test data
    y_pred = model.predict(X_test)

    try:
        y_proba = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_proba[:, 1])
    except:
        auc = 0.0

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append([
        name, accuracy, auc, precision, recall, f1, mcc
    ])

    # Save model
    path = f"model/trained_models/{filename}"
    with open(path, "wb") as f:
        pickle.dump(model, f)

    print(f"✓ Saved → {path}")


# Results table
results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC",
    ],
)

print("\nModel Comparison")
print(results_df.to_string(index=False))


# Save CSV
results_df.to_csv(
    "/Users/shrayvijay/Downloads/Bits/ML/ML_assignment/Assignment_2/data.csv",
    index=False
)

print("\nAll models trained & saved successfully!")
