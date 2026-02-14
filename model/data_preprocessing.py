import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess():
    """
    Loads Kaggle Breast Cancer dataset from local CSV,
    encodes target (Malignant = 1),
    splits data, exports test_data.csv,
    scales features and saves scaler.
    """

    # LOAD DATASET FROM LOCAL CSV
    df = pd.read_csv("/Users/shrayvijay/Downloads/Bits/ML/ML_assignment/Assignment_2/breast-cancer-app/breast-cancer-app/data/breast-cancer.csv")

    print("✓ Dataset loaded from local CSV")

    # DROP ID COLUMN (if exists)
    if "id" in df.columns:
        df.drop("id", axis=1, inplace=True)


    # ENCODE TARGET COLUMN
    # Malignant = 1 (Positive)
    # Benign    = 0 (Negative)
    df["diagnosis"] = df["diagnosis"].map({
        "M": 1,
        "B": 0
    })


    # SPLIT FEATURES & TARGET
    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    # TRAIN TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("✓ Train-Test split completed")


    # EXPORT TEST DATA CSV
    os.makedirs("data", exist_ok=True)

    test_df = X_test.copy()
    test_df["target"] = y_test.values

    test_df.to_csv("data/test_data.csv", index=False)

    print("✓ test_data.csv exported to /data")


    # FEATURE SCALING
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns
    )

    # SAVE SCALER
    os.makedirs("model", exist_ok=True)

    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("✓ Scaler saved")

    return X_train_scaled, X_test_scaled, y_train, y_test
