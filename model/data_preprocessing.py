import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer


def load_and_preprocess():
    """
    Loads dataset, splits, scales, saves scaler
    and exports test_data.csv with target column.
    """

    # LOAD DATASET
    data = load_breast_cancer()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    print("Dataset loaded successfully!")

    # TRAIN TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Train-Test split completed!")

    # SAVE TEST DATA CSV
    # Create data folder if not exists
    os.makedirs("data", exist_ok=True)

    # Combine test features + target
    test_df = X_test.copy()
    test_df["target"] = y_test.values

    # Export CSV
    test_df.to_csv("data/test_data.csv", index=False)

    print("✓ test_data.csv exported inside /data folder")

    # SCALING
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("✓ Scaler saved!")

    return X_train_scaled, X_test_scaled, y_train, y_test
