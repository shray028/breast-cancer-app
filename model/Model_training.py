"""
Training all 6 models models using sklearn
and saving them as pickle files.

"""

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, classification_report
)


# STEP 1: LOADING DATASET

df = pd.read_csv('/content/data.csv')

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Dataset loaded successfully!")
print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
print(f"Classes: {np.unique(y)}")

# STEP 2: SPLITTING DATA INTO TRAIN AND TEST

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")


# STEP 3: PREPROCESSING DATA (SCALING)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for later use
with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nData preprocessing completed!")


# STEP 4: TRAINING ALL MODELS

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
}

model_filenames = {
    'Logistic Regression': 'logistic.pkl',
    'Decision Tree': 'decision_tree.pkl',
    'K-Nearest Neighbors': 'knn.pkl',
    'Naive Bayes': 'naive_bayes.pkl',
    'Random Forest': 'random_forest.pkl',
    'XGBoost': 'xgboost.pkl'
}

results = []

print("\n" + "="*60)
print("TRAINING MODELS")
print("="*60)

for name, model in models.items():
    print(f"\nTraining {name}...")

    # Train the model
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Get probability predictions
    try:
        y_pred_proba = model.predict_proba(X_test_scaled)
    except:
        y_pred_proba = None

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Calculate AUC
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            auc = 0.0
    else:
        auc = 0.0

    # Save results
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'AUC': auc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'MCC': mcc
    })

    # Save the trained model
    filename = model_filenames[name]
    with open(f'model/{filename}', 'wb') as f:
        pickle.dump(model, f)

    print(f"✓ {name} trained and saved!")
    print(f"  Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")


# STEP 5: DISPLAYING RESULTS TABLE

print("\n" + "="*60)
print("MODEL PERFORMANCE COMPARISON")
print("="*60)

results_df = pd.DataFrame(results)
print("\n", results_df.to_string(index=False))

# Save results to CSV
results_df.to_csv('model_comparison_results.csv', index=False)
print("\n✓ Results saved to 'model_comparison_results.csv'")


# STEP 6: SAVING TEST DATA
# Creating a test dataframe with features and target

test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
test_df['target'] = y_test.values

# Save to CSV
test_df.to_csv('data/test_data.csv', index=False)
print("\n✓ Test data saved to 'data/test_data.csv'")


# STEP 7: PRINTING OBSERVATIONS (for README)

print("\n" + "="*60)
print("OBSERVATIONS FOR README")
print("="*60)

best_model = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
best_accuracy = results_df['Accuracy'].max()

print(f"\nBest Model: {best_model} with accuracy: {best_accuracy:.4f}")

print("\nCopy this table to your README.md:")
print("\n| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |")
print("|--------------|----------|-----|-----------|--------|----|----|")
for _, row in results_df.iterrows():
    print(f"| {row['Model']} | {row['Accuracy']:.4f} | {row['AUC']:.4f} | "
          f"{row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1']:.4f} | {row['MCC']:.4f} |")

print("\n" + "="*60)
print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
print("="*60)