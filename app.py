import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Breast Cancer Classification",
    layout="wide"
)

# ---------------- THEME (Light Blue + White) ----------------
st.markdown("""
<style>
body {
    background-color: #f5fbff;
}
[data-testid="stSidebar"] {
    background-color: #e6f2ff;
}
h1, h2, h3 {
    color: #1f77b4;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🩺 Breast Cancer Classification Dashboard")
st.write("Upload **test dataset only**, select a model, and view evaluation metrics.")

# ---------------- SIDEBAR ----------------
st.sidebar.header("🔧 Configuration")

model_choice = st.sidebar.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

# ---------------- MODEL LOADER ----------------
MODEL_PATHS = {
    "Logistic Regression": "model/saved_models/logistic.pkl",
    "Decision Tree": "model/saved_models/dt.pkl",
    "KNN": "model/saved_models/knn.pkl",
    "Naive Bayes": "model/saved_models/nb.pkl",
    "Random Forest": "model/saved_models/rf.pkl",
    "XGBoost": "model/saved_models/xgb.pkl"
}

FEATURE_PATH = "model/saved_models/features.pkl"

# ---------------- MAIN LOGIC ----------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if "target" not in data.columns:
        st.error("CSV must contain a 'target' column")
        st.stop()

    X_test = data.drop(columns=["target"])
    y_test = data["target"]

    features = joblib.load(FEATURE_PATH)
    X_test = X_test[features]

    model = joblib.load(MODEL_PATHS[model_choice])

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    # ---------------- METRICS ----------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")

    col4, col5, col6 = st.columns(3)

    col4.metric("F1 Score", f"{f1_score(y_test, y_pred):.3f}")
    col5.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.3f}")

    if y_prob is not None:
        col6.metric("AUC", f"{roc_auc_score(y_test, y_prob):.3f}")
    else:
        col6.metric("AUC", "N/A")

    # ---------------- CONFUSION MATRIX ----------------
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ---------------- CLASSIFICATION REPORT ----------------
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

else:
    st.info("⬅ Upload a CSV file from the sidebar to begin")
