# 🧠 Breast Cancer Classification — ML Deployment Project

---

## 📌 Problem Statement

The objective of this project is to build and evaluate multiple Machine Learning classification models to predict whether a breast tumor is **Malignant** or **Benign** based on diagnostic measurements.

The project also involves developing an interactive **Streamlit web application** where users can upload test data, select trained models, and visualize predictions along with evaluation metrics.

This demonstrates a complete end-to-end ML workflow including model training, evaluation, comparison, and deployment.

---

## 📊 Dataset Description

* **Dataset Name:** Breast Cancer Wisconsin Diagnostic Dataset
* **Source:** Kaggle / UCI Machine Learning Repository
* **Total Instances:** 569
* **Total Features:** 30 numerical features
* **Target Variable:** Diagnosis

### Target Classes

| Value | Class     |
| ----- | --------- |
| 0     | Malignant |
| 1     | Benign    |

### Feature Information

Features are computed from digitized images of Fine Needle Aspirate (FNA) of breast masses and describe characteristics of cell nuclei such as:

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Concavity
* Symmetry
* Fractal Dimension

The dataset contains **no missing values** and all features are numeric.

---

## 🤖 Machine Learning Models Implemented

The following six classification algorithms were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

All models were trained using an **80-20 stratified train-test split**.

---

## 📈 Evaluation Metrics Used

Each model was evaluated using the following metrics:

* Accuracy
* AUC Score
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

All evaluation metrics were calculated on the **test dataset** to measure generalization performance.

---

## 📋 Model Comparison Table

| ML Model Name            | Accuracy | AUC      | Precision | Recall   | F1 Score | MCC      |
| ------------------------ | -------- | -------- | --------- | -------- | -------- | -------- |
| Logistic Regression      | 0.982456 | 0.995370 | 0.982456  | 0.982456 | 0.982456 | 0.962302 |
| Decision Tree            | 0.912281 | 0.915675 | 0.916072  | 0.912281 | 0.913021 | 0.817412 |
| KNN                      | 0.956140 | 0.978836 | 0.956073  | 0.956140 | 0.956027 | 0.905447 |
| Naive Bayes              | 0.929825 | 0.986772 | 0.929825  | 0.929825 | 0.929825 | 0.849206 |
| Random Forest (Ensemble) | 0.947368 | 0.993717 | 0.947368  | 0.947368 | 0.947368 | 0.886905 |
| XGBoost (Ensemble)       | 0.947368 | 0.993717 | 0.947440  | 0.947368 | 0.947087 | 0.886414 |

---

## 🔍 Model Performance Observations

| ML Model Name            | Observation about Model Performance                                                                |
| ------------------------ | -------------------------------------------------------------------------------------------------- |
| Logistic Regression      | Achieved the highest accuracy and MCC score, indicating strong linear separability in the dataset. |
| Decision Tree            | Lowest performance due to overfitting and high variance typical of single tree models.             |
| KNN                      | Performed well due to effective distance-based classification after feature scaling.               |
| Naive Bayes              | High AUC but slightly lower accuracy due to independence assumptions between correlated features.  |
| Random Forest (Ensemble) | Strong ensemble performance with high AUC, though marginally below Logistic Regression.            |
| XGBoost (Ensemble)       | Comparable to Random Forest; boosting gains limited due to already well-engineered features.       |

---

## 🧪 Test Dataset for Deployment

A separate **test_data.csv** file was generated during preprocessing containing:

* All test samples
* All feature columns
* Target column for validation

This file is used in the Streamlit app for prediction and performance evaluation.

---

## 🌐 Streamlit Web Application Features

The deployed application includes:

* CSV dataset upload (Test data only)
* Model selection dropdown
* Prediction output display
* Evaluation metrics display
* Confusion matrix / classification report visualization

---

## 📁 Project Repository Structure

```
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│
├── data/
│   └── test_data.csv
│
└── model/
    │-- train.py
    │-- data_preprocessing.py
    │-- trained_models/
    │-- scaler.pkl
```

---

## 🚀 Deployment

The Streamlit application is deployed on **Streamlit Community Cloud** and provides an interactive interface for real-time predictions using trained ML models.

---

## ✅ Conclusion

All six Machine Learning models successfully classified breast cancer tumors with high accuracy. Logistic Regression emerged as the best performer due to the dataset’s linear separability and low noise characteristics. Ensemble models such as Random Forest and XGBoost also demonstrated strong predictive capability.

This project showcases a complete end-to-end Machine Learning pipeline including data preprocessing, model training, evaluation, comparison, and deployment.

---
