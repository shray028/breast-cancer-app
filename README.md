# Breast Cancer Classification — Machine Learning Deployment Project

---

## 📌 Problem Statement

The objective of this project is to develop and evaluate multiple Machine Learning classification models to predict whether a breast tumor is **Malignant** (cancerous) or **Benign** (non-cancerous) based on diagnostic measurements.

In addition to model development, an interactive **Streamlit web application** was built to demonstrate real-time predictions, model comparison, and evaluation on unseen test data.

This project showcases a complete end-to-end Machine Learning workflow including preprocessing, training, evaluation, and deployment.

---

## 📊 Dataset Description

* **Dataset Name:** Breast Cancer Dataset
* **Source:** Kaggle — Breast Cancer Dataset by Yasser H.
* **Instances:** 569
* **Features:** 30 numerical features
* **Problem Type:** Binary Classification

The dataset contains computed features extracted from digitized images of fine needle aspirate (FNA) of breast masses. These features describe characteristics of cell nuclei such as:

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Concavity
* Symmetry
* Fractal Dimension

No missing values are present and all features are numeric.

---

## 🎯 Target Variable

Column: `diagnosis`

| Encoded Value | Class Label | Meaning                        |
| ------------- | ----------- | ------------------------------ |
| 1             | Malignant   | Cancerous (Positive Class)     |
| 0             | Benign      | Non-cancerous (Negative Class) |

Malignant tumors were treated as the **positive class** for medical relevance and correct interpretation of evaluation metrics such as Recall (Sensitivity).

---

## 🤖 Machine Learning Models Implemented

The following six classification models were trained on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

A stratified **80-20 train-test split** was used to ensure balanced class distribution.

---

## 📈 Evaluation Metrics

Each model was evaluated using the following metrics:

* Accuracy
* AUC Score
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

All metrics were computed on the **test dataset** to measure real generalization performance.

---

## 📋 Model Comparison Table

| ML Model Name            | Accuracy | AUC      | Precision | Recall   | F1 Score | MCC      |
| ------------------------ | -------- | -------- | --------- | -------- | -------- | -------- |
| Logistic Regression      | 0.964912 | 0.996032 | 0.975000  | 0.928571 | 0.951220 | 0.924518 |
| Decision Tree            | 0.929825 | 0.924603 | 0.904762  | 0.904762 | 0.904762 | 0.849206 |
| KNN                      | 0.956140 | 0.982308 | 0.974359  | 0.904762 | 0.938272 | 0.905824 |
| Naive Bayes              | 0.921053 | 0.989087 | 0.923077  | 0.857143 | 0.888889 | 0.829162 |
| Random Forest (Ensemble) | 0.973684 | 0.994709 | 1.000000  | 0.928571 | 0.962963 | 0.944155 |
| XGBoost (Ensemble)       | 0.964912 | 0.994709 | 1.000000  | 0.904762 | 0.950000 | 0.925820 |

---

## 🔍 Model Performance Observations

| ML Model Name                 | Observation about Model Performance                                                                                                                                                                                                                                                                                                                                         |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression**       | Logistic Regression achieved very high accuracy (96.49%) and AUC (0.996), indicating that the dataset is highly linearly separable. The strong MCC score further confirms balanced performance across both classes. Slightly lower recall suggests that a few malignant cases were misclassified, which is common in linear models when class boundaries are close.         |
| **Decision Tree**             | Decision Tree recorded the lowest performance among all models, with reduced accuracy and MCC. This is primarily due to high variance and overfitting tendencies of single trees. Without depth restriction, the model memorizes training patterns but fails to generalize well on unseen test data.                                                                        |
| **K-Nearest Neighbors (KNN)** | KNN demonstrated strong classification capability with accuracy above 95%. Feature scaling significantly improved distance-based separation. However, recall was slightly lower, indicating that some malignant cases were missed. The chosen k=5 provided a good balance between bias and variance.                                                                        |
| **Naive Bayes**               | Naive Bayes produced a high AUC (0.989), showing good probability ranking ability. However, lower recall and F1 score indicate limitations due to the independence assumption between features. Since tumor measurements are highly correlated, this assumption reduces classification effectiveness.                                                                       |
| **Random Forest (Ensemble)**  | Random Forest emerged as the best overall performer with the highest accuracy (97.37%) and MCC (0.944). The ensemble approach reduced variance and improved generalization. Hyperparameters such as 300 estimators and depth restriction (max_depth=6) helped balance overfitting and underfitting while maintaining strong predictive power.                               |
| **XGBoost (Ensemble)**        | XGBoost achieved performance comparable to Random Forest with perfect precision (1.00), meaning all predicted malignant cases were correct. However, slightly lower recall indicates that some malignant samples were not detected. Gradient boosting effectively captured complex feature interactions, aided by tuned parameters like learning_rate=0.05 and max_depth=3. |

---

## 📊 Comparative Insights

* Ensemble models (Random Forest, XGBoost) outperformed individual learners due to variance reduction and better feature interaction learning.
* Logistic Regression performed exceptionally well, confirming strong linear separability in the dataset.
* Distance-based KNN benefited from feature scaling but showed minor sensitivity to malignant recall.
* Naive Bayes was limited by correlated medical features violating independence assumptions.
* Decision Tree showed the weakest generalization due to overfitting behavior.

---

## 🧪 Test Dataset for Deployment

A separate `test_data.csv` file was generated during preprocessing containing:

* All test samples
* Feature columns
* Target column for validation

This dataset is used in the Streamlit application for prediction and evaluation.

---

## 🌐 Streamlit Application Features

The deployed Streamlit web application includes:

* CSV dataset upload (Test data only)
* Model selection dropdown
* Prediction output display
* Evaluation metrics display
* Confusion matrix / classification report visualization

---

## 📁 Repository Structure

```
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│
├── data/
│   ├── data.csv
│   └── test_data.csv
│
└── model/
    ├── train.py
    ├── data_preprocessing.py
    ├── scaler.pkl
    └── trained_models/
```

---

## 🚀 Deployment

The application is deployed on **Streamlit Community Cloud** and provides an interactive interface for real-time breast cancer prediction using trained ML models.

---

## ✅ Conclusion

All six Machine Learning models successfully classified breast cancer tumors with high predictive performance. Random Forest emerged as the best performing model, achieving the highest accuracy and MCC score. Logistic Regression and XGBoost also demonstrated strong diagnostic capability.

This project demonstrates a complete end-to-end Machine Learning pipeline including preprocessing, model training, evaluation, comparison, and deployment via Streamlit.

---
