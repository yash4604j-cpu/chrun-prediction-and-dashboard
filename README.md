# 🎯 Customer Churn Prediction & Interactive Retention Dashboard

[![Live Dashboard](https://img.shields.io/badge/Streamlit-Live_Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://chrun-prediction-and-dashboard.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-228B22?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Coventry University](https://img.shields.io/badge/Coventry_University-7005SCN-00529B?style=for-the-badge)](https://www.coventry.ac.uk/)
[![Ethics ID](https://img.shields.io/badge/Ethics_ID-P194842-brightgreen?style=for-the-badge)](https://www.coventry.ac.uk/)

> **M.Sc. Data Science Individual Research Project (7005SCN)**  
> **Author:** Yash Anand Kumar  
> **Supervisor:** Dr. Mohammed Ahmed (CEES)  
> **Ethics Project ID:** P194842 (Authorised — Low Risk)

---

## 🌐 Live Web Application
🔗 **Access the Live Dashboard:** [https://chrun-prediction-and-dashboard.streamlit.app](https://chrun-prediction-and-dashboard.streamlit.app)

---

## 📌 Project Overview
This repository contains an end-to-end Machine Learning pipeline and an interactive Streamlit web dashboard designed to predict customer churn and recommend automated, personalized retention actions for subscription-based businesses.

Developed as part of the **7005SCN Individual Research Project** (**MSc Data Science, Coventry University**), this project addresses class imbalance, predictive accuracy, feature interpretability, and business ROI to translate machine learning predictions into actionable retention strategies for SMEs.

---

## 🎯 Key Features & Dashboard Capabilities

- 🧹 **Automated Preprocessing & Feature Engineering:** Cleans recency anomalies (capping `-999` login values using IQR bounds), imputes missing fields with mode/median indicators, engineers customer tenure in months, maps feedback sentiment (-2 to +2), and synthesizes active complaint flags.
- 🖥️ **Interactive 4-Module Streamlit Dashboard (`app.py`):**
  - **Module 1 — Churn Risk Engine:** Real-time single & batch customer churn probability scoring with automated risk-tiered retention recommendations (High, Medium, Low).
  - **Module 2 — Financial ROI Simulator:** Interactive calculator for retention campaign budgets, CLV recovery, and net retention profit.
  - **Module 3 — Explainable AI (SHAP):** Global feature importance and individual customer prediction explanations using SHAP force & summary plots.
  - **Module 4 — Empirical Benchmark Visualizer:** Interactive ROC curves, Precision-Recall curves, confusion matrices, and data schema stats.
- ⚖️ **Imbalance & Cross-Validation Benchmarking:** Evaluates Logistic Regression, Random Forest, and XGBoost using Stratified 10-Fold Cross-Validation across four imbalance mitigation strategies (Baseline, SMOTE Oversampling, Random Under-Sampling, and Cost-Sensitive Weighting).

---

## 📊 Empirical Model Performance Summary

| Model | Imbalance Strategy | 10-Fold CV AUC (Mean ± Std) | Holdout AUC-ROC | Holdout Precision | Holdout Recall | Holdout F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Champion)** | **Baseline (No Sampling)** | **0.9779 ± 0.0046** | **0.9753** | **0.9510** | **0.9311** | **0.9409** |
| XGBoost | SMOTE Oversampling | 0.9783 ± 0.0044 | 0.9753 | 0.9516 | 0.9286 | 0.9399 |
| XGBoost | Cost-Sensitive Weighting | 0.9778 ± 0.0047 | 0.9753 | 0.9544 | 0.9258 | 0.9399 |
| Random Forest | Cost-Sensitive Weighting | 0.9785 ± 0.0044 | 0.9758 | 0.9483 | 0.9248 | 0.9364 |
| Random Forest | Baseline (No Sampling) | 0.9774 ± 0.0053 | 0.9753 | 0.9439 | 0.9291 | 0.9364 |
| Logistic Regression | Baseline (No Sampling) | 0.9443 ± 0.0055 | 0.9440 | 0.8592 | 0.8688 | 0.8640 |

---

## 📁 Repository Structure

```
.
├── 📄 app.py                     # Main Streamlit Web Application
├── 📊 dataset.csv                # E-commerce Customer Dataset (36,992 rows)
├── 📦 churn_pipeline.joblib      # Serialized Machine Learning Pipeline
├── 📋 requirements.txt           # Python Dependencies
├── 📂 src/
│   ├── 🛠️ preprocess.py          # Data Cleaning & Feature Engineering Pipeline
│   └── 🤖 train.py               # Model Training, 10-Fold CV & SHAP Generation
├── 📂 models/
│   └── 📦 churn_pipeline.joblib  # Trained Champion Model Pipeline
├── 📂 reports/                   # Performance Metrics & Benchmark Plots
│   ├── 📈 cv_10fold_performance.csv
│   ├── 📊 roc_curves_comparison.png
│   ├── 📉 pr_curves_comparison.png
│   └── 🐝 shap_summary_dot.png
└── 📖 README.md                  # Project Documentation
```

---

## 📊 Dataset Overview

- **Source:** Subscription E-Commerce Customer Dataset
- **Scope:** 36,992 customer records with 23 behavioural and demographic attributes.
- **Key Attributes:** `days_since_last_login`, `avg_transaction_value`, `points_in_wallet`, `membership_category`, `feedback`, `complaint_status`, `joining_date`.
- **Target Variable:** `churn_risk_score` (0 = Retained, 1 = Churned with a ~46% / 54% distribution).

---

## 🚀 Quick Start Guide

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yash4604j-cpu/chrun-prediction-and-dashboard.git
cd chrun-prediction-and-dashboard
```

### 2️⃣ Install Required Packages
```bash
pip install -r requirements.txt
```

### 3️⃣ Run Model Training & Benchmark Pipeline
```bash
python src/train.py
```

### 4️⃣ Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
*The dashboard will automatically open in your default browser at [http://localhost:8501](http://localhost:8501).*

---

## 🏆 Key Findings & Business Impact

- 💡 **Predictive Performance:** The champion XGBoost Classifier achieved a benchmark Holdout AUC-ROC of 0.975 and 95.1% Precision, demonstrating strong generalization.
- 🔑 **Top Attrition Drivers:** SHAP feature importance analysis identified points in wallet, login recency, membership tier, and unresolved customer complaints as the primary indicators of churn risk.
- 💰 **Business Impact:** The automated retention engine converts model probabilities into tailored risk-tiered interventions (e.g., targeted vouchers for high-risk customers, customer service outreach for open complaints), optimizing retention campaign ROI.

---

## 📄 License & Academic Info

- **Author:** Yash Anand Kumar
- **Module:** 7005SCN — Individual Research Project (MSc Data Science)
- **Institution:** Coventry University (College of Engineering, Environment and Science)
- **Ethics Project ID:** P194842 (Authorised — Low Risk)
- **License:** Distributed under the MIT License.
- **Repository Link:** [yash4604j-cpu/chrun-prediction-and-dashboard](https://github.com/yash4604j-cpu/chrun-prediction-and-dashboard)
- **Live App URL:** [chrun-prediction-and-dashboard.streamlit.app](https://chrun-prediction-and-dashboard.streamlit.app)



