# 🎯 Customer Churn Prediction & Interactive Retention Dashboard

## 📌 Project Overview
This repository contains an end-to-end Machine Learning pipeline and an interactive Streamlit web dashboard designed to predict customer churn and recommend automated, personalized retention actions for subscription-based businesses.

Developed as part of the **7005SCN Individual Research Project** (**MSc Data Science, Coventry University**), this project addresses class imbalance, predictive accuracy, feature interpretability, and business ROI to translate machine learning predictions into actionable retention strategies for SMEs.

---

## 🎯 Key Features & Workflow

- 🧹 **Automated Preprocessing & Feature Engineering:** Cleans recency anomalies (capping `-999` login values), imputes missing fields with mode/median indicators, engineers customer tenure in months, maps feedback sentiment (-2 to +2), and synthesizes active complaint flags.
- ⚖️ **Imbalance & Cross-Validation Benchmarking:** Evaluates Logistic Regression, Random Forest, and XGBoost using Stratified 10-Fold Cross-Validation across four imbalance mitigation strategies (Baseline, SMOTE Oversampling, Random Under-Sampling, and Cost-Sensitive Weighting).
- 🖥️ **Interactive Streamlit Dashboard (`app.py`):** 
  - **Batch & Single Churn Risk Engine:** Real-time customer churn risk scoring and automated risk-tiered retention recommendations.
  - **Financial ROI Simulator:** Interactive calculator for retention campaign budgets, CLV recovery, and net retention profit.
  - **Explainable AI (SHAP):** Global feature importance and individual customer prediction explanations.
  - **Benchmark Visualizer:** Interactive ROC curves, Precision-Recall curves, and confusion matrix displays.

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

## 🏆 Key Findings & Results Summary

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


