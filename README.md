# TelcoChurn 📉 or Telecom-Customer-Churn-Analysis-and-Prediction

> **End-to-end customer churn prediction pipeline** built with Python — combining exploratory data analysis, feature engineering, and model benchmarking to flag at-risk telecom customers before they leave.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-013243?logo=numpy&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML%20Models-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-006400?logo=xgboost&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-4c72b0)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

---

## Overview

Customer churn — when a customer stops doing business with a company — is one of the most expensive problems in the telecom industry. This project analyzes customer-level data to understand **why** customers churn and builds a predictive model to flag **who** is likely to churn next, so retention efforts can be targeted proactively.

The project is split into two notebooks: one dedicated to exploratory data analysis, and one covering preprocessing, model training, and evaluation.

A **class-weighted Random Forest** was selected as the final model after benchmarking it against Logistic Regression (with and without class weighting, and with threshold tuning) and XGBoost.

| Notebook | Purpose |
|---|---|
| `ChurnAnalysis_DAA.ipynb` | Exploratory Data Analysis (EDA) — understanding churn drivers, distributions, and outliers |
| `Untitled8__1_.ipynb` | Data preprocessing, model building, model comparison, and final model selection |

---

## Dataset

The dataset (`Customer_Data.csv`) contains **6,418 customer records** with **32 features**, covering:

- **Demographics** — Age, Gender, Married, State
- **Account information** — Tenure, Contract type, Payment method, Paperless billing
- **Services subscribed** — Phone, Internet type, Online security/backup, Streaming TV/Movies/Music, Premium support, Unlimited data
- **Financials** — Monthly charge, Total charges, Total refunds, Extra data charges, Long distance charges, Total revenue
- **Target** — `Customer_Status` (Stayed / Churned / Joined)

**Overall churn rate: ~26.99%** — roughly 1 in 4 customers churned.

> `Churn_Category` and `Churn_Reason` (only available for already-churned customers) were **intentionally excluded** from the model. Since this information is only known *after* a customer has already left, including it would leak the outcome into the model and make the predictions unrealistic for real-world use, where the goal is to predict churn *before* it happens.

---

## Exploratory Data Analysis — Key Insights

- **Missing data**: Several service-related columns (Internet Type, Online Security, Streaming services, etc.) had missing values for customers without internet service — these were treated as `"No"` rather than dropped.
- **Contract type** and **Payment method** show a visible relationship with churn — month-to-month customers and certain payment methods churn more often.
- **Tenure**: Churn rate stays fairly stable across tenure groups (~27% in the 0–12, 13–24, and 25–36 month bands), suggesting churn risk doesn't fade quickly with customer age in this dataset.
- **Monthly and Total Charges**: Distributions differ noticeably between churned and retained customers.
- **Outlier check** on `Monthly_Charge` using the IQR method found **zero outliers**, indicating the charge values are well-behaved.
- A **correlation heatmap** was used to check relationships between numeric features before modeling.

---

## Data Preprocessing

1. **Missing value imputation** — service-related NaNs filled with `"No"`
2. **Binary encoding** — Yes/No and Male/Female columns mapped to 1/0
3. **One-hot encoding** — multi-category columns (`State`, `Internet_Type`, `Contract`, `Payment_Method`) encoded with `drop_first=True` to avoid multicollinearity
4. **Target construction** — `Customer_Status` filtered to remove `"Joined"` customers (too new to have meaningfully churned or stayed) and mapped to a binary `Churn` target: `Churned = 1`, `Stayed = 0`
5. **Train/test split** — 80/20 stratified split to preserve the churn ratio in both sets
6. **Feature scaling** — `StandardScaler` fit on training data only, to avoid data leakage into the test set

---

## Modeling Approach

Multiple models were trained and compared before arriving at a final choice:

| Model | ROC-AUC | Churn Recall | Churn Precision |
|---|---|---|---|
| Logistic Regression (default) | 0.861 | 0.63 | 0.68 |
| Logistic Regression (class-weighted) | 0.860 | 0.82 | 0.58 |
| Logistic Regression (threshold = 0.35) | 0.861 | 0.78 | 0.62 |
| **Random Forest (class-weighted)** | **0.889** | **0.76** | **0.69** |
| XGBoost (weighted) | 0.888 | 0.76 | 0.67 |

**Why Random Forest was chosen as the final model:**

- It achieved the **highest ROC-AUC (0.889)** of all models tested, meaning it's the best at ranking customers by churn risk overall.
- It improved **recall for churners from 0.63 → 0.76** compared to the baseline Logistic Regression — catching significantly more at-risk customers — **without sacrificing precision** (0.69, actually higher than the original baseline).
- Logistic Regression variants could push recall higher (up to 0.82), but only by trading away a large chunk of precision, which means more false alarms and wasted retention spend on customers who were never going to churn.
- XGBoost performed almost identically to Random Forest, but Random Forest was selected for its simplicity, faster training, and easier interpretability via feature importances — with no real drop in predictive power.

After experimenting with **class-weighting**, **threshold tuning**, and **tree-based ensembles**, the **class-weighted Random Forest** offered the best overall balance between catching churners and maintaining prediction quality, making it the most practical choice for a real-world retention pipeline.

---

## Results Summary

- **Final Model**: Random Forest Classifier (`class_weight="balanced"`, `n_estimators=300`, `max_depth=8`)
- **ROC-AUC**: 0.889
- **Accuracy**: 0.83
- **Churn Recall**: 0.76
- **Churn Precision**: 0.69

---

## Tech Stack

- **Python**
- **Pandas / NumPy** — data manipulation
- **Matplotlib / Seaborn** — visualization
- **Scikit-learn** — preprocessing, Logistic Regression, Random Forest, evaluation metrics
- **XGBoost** — gradient boosting comparison model
- **Jupyter Notebook** — development environment

---

## Repository Structure

```
├── Customer_Data.csv              # Raw dataset
├── ChurnAnalysis_DAA.ipynb        # Exploratory Data Analysis
├── Untitled8__1_.ipynb            # Preprocessing + modeling + evaluation
└── README.md                      # Project documentation
```

---

## How to Run

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```
3. Open the notebooks in Jupyter and run cells in order:
   - Start with `ChurnAnalysis_DAA.ipynb` for the EDA
   - Then run `Untitled8__1_.ipynb` for preprocessing, model training, and evaluation

---

## Future Improvements

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV) for the Random Forest model
- SHAP-based feature importance analysis for deeper interpretability
- Cost-sensitive threshold selection based on actual retention campaign costs vs. customer lifetime value
- Deployment as a simple API or dashboard for business stakeholders
