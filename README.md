# 💼 Global AI Job Salary Prediction

A complete end-to-end machine learning project that predicts annual salary (USD) for AI/tech roles across 20 countries. Built as Portfolio Project #1 to demonstrate the full data science workflow — from raw data to a deployed web application.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/user/AbiodunOladejo)

---

## 📁 Repository Structure

```
├── AI_Salary_Prediction_FINAL.ipynb   # Complete ML notebook
├── app.py                             # Streamlit web application
├── salary_model_final.pkl             # Trained model (XGBoost Pipeline)
├── encoding_ref.json                  # Feature encoding reference for app
├── ai_job_dataset.csv                 # Dataset — 15,000 AI job postings
├── requirements.txt                   # Python dependencies
├── data_dictionary.md                 # Column definitions and use cases
└── README.md                          # This file
```

---

## 🎯 Problem Statement

> Given a job posting's features — title, experience level, location, company size, and others — can we accurately predict the annual salary in USD?

This is a **supervised regression** problem. The target variable is `salary_usd`.

---

## 📊 Dataset

- **Size:** 15,000 rows, 19 columns (14,517 after outlier removal)
- **Coverage:** 20 countries, 15 industries, 20 unique job titles
- **Missing values:** None

See [`data_dictionary.md`](data_dictionary.md) for full column descriptions and feature decisions.

---

## 🔧 Workflow Summary

| Step | Description |
|---|---|
| 1 | **Baseline** — predict mean salary for all jobs → MAE ≈ $41,225 (minimum bar to beat) |
| 2 | **Data Cleaning** — `wrangle()` function, drop irrelevant columns, remove salary outliers via IQR |
| 3 | **EDA** — salary patterns by experience, company size, country, industry, work mode |
| 4 | **Feature Engineering** — encode categoricals, engineer `same_country` binary feature |
| 5 | **Round 1** — 5 models with manual ordinal + frequency-rank encoding |
| 6 | **Round 2** — same 5 models rebuilt using `make_pipeline` + `OneHotEncoder` |
| 7 | **Model Selection** — compare all 10 on MAE, RMSE, R², Train vs Test gap |
| 8 | **Cross-Validation** — 5-fold CV confirms stability (R² = 0.882 ± 0.004) |
| 9 | **Hyperparameter Tuning** — GridSearchCV on winning model |
| 10 | **Deployment** — Streamlit web app with live salary predictor |

---

## 📈 Results

### All 10 Models — Test Set

| Regressor | Approach | R² | MAE | RMSE % of Range |
|---|---|---|---|---|
| **XGBoost ⭐** | **Pipeline + OHE** | **0.884** | **$12,912** | **7.54%** |
| Gradient Boosting | Pipeline + OHE | 0.871 | $13,576 | 7.97% |
| Linear Regression | Pipeline + OHE | 0.865 | $14,231 | 8.14% |
| Ridge Regression | Pipeline + OHE | 0.865 | $14,175 | 8.14% |
| Random Forest | Pipeline + OHE | 0.864 | $14,239 | 8.18% |
| Gradient Boosting | Manual Encoding | 0.644 | $23,884 | 13.23% |
| Ridge Regression | Manual Encoding | 0.638 | $24,111 | 13.33% |
| Linear Regression | Manual Encoding | 0.638 | $24,111 | 13.33% |
| XGBoost | Manual Encoding | 0.637 | $23,983 | 13.36% |
| Random Forest ⚠️ | Manual Encoding | 0.635 | $24,036 | 13.40% |

### Baseline vs Best Model

| | MAE |
|---|---|
| Baseline (predict mean salary for everyone) | ~$41,225|
| **Tuned XGBoost Pipeline** | **~$12,945** |
| **Improvement** | **68%+ reduction in prediction error** |

### Cross-Validation (5-fold)
Fold scores: `0.888, 0.879, 0.878, 0.880, 0.885` → **Mean R²: 0.882 ± 0.004**

---

## 🏆 Key Findings

1. **Experience level is the #1 salary driver** — Executives earn ~3× more than Entry-level
2. **Pipeline + OHE dramatically outperforms manual encoding** — R² improved from 0.64 to 0.88
3. **Large companies pay ~$28K more** on average than small companies
4. **Geography matters significantly** — top-paying countries earn up to 2× more than lowest
5. **XGBoost benefits most from rich feature representation** — ranked near-bottom in Round 1, first in Round 2

---

## 🛠️ Tech Stack

| | |
|---|---|
| Data | pandas, numpy |
| Visualisation | matplotlib, seaborn, plotly |
| Machine Learning | scikit-learn, XGBoost |
| Pipeline | `make_pipeline`, `ColumnTransformer`, `OneHotEncoder` |
| Tuning | `GridSearchCV`, `cross_val_score` |
| Deployment | Streamlit |
| Model persistence | joblib |

---

## 🚀 Running Locally

```bash
git clone https://github.com/AbiodunOladejo/ai-job-salary-prediction.git
cd ai-job-salary-prediction
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔮 Future Improvements

- Apply NLP to `required_skills` using TF-IDF or word embeddings
- Add geographic cost-of-living adjustments as a feature
- Explore interaction features (e.g. `experience_level × company_size`)
- Experiment with model stacking

---

## 👤 Author

**Abiodun Oladejo**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/abiodun-amina-oladejo/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/AbiodunOladejo)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/abiodunoladejo)
[![Medium](https://img.shields.io/badge/Medium-12100E?style=flat&logo=medium&logoColor=white)](https://medium.com/@AbiodunOladejo)
[![Portfolio](https://img.shields.io/badge/Portfolio-FF5722?style=flat&logo=Google-chrome&logoColor=white)](https://www.datascienceportfol.io/AbiodunOladejo)

📧 abiodunoladej@gmail.com
