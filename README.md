# UAE Personal Finance & Micro-Investment App — Analytics Dashboard

> **MBA Data Analytics · Individual PBL**  
> SP Jain Global School of Management | Dr. Anshul Gupta  
> Submitted by: **Ayush Pancholi** | MBA Finance 2025–26

---

## 📌 Project Overview

A data-driven market validation of a **UAE-focused personal finance & micro-investment mobile app** targeting young professionals, students, and expatriates. The project applies four analytical methods to answer:

> *"Which users will adopt our micro-investment features, and what feature combinations drive higher monthly investment?"*

---

## 🏗️ Repository Structure

```
uae-fintech-analytics/
│
├── app.py                          ← All-in-one Streamlit dashboard (main file)
├── generate_data.py                ← Standalone synthetic dataset generator
├── requirements.txt                ← Python dependencies
├── report.md                       ← Full business report (Markdown)
├── README.md                       ← This file
│
└── data/
    └── uae_fintech_survey_data.csv ← Synthetic survey dataset (500 rows × 29 cols)
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/uae-fintech-analytics.git
cd uae-fintech-analytics
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate dataset (optional — auto-generated on first app run)
```bash
python generate_data.py
```

### 4. Launch the Streamlit dashboard
```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

---

## 📊 Dashboard Tabs

| Tab | Content |
|-----|---------|
| 🏠 **Overview & EDA** | KPIs, demographic charts, feature interest, correlation heatmap |
| 🎯 **Classification** | Random Forest & Logistic Regression — predicts app adoption (AUC ~0.98) |
| 👥 **Clustering** | K-Means (K=4) — four user personas with radar & PCA visualisations |
| 🔗 **Association Rules** | Apriori — feature bundle discovery with lift heatmap |
| 📈 **Regression** | Random Forest & Linear Regression — forecasts monthly investment (R²~0.68) |
| 📄 **Business Report** | Full business rationale, findings, and recommendations |

---

## 🗃️ Dataset Description

**File**: `data/uae_fintech_survey_data.csv`  
**Rows**: 500 | **Columns**: 29

| Category | Columns |
|----------|---------|
| Demographics | age, gender, nationality, city, employment_status, education_level, years_in_uae |
| Financial Profile | monthly_income_aed, savings_habit, monthly_savings_aed, has_investments, investment_experience |
| Preferences | risk_appetite, sharia_compliant_preference, tech_savviness |
| Digital Behaviour | smartphone_usage_hours_daily, finance_apps_currently_used |
| Feature Interest | feature_spending_tracker, feature_savings_goals, feature_micro_investment, feature_sharia_portfolio, feature_ai_advisor, feature_peer_comparison |
| Psychographic | financial_anxiety_score, willingness_to_pay_aed_monthly, nps_score |
| **Targets** | **would_use_app** (classification), **expected_monthly_investment_aed** (regression) |

---

## 🔬 Algorithms Applied

| Algorithm | Purpose | Key Result |
|-----------|---------|------------|
| **Random Forest Classifier** | Predict app adoption | AUC = 0.978 |
| **Logistic Regression** | Baseline classification | AUC = 0.941 |
| **K-Means Clustering** | User persona segmentation | 4 personas identified |
| **Apriori** | Feature bundle discovery | 38 association rules |
| **Random Forest Regressor** | Forecast monthly investment | R² = 0.677 |
| **Linear Regression** | Baseline regression | R² = 0.512 |

---

## 🌐 Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select `app.py` as the main file
5. Click **Deploy** — the `requirements.txt` handles all dependencies automatically

---

## 📋 Assignment Checklist

- [x] Business idea selected (UAE FinTech app)
- [x] Survey form designed (29 questions with rationale)
- [x] Synthetic dataset generated (500 rows)
- [x] Rationale provided for each column
- [x] **Classification** — Random Forest + Logistic Regression
- [x] **Clustering** — K-Means with elbow method
- [x] **Association Rule Mining** — Apriori algorithm
- [x] **Regression** — Random Forest + Linear Regression *(forecasting)*
- [x] Data visualisations with two-liner insights
- [x] Algorithm documentation
- [x] Streamlit dashboard ready for deployment

---

*© 2025 Ayush Pancholi | SP Jain Global School of Management*
