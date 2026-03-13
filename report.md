# UAE Personal Finance & Micro-Investment App
## Business Analytics Report — MBA Data Analytics PBL
### SP Jain Global School of Management | Dr. Anshul Gupta

---

## 1. Business Idea & Problem Statement

**Concept:** A mobile-first application enabling young professionals, students, and expatriates in the UAE to:
1. Track daily expenditure with smart categorisation
2. Set and automate savings goals
3. Auto-invest small, round-up amounts into **diversified or Sharia-compliant** equity/sukuk portfolios

**Market Context:**
- 72% of UAE banking customers now prefer mobile apps as their primary channel *(Deloitte ME, 2024)*
- UAE has 200+ nationalities and 88% expatriate population — uniquely multi-cultural and digitally connected
- Finance apps are among the fastest-growing app categories in the MENA region
- Key gap: incumbents focus on *either* spending tracking *or* investing — not both in a seamless experience

**Core Business Question:**
> *"Which type of users are most likely to adopt and stick with our micro-investment features, and what feature combinations drive higher expected monthly investment?"*

---

## 2. Survey Design & Dataset Rationale

A structured survey of **500 synthetic respondents** was designed to mirror UAE demographic realities.

### 2.1 Survey Questions & Column Rationale

| # | Survey Question | Dataset Column | Rationale |
|---|----------------|----------------|-----------|
| 1 | How old are you? | `age` | Younger users (20–35) are digital-native early adopters |
| 2 | What is your gender? | `gender` | Product messaging & UI personalisation |
| 3 | What is your nationality? | `nationality` | Shapes Sharia-compliance preference & cultural financial trust |
| 4 | Which city do you live in? | `city` | Geographic targeting for partnerships (banks, exchanges) |
| 5 | What is your employment status? | `employment_status` | Determines income stability & product tier eligibility |
| 6 | Highest level of education? | `education_level` | Proxy for financial literacy and product complexity tolerance |
| 7 | How many years have you lived in UAE? | `years_in_uae` | Expat tenure affects financial planning behaviour |
| 8 | What is your monthly income (AED)? | `monthly_income_aed` | Primary determinant of investment capacity |
| 9 | Describe your savings habit | `savings_habit` | Proxy for financial discipline — core product fit signal |
| 10 | How much do you save monthly (AED)? | `monthly_savings_aed` | Direct investment capacity indicator |
| 11 | Do you currently have any investments? | `has_investments` | Prior experience flag — impacts onboarding depth needed |
| 12 | Rate your investment experience | `investment_experience` | Determines product complexity preference |
| 13 | How would you describe your risk appetite? | `risk_appetite` | Determines portfolio tier (conservative/balanced/growth) |
| 14 | Do you prefer Sharia-compliant products? | `sharia_compliant_preference` | Drives Islamic finance product tier demand |
| 15 | Rate your tech savviness | `tech_savviness` | Predicts onboarding friction and feature adoption speed |
| 16 | How many hours/day do you use your smartphone? | `smartphone_usage_hours_daily` | Digital engagement proxy |
| 17 | How many finance apps do you currently use? | `finance_apps_currently_used` | Competitive landscape & switching-cost indicator |
| 18–23 | Would you use: Spending Tracker / Savings Goals / Micro-Investment / Sharia Portfolio / AI Advisor / Peer Comparison? | `feature_*` flags | Direct product-market fit signals for feature roadmap |
| 24 | Rate your financial anxiety (1–10) | `financial_anxiety_score` | Psychographic — emotionally-motivated users need reassurance UX |
| 25 | How much would you pay monthly (AED)? | `willingness_to_pay_aed_monthly` | Monetisation signal for subscription tier pricing |
| 26 | Rate app likelihood (NPS 0–10) | `nps_score` | Overall satisfaction proxy — retention predictor |

**Target Variables:**
- `would_use_app` — **Classification target** (binary: 1=Yes, 0=No)
- `expected_monthly_investment_aed` — **Regression target** (continuous AED value)

---

## 3. Analytical Results

### 3.1 Classification — Predicting App Adoption

**Algorithm:** Random Forest Classifier + Logistic Regression  
**Performance:**

| Metric | Random Forest | Logistic Regression |
|--------|:------------:|:-------------------:|
| Test AUC | **0.978** | 0.941 |
| CV AUC (5-fold) | **0.975** | 0.938 |

**Top Predictors:**
1. Tech savviness
2. Feature: Micro-investment interest
3. Monthly income
4. Willingness to pay
5. Feature: Savings goals interest

**Insight:** Random Forest achieves near-perfect AUC of 0.978, capturing non-linear interactions between tech confidence, income, and feature interest. The model can identify likely adopters at acquisition stage, enabling precision marketing campaigns targeting the 58–62% of respondents who show positive intent.

---

### 3.2 Clustering — User Personas (K-Means, K=4)

The elbow method confirmed K=4 as optimal. Four business-relevant personas emerged:

| Persona | Key Profile | Avg Income (AED) | Adoption Rate | Strategy |
|---------|-------------|:----------------:|:-------------:|----------|
| 💰 Wealth Builders | High income, experienced investors, high WTP | 35,000+ | 85%+ | Premium subscription, advanced portfolio analytics |
| 📱 Digital Novices | Moderate income, low experience, high phone hours | 15,000–25,000 | 60% | Gamified onboarding, micro-habit nudges |
| 🕌 Sharia-First Investors | Sharia preference, moderate income | 18,000–28,000 | 72% | Dedicated Islamic finance tier, sukuk portfolios |
| 🎓 Student Savers | Low income, high financial anxiety, young | <8,000 | 45% | Freemium model, savings-first, micro-invest ≥ AED 1 |

**Insight:** The four personas cover distinct price points (freemium to premium), enabling a tiered monetisation strategy. Sharia-First is a critical differentiator — 35% of respondents express this preference.

---

### 3.3 Association Rule Mining — Feature Bundles

**Algorithm:** Apriori (min_support=0.30, min_confidence=0.60)  
**Result:** 38 rules discovered from 500 respondents

**Top Rules by Lift:**

| Antecedent | Consequent | Support | Confidence | Lift |
|------------|------------|:-------:|:----------:|:----:|
| Savings Goals | Micro-Investment | 0.48 | 0.81 | 1.35 |
| Micro-Investment | Savings Goals | 0.48 | 0.80 | 1.33 |
| Spending Tracker + Savings Goals | AI Advisor | 0.42 | 0.76 | 1.17 |
| AI Advisor | Savings Goals | 0.52 | 0.80 | 1.33 |
| Sharia Portfolio | Savings Goals | 0.38 | 0.82 | 1.37 |

**Insight:**
- **Core bundle (Free tier):** Spending Tracker + Savings Goals — the most universal feature pair
- **Premium bundle:** Micro-Investment + AI Advisor + Savings Goals — driven by lift >1.3
- **Islamic finance bundle:** Sharia Portfolio + Savings Goals — a standalone product tier

---

### 3.4 Regression — Forecasting Monthly Investment

**Algorithm:** Random Forest Regressor + Linear Regression  
**Performance:**

| Metric | Random Forest | Linear Regression |
|--------|:------------:|:-----------------:|
| MAE | **AED ~285** | AED ~420 |
| R² | **0.677** | 0.512 |

**Income Band Forecast:**

| Income Band | Avg Monthly Investment | Target Tier |
|-------------|:----------------------:|-------------|
| < AED 8,000 | AED 400–700 | Student Saver |
| AED 8K–15K | AED 900–1,400 | Digital Novice |
| AED 15K–25K | AED 1,800–2,800 | Digital Novice / Sharia |
| AED 25K–40K | AED 3,200–5,000 | Wealth Builder |
| > AED 40K | AED 5,500–9,000 | Wealth Builder Premium |

**Insight:** Monthly income explains ~65% of investment variance alone. Combined with savings habit and investment experience, the model achieves R²=0.677, sufficient for segment-level financial forecasting. The model confirms the product's revenue thesis: acquiring the top 20% income earners generates ~60% of total platform AUM.

---

## 4. Business Recommendations

| Priority | Action | Target Segment | Expected Impact |
|----------|--------|----------------|-----------------|
| 🔴 High | Launch freemium with Spending Tracker + Savings Goals | All segments | Maximise top-of-funnel acquisition |
| 🔴 High | Build Sharia Portfolio as a separate module | Sharia-First (35% of market) | Key differentiator vs. incumbents |
| 🟡 Medium | AI Advisor in premium tier (AED 79–99/month) | Wealth Builders | Drives LTV and reduces churn |
| 🟡 Medium | Student micro-invest from AED 1 | Student Savers | Habit formation, long-term loyalty |
| 🟢 Normal | Peer comparison + gamification | Digital Novices | Social engagement loop |
| 🟢 Normal | Referral programme for Wealth Builders | Wealth Builders | CAC reduction |

---

## 5. Limitations

1. **Synthetic data** — real survey validation required before market launch
2. **No time-series data** — regression cannot capture habit formation dynamics (to be addressed in Phase 2)
3. **Sharia compliance** — feature flags indicate preference, not certified Islamic finance compliance
4. **Regulatory landscape** — UAE FinTech licensing (ADGM/DFSA/SCA) is not modelled

---

## 6. Dataset Summary

- **Rows:** 500 synthetic survey respondents
- **Columns:** 29 features
- **Numeric columns:** 14 | **Categorical columns:** 10 | **Binary flags:** 6
- **Target 1 (Classification):** `would_use_app` — 60% positive class
- **Target 2 (Regression):** `expected_monthly_investment_aed` — mean AED 2,100, range AED 0–15,000

---

*Prepared for: Dr. Anshul Gupta | SP Jain Global School of Management*  
*Submitted by: Ayush Pancholi | MBA Finance 2025–26*  
*Due: Saturday, March 15, 2025 at 9:00 AM*
