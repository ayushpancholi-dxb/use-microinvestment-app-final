"""
generate_data.py
Synthetic dataset generator for UAE Personal Finance & Micro-Investment App study.
Run standalone:  python generate_data.py
"""

import numpy as np
import pandas as pd
import os

SEED = 42
N = 500


def generate_dataset(n: int = N, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── Demographics ─────────────────────────────────────────────────────────────
    age = rng.integers(20, 46, size=n)
    gender = rng.choice(["Male", "Female", "Non-binary/Other"],
                        size=n, p=[0.55, 0.42, 0.03])
    nationality = rng.choice(
        ["Indian", "Emirati", "Pakistani", "Filipino", "British", "Other"],
        size=n, p=[0.30, 0.15, 0.15, 0.10, 0.10, 0.20],
    )
    city = rng.choice(
        ["Dubai", "Abu Dhabi", "Sharjah", "Ajman", "Other"],
        size=n, p=[0.48, 0.25, 0.15, 0.07, 0.05],
    )
    employment_status = rng.choice(
        ["Employed", "Student", "Self-Employed", "Unemployed"],
        size=n, p=[0.55, 0.25, 0.15, 0.05],
    )
    education_level = rng.choice(
        ["High School", "Bachelor", "Master", "PhD"],
        size=n, p=[0.10, 0.50, 0.33, 0.07],
    )
    years_in_uae = rng.integers(0, 16, size=n)
    years_in_uae = np.where(nationality == "Emirati",
                            rng.integers(0, 46, size=n),
                            years_in_uae)

    # ── Income & savings ──────────────────────────────────────────────────────────
    base_income_map = {
        "Employed": 18000, "Self-Employed": 22000,
        "Student": 3000,   "Unemployed": 4000,
    }
    base_income = np.array([base_income_map[e] for e in employment_status])
    monthly_income_aed = (base_income + rng.normal(0, 5000, size=n)).clip(0, 80000).astype(int)

    savings_habit = np.empty(n, dtype=object)
    low_mask  = monthly_income_aed < 8000
    high_mask = ~low_mask
    savings_habit[low_mask]  = rng.choice(["None","Irregular","Regular"],
                                           size=low_mask.sum(), p=[0.50, 0.35, 0.15])
    savings_habit[high_mask] = rng.choice(["None","Irregular","Regular"],
                                           size=high_mask.sum(), p=[0.10, 0.40, 0.50])

    monthly_savings_aed = np.where(
        savings_habit == "None", 0,
        np.where(savings_habit == "Irregular",
                 rng.integers(100, 1500, size=n),
                 rng.integers(500, 5000, size=n)),
    )

    # ── Investment profile ────────────────────────────────────────────────────────
    has_investments = rng.choice([1, 0], size=n, p=[0.38, 0.62])
    investment_experience = np.empty(n, dtype=object)
    no_inv  = has_investments == 0
    yes_inv = ~no_inv
    investment_experience[no_inv]  = rng.choice(["None", "Beginner"],
                                                 size=no_inv.sum(), p=[0.70, 0.30])
    investment_experience[yes_inv] = rng.choice(
        ["Beginner", "Intermediate", "Advanced"],
        size=yes_inv.sum(), p=[0.40, 0.40, 0.20])

    risk_appetite = rng.choice(["Low", "Medium", "High"], size=n, p=[0.30, 0.45, 0.25])
    sharia_pref   = rng.choice(["Yes", "No", "No Preference"],
                               size=n, p=[0.35, 0.30, 0.35])

    # ── Tech & digital behaviour ──────────────────────────────────────────────────
    tech_savviness      = rng.choice(["Low", "Medium", "High"], size=n, p=[0.15, 0.45, 0.40])
    smartphone_hours    = rng.normal(5.5, 2.0, size=n).clip(1, 14).round(1)
    finance_apps_used   = rng.integers(0, 5, size=n)

    # ── Feature-interest flags ────────────────────────────────────────────────────
    feat_spending   = rng.choice([0, 1], size=n, p=[0.25, 0.75])
    feat_savings    = rng.choice([0, 1], size=n, p=[0.20, 0.80])
    feat_micro_inv  = rng.choice([0, 1], size=n, p=[0.40, 0.60])

    feat_sharia = np.empty(n, dtype=int)
    sharia_yes  = sharia_pref == "Yes"
    sharia_no   = ~sharia_yes
    feat_sharia[sharia_yes] = rng.choice([0, 1], size=sharia_yes.sum(), p=[0.10, 0.90])
    feat_sharia[sharia_no]  = rng.choice([0, 1], size=sharia_no.sum(),  p=[0.70, 0.30])

    feat_ai_advisor = rng.choice([0, 1], size=n, p=[0.35, 0.65])
    feat_peer_cmp   = rng.choice([0, 1], size=n, p=[0.50, 0.50])

    # ── Psychographic ─────────────────────────────────────────────────────────────
    financial_anxiety    = rng.integers(1, 11, size=n)
    willingness_to_pay   = np.where(
        monthly_income_aed > 20000, rng.integers(50, 200, size=n),
        np.where(monthly_income_aed > 8000, rng.integers(20, 100, size=n),
                 rng.integers(0, 40, size=n))
    )

    # ── TARGET 1 – Classification: would_use_app ──────────────────────────────────
    score = (
        0.30 * (tech_savviness == "High").astype(int) +
        0.20 * feat_micro_inv +
        0.15 * feat_savings +
        0.10 * (monthly_income_aed > 12000).astype(int) +
        0.10 * has_investments +
        0.05 * (finance_apps_used >= 2).astype(int) +
        0.05 * (age < 35).astype(int) +
        0.05 * (risk_appetite == "High").astype(int)
    )
    noise = rng.uniform(0, 0.15, size=n)
    would_use_app = (score + noise > 0.45).astype(int)

    # ── TARGET 2 – Regression: expected_monthly_investment_aed ───────────────────
    exp_map = {"None": 0, "Beginner": 200, "Intermediate": 600, "Advanced": 1200}
    risk_map = {"Low": 0, "Medium": 400, "High": 800}
    exp_bonus  = np.array([exp_map[e]  for e in investment_experience])
    risk_bonus = np.array([risk_map[r] for r in risk_appetite])

    base_invest = (
        monthly_income_aed * 0.05
        + monthly_savings_aed * 0.20
        + risk_bonus
        + exp_bonus
        + feat_micro_inv * 300
    )
    expected_monthly_investment_aed = (
        base_invest + rng.normal(0, 300, size=n)
    ).clip(0, 15000).round(0).astype(int)

    # ── NPS score ─────────────────────────────────────────────────────────────────
    nps_score = (
        rng.integers(5, 11, size=n) * would_use_app +
        rng.integers(0, 7, size=n)  * (1 - would_use_app)
    ).clip(0, 10)

    df = pd.DataFrame({
        "respondent_id":                   [f"R{str(i).zfill(4)}" for i in range(1, n + 1)],
        "age":                             age,
        "gender":                          gender,
        "nationality":                     nationality,
        "city":                            city,
        "employment_status":               employment_status,
        "education_level":                 education_level,
        "years_in_uae":                    years_in_uae,
        "monthly_income_aed":              monthly_income_aed,
        "savings_habit":                   savings_habit,
        "monthly_savings_aed":             monthly_savings_aed,
        "has_investments":                 has_investments,
        "investment_experience":           investment_experience,
        "risk_appetite":                   risk_appetite,
        "sharia_compliant_preference":     sharia_pref,
        "tech_savviness":                  tech_savviness,
        "smartphone_usage_hours_daily":    smartphone_hours,
        "finance_apps_currently_used":     finance_apps_used,
        "feature_spending_tracker":        feat_spending,
        "feature_savings_goals":           feat_savings,
        "feature_micro_investment":        feat_micro_inv,
        "feature_sharia_portfolio":        feat_sharia,
        "feature_ai_advisor":              feat_ai_advisor,
        "feature_peer_comparison":         feat_peer_cmp,
        "financial_anxiety_score":         financial_anxiety,
        "willingness_to_pay_aed_monthly":  willingness_to_pay,
        "nps_score":                       nps_score,
        "would_use_app":                   would_use_app,
        "expected_monthly_investment_aed": expected_monthly_investment_aed,
    })
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_dataset()
    path = os.path.join("data", "uae_fintech_survey_data.csv")
    df.to_csv(path, index=False)
    print(f"✅  Dataset saved to {path}  ({len(df)} rows × {len(df.columns)} cols)")
    print(df.dtypes)
    print(df.head(3))
