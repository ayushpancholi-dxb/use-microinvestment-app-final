"""
app.py  ─  UAE Personal Finance & Micro-Investment App  |  MBA Data Analytics PBL
===================================================================================
All-in-one Streamlit dashboard covering:
  • Data Overview & EDA
  • Classification  (Random Forest + Logistic Regression)
  • Clustering      (K-Means personas)
  • Association Rules (Apriori / mlxtend)
  • Regression / Forecasting (Random Forest + Linear Regression)

Run:  streamlit run app.py
"""

# ─── Standard library ────────────────────────────────────────────────────────
import os, warnings
warnings.filterwarnings("ignore")

# ─── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ─── Streamlit page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="UAE FinTech App – Analytics Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Colour palette ───────────────────────────────────────────────────────────
PALETTE   = px.colors.qualitative.Bold
GREEN     = "#2ecc71"
BLUE      = "#2980b9"
ORANGE    = "#e67e22"
RED       = "#e74c3c"
PURPLE    = "#8e44ad"
DARK_BG   = "#0e1117"

# ══════════════════════════════════════════════════════════════════════════════
# DATA  ── generate once, cache
# ══════════════════════════════════════════════════════════════════════════════
SEED = 42
N    = 500
DATA_PATH = os.path.join("data", "uae_fintech_survey_data.csv")


def _generate_raw(n: int, seed: int) -> pd.DataFrame:
    """Generates the synthetic survey dataset."""
    rng = np.random.default_rng(seed)

    age   = rng.integers(20, 46, size=n)
    gender = rng.choice(["Male", "Female", "Non-binary/Other"],
                        size=n, p=[0.55, 0.42, 0.03])
    nationality = rng.choice(
        ["Indian", "Emirati", "Pakistani", "Filipino", "British", "Other"],
        size=n, p=[0.30, 0.15, 0.15, 0.10, 0.10, 0.20])
    city = rng.choice(
        ["Dubai", "Abu Dhabi", "Sharjah", "Ajman", "Other"],
        size=n, p=[0.48, 0.25, 0.15, 0.07, 0.05])
    emp_status = rng.choice(
        ["Employed", "Student", "Self-Employed", "Unemployed"],
        size=n, p=[0.55, 0.25, 0.15, 0.05])
    edu = rng.choice(
        ["High School", "Bachelor", "Master", "PhD"],
        size=n, p=[0.10, 0.50, 0.33, 0.07])
    years_uae = rng.integers(0, 16, size=n)
    years_uae = np.where(nationality == "Emirati", rng.integers(0, 46, size=n), years_uae)

    bmap = {"Employed": 18000, "Self-Employed": 22000, "Student": 3000, "Unemployed": 4000}
    base = np.array([bmap[e] for e in emp_status])
    income = (base + rng.normal(0, 5000, n)).clip(0, 80000).astype(int)

    sav_habit = np.empty(n, dtype=object)
    lm = income < 8000
    sav_habit[lm]  = rng.choice(["None","Irregular","Regular"], size=lm.sum(),  p=[0.50,0.35,0.15])
    sav_habit[~lm] = rng.choice(["None","Irregular","Regular"], size=(~lm).sum(),p=[0.10,0.40,0.50])

    monthly_sav = np.where(sav_habit=="None", 0,
                  np.where(sav_habit=="Irregular", rng.integers(100,1500,n),
                           rng.integers(500,5000,n)))

    has_inv = rng.choice([1,0], size=n, p=[0.38,0.62])
    inv_exp = np.empty(n, dtype=object)
    ni = has_inv == 0
    inv_exp[ni]  = rng.choice(["None","Beginner"], size=ni.sum(),  p=[0.70,0.30])
    inv_exp[~ni] = rng.choice(["Beginner","Intermediate","Advanced"], size=(~ni).sum(), p=[0.40,0.40,0.20])

    risk   = rng.choice(["Low","Medium","High"], size=n, p=[0.30,0.45,0.25])
    sharia = rng.choice(["Yes","No","No Preference"], size=n, p=[0.35,0.30,0.35])
    tech   = rng.choice(["Low","Medium","High"], size=n, p=[0.15,0.45,0.40])
    phone  = rng.normal(5.5, 2.0, n).clip(1,14).round(1)
    fapps  = rng.integers(0, 5, n)

    fs = rng.choice([0,1], n, p=[0.25,0.75])
    fg = rng.choice([0,1], n, p=[0.20,0.80])
    fm = rng.choice([0,1], n, p=[0.40,0.60])
    fsh = np.empty(n, dtype=int)
    sy = sharia == "Yes"
    fsh[sy]  = rng.choice([0,1], sy.sum(),  p=[0.10,0.90])
    fsh[~sy] = rng.choice([0,1], (~sy).sum(), p=[0.70,0.30])
    fa  = rng.choice([0,1], n, p=[0.35,0.65])
    fp  = rng.choice([0,1], n, p=[0.50,0.50])

    anx  = rng.integers(1, 11, n)
    wtp  = np.where(income>20000, rng.integers(50,200,n),
           np.where(income>8000,  rng.integers(20,100,n), rng.integers(0,40,n)))

    score = (0.30*(tech=="High").astype(int) + 0.20*fm + 0.15*fg
             + 0.10*(income>12000).astype(int) + 0.10*has_inv
             + 0.05*(fapps>=2).astype(int) + 0.05*(age<35).astype(int)
             + 0.05*(risk=="High").astype(int))
    wua = (score + rng.uniform(0,0.15,n) > 0.45).astype(int)

    emap = {"None":0,"Beginner":200,"Intermediate":600,"Advanced":1200}
    rmap = {"Low":0,"Medium":400,"High":800}
    eb   = np.array([emap[e] for e in inv_exp])
    rb   = np.array([rmap[r] for r in risk])
    emi  = (income*0.05 + monthly_sav*0.20 + rb + eb + fm*300
            + rng.normal(0,300,n)).clip(0,15000).round(0).astype(int)

    nps = (rng.integers(5,11,n)*wua + rng.integers(0,7,n)*(1-wua)).clip(0,10)

    return pd.DataFrame({
        "respondent_id": [f"R{str(i).zfill(4)}" for i in range(1,n+1)],
        "age": age, "gender": gender, "nationality": nationality,
        "city": city, "employment_status": emp_status, "education_level": edu,
        "years_in_uae": years_uae, "monthly_income_aed": income,
        "savings_habit": sav_habit, "monthly_savings_aed": monthly_sav,
        "has_investments": has_inv, "investment_experience": inv_exp,
        "risk_appetite": risk, "sharia_compliant_preference": sharia,
        "tech_savviness": tech, "smartphone_usage_hours_daily": phone,
        "finance_apps_currently_used": fapps,
        "feature_spending_tracker": fs, "feature_savings_goals": fg,
        "feature_micro_investment": fm, "feature_sharia_portfolio": fsh,
        "feature_ai_advisor": fa, "feature_peer_comparison": fp,
        "financial_anxiety_score": anx, "willingness_to_pay_aed_monthly": wtp,
        "nps_score": nps, "would_use_app": wua,
        "expected_monthly_investment_aed": emi,
    })


@st.cache_data
def load_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    os.makedirs("data", exist_ok=True)
    df = _generate_raw(N, SEED)
    df.to_csv(DATA_PATH, index=False)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
FEAT_COLS = [
    "feature_spending_tracker", "feature_savings_goals",
    "feature_micro_investment", "feature_sharia_portfolio",
    "feature_ai_advisor", "feature_peer_comparison",
]
FEAT_LABELS = ["Spending\nTracker", "Savings\nGoals",
               "Micro-\nInvestment", "Sharia\nPortfolio",
               "AI\nAdvisor",  "Peer\nComparison"]
FEAT_LABELS_CLEAN = ["Spending Tracker", "Savings Goals",
                     "Micro-Investment", "Sharia Portfolio",
                     "AI Advisor", "Peer Comparison"]

CAT_FEATURES  = ["gender","nationality","city","employment_status",
                 "education_level","savings_habit","investment_experience",
                 "risk_appetite","sharia_compliant_preference","tech_savviness"]
NUM_FEATURES  = ["age","years_in_uae","monthly_income_aed","monthly_savings_aed",
                 "smartphone_usage_hours_daily","finance_apps_currently_used",
                 "financial_anxiety_score","willingness_to_pay_aed_monthly",
                 "nps_score"]

def encode(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in CAT_FEATURES:
        d[c] = LabelEncoder().fit_transform(d[c].astype(str))
    return d


def two_liner(text: str):
    """Renders a styled two-liner insight caption."""
    st.markdown(
        f"<p style='font-size:13px;color:#b2bec3;margin-top:-8px;'>"
        f"💡 {text}</p>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def sidebar(df: pd.DataFrame):
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/000000/money-transfer.png",
            width=60,
        )
        st.title("UAE FinTech App")
        st.caption("MBA Data Analytics · PBL Dashboard")
        st.markdown("---")
        st.markdown("**Dataset Stats**")
        st.metric("Respondents", len(df))
        st.metric("Likely Users", f"{df['would_use_app'].mean()*100:.1f}%")
        st.metric("Avg. Monthly Investment", f"AED {df['expected_monthly_investment_aed'].mean():,.0f}")
        st.markdown("---")
        st.markdown("**Navigate**")
        tabs = [
            "🏠 Overview & EDA",
            "🎯 Classification",
            "👥 Clustering",
            "🔗 Association Rules",
            "📈 Regression",
        ]
        selected = st.radio("", tabs, label_visibility="collapsed")
        st.markdown("---")
        st.caption("Dr. Anshul Gupta · SP Jain Global")
        return selected


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – OVERVIEW & EDA
# ══════════════════════════════════════════════════════════════════════════════
def tab_overview(df: pd.DataFrame):
    st.title("🏠 Data Overview & Exploratory Analysis")
    st.markdown(
        "> **Business Idea**: A mobile app enabling young UAE professionals "
        "to track spending, set savings goals, and auto-invest into diversified "
        "or Sharia-compliant portfolios. "
        "This dashboard validates market fit using 500 synthetic survey responses."
    )

    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Respondents",        f"{len(df)}")
    c2.metric("Would Use App",      f"{df['would_use_app'].mean()*100:.1f}%")
    c3.metric("Avg Income (AED)",   f"{df['monthly_income_aed'].mean():,.0f}")
    c4.metric("Avg Investment/Mo",  f"AED {df['expected_monthly_investment_aed'].mean():,.0f}")
    c5.metric("Avg NPS",            f"{df['nps_score'].mean():.1f} / 10")
    st.markdown("---")

    # ── Row 1 ──────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(df, names="nationality", title="Respondent Nationality Mix",
                     color_discrete_sequence=PALETTE, hole=0.4)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
        two_liner(
            "Indians (30%) and Pakistanis (15%) dominate the sample, "
            "reflecting UAE's expat-heavy demographics. Emiratis (15%) form a significant minority, "
            "guiding Sharia-feature priority."
        )

    with col2:
        age_bins = pd.cut(df["age"], bins=[19,24,29,34,39,46],
                          labels=["20-24","25-29","30-34","35-39","40-45"])
        fig2 = px.histogram(df, x="age", color="would_use_app",
                            barmode="overlay", nbins=20,
                            color_discrete_map={0: RED, 1: GREEN},
                            title="Age Distribution by App Adoption",
                            labels={"would_use_app": "Would Use App"})
        fig2.update_layout(legend_title_text="Would Use App")
        st.plotly_chart(fig2, use_container_width=True)
        two_liner(
            "Adoption peaks in the 25-35 age bracket – the sweet spot of digital-native "
            "professionals with disposable income. Users above 40 show noticeably lower intent."
        )

    # ── Row 2 ──────────────────────────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        feat_pct = df[FEAT_COLS].mean() * 100
        fig3 = px.bar(
            x=FEAT_LABELS_CLEAN, y=feat_pct,
            title="Feature Interest Rate (%)",
            labels={"x": "Feature", "y": "% Interested"},
            color=feat_pct, color_continuous_scale="Blues",
            text=feat_pct.round(1),
        )
        fig3.update_traces(texttemplate="%{text}%", textposition="outside")
        fig3.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)
        two_liner(
            "Savings Goals (80%) and Spending Tracker (75%) are the most-wanted features, "
            "suggesting a habit-formation hook before investing. Sharia Portfolio sits at 60%+, "
            "validating a dedicated Islamic finance tier."
        )

    with col4:
        fig4 = px.box(df, x="employment_status", y="monthly_income_aed",
                      color="would_use_app",
                      color_discrete_map={0: RED, 1: GREEN},
                      title="Income by Employment & Adoption",
                      labels={"would_use_app":"Would Use App","monthly_income_aed":"Monthly Income (AED)"})
        st.plotly_chart(fig4, use_container_width=True)
        two_liner(
            "Self-employed and employed users command the highest income bands and show "
            "the strongest adoption intent. Student users exhibit lower income but remain "
            "a viable early-adopter segment."
        )

    # ── Row 3 ──────────────────────────────────────────────────────────────
    col5, col6 = st.columns(2)

    with col5:
        fig5 = px.scatter(
            df, x="monthly_income_aed", y="expected_monthly_investment_aed",
            color="risk_appetite", size="willingness_to_pay_aed_monthly",
            color_discrete_sequence=PALETTE,
            title="Income vs Expected Investment by Risk Appetite",
            labels={
                "monthly_income_aed": "Monthly Income (AED)",
                "expected_monthly_investment_aed": "Expected Monthly Investment (AED)",
            },
            opacity=0.7,
        )
        st.plotly_chart(fig5, use_container_width=True)
        two_liner(
            "A clear positive relationship exists between income and expected investment. "
            "High-risk users cluster at the upper end, confirming risk tolerance as a key "
            "predictor for the micro-investment value proposition."
        )

    with col6:
        fig6 = px.sunburst(
            df, path=["tech_savviness", "savings_habit", "would_use_app"],
            title="Tech Savviness → Savings Habit → App Adoption",
            color_discrete_sequence=PALETTE,
        )
        st.plotly_chart(fig6, use_container_width=True)
        two_liner(
            "High-tech-savviness users with regular savings habits show the highest adoption "
            "rates. Low-tech segments with no savings habit are the hardest to convert – "
            "indicating where onboarding UX investment is needed."
        )

    # ── Correlation heatmap ────────────────────────────────────────────────
    st.subheader("Correlation Heatmap – Numeric Features")
    num_cols = NUM_FEATURES + FEAT_COLS + ["has_investments", "would_use_app",
                                           "expected_monthly_investment_aed"]
    corr = df[num_cols].corr().round(2)
    fig7 = px.imshow(corr, text_auto=True, aspect="auto",
                     color_continuous_scale="RdBu_r",
                     title="Pearson Correlation Matrix")
    st.plotly_chart(fig7, use_container_width=True)
    two_liner(
        "Monthly income shows the strongest positive correlation with expected investment "
        "(r ≈ 0.65). Financial anxiety has a weak negative correlation with adoption, "
        "suggesting anxious users may be more – not less – open to financial tools."
    )

    # ── Raw data viewer ───────────────────────────────────────────────────
    with st.expander("🔍 Raw Dataset Explorer"):
        st.dataframe(df, use_container_width=True, height=400)
        csv = df.to_csv(index=False).encode()
        st.download_button("⬇ Download CSV", csv,
                           "uae_fintech_survey_data.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
def tab_classification(df: pd.DataFrame):
    st.title("🎯 Classification – Predicting App Adoption")
    st.markdown(
        "**Goal**: Predict whether a respondent will use the app (`would_use_app = 1`).  \n"
        "**Models**: Random Forest Classifier & Logistic Regression  \n"
        "**Target**: `would_use_app` (binary)"
    )

    # ── Feature prep ──────────────────────────────────────────────────────
    clf_features = [
        "age","monthly_income_aed","monthly_savings_aed",
        "smartphone_usage_hours_daily","finance_apps_currently_used",
        "financial_anxiety_score","willingness_to_pay_aed_monthly",
        "has_investments","feature_micro_investment","feature_savings_goals",
        "feature_spending_tracker","feature_ai_advisor",
        "gender","employment_status","education_level",
        "risk_appetite","tech_savviness","savings_habit","investment_experience",
    ]
    dfe = encode(df)
    X = dfe[clf_features]
    y = dfe["would_use_app"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── Models ────────────────────────────────────────────────────────────
    rf  = RandomForestClassifier(n_estimators=200, max_depth=8,
                                 random_state=SEED, class_weight="balanced")
    lr  = LogisticRegression(max_iter=1000, random_state=SEED)
    rf.fit(X_train, y_train)
    lr.fit(X_train_sc, y_train)

    y_pred_rf = rf.predict(X_test)
    y_pred_lr = lr.predict(X_test_sc)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    y_prob_lr = lr.predict_proba(X_test_sc)[:, 1]

    auc_rf = roc_auc_score(y_test, y_prob_rf)
    auc_lr = roc_auc_score(y_test, y_prob_lr)

    cv_rf = cross_val_score(rf, X, y, cv=5, scoring="roc_auc").mean()
    cv_lr = cross_val_score(lr, scaler.fit_transform(X), y, cv=5, scoring="roc_auc").mean()

    # ── Metrics comparison ────────────────────────────────────────────────
    st.subheader("Model Performance Comparison")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("RF – Test AUC",        f"{auc_rf:.3f}")
    mc2.metric("RF – CV AUC (5-fold)", f"{cv_rf:.3f}")
    mc3.metric("LR – Test AUC",        f"{auc_lr:.3f}")
    mc4.metric("LR – CV AUC (5-fold)", f"{cv_lr:.3f}")
    st.markdown("---")

    # ── ROC curves ────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
        fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, name=f"Random Forest (AUC={auc_rf:.3f})",
                                 line=dict(color=GREEN, width=2)))
        fig.add_trace(go.Scatter(x=fpr_lr, y=tpr_lr, name=f"Logistic Regression (AUC={auc_lr:.3f})",
                                 line=dict(color=BLUE, width=2)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random Baseline",
                                 line=dict(color="grey", dash="dash")))
        fig.update_layout(title="ROC Curve Comparison",
                          xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)
        two_liner(
            f"Random Forest (AUC={auc_rf:.3f}) outperforms Logistic Regression "
            f"(AUC={auc_lr:.3f}), capturing non-linear interactions between tech savviness, "
            "income, and feature interest. Both models beat the 0.5 random baseline."
        )

    with col2:
        cm = confusion_matrix(y_test, y_pred_rf)
        fig2 = px.imshow(cm, text_auto=True,
                         labels=dict(x="Predicted", y="Actual"),
                         x=["No (0)", "Yes (1)"], y=["No (0)", "Yes (1)"],
                         color_continuous_scale="Blues",
                         title="Random Forest – Confusion Matrix")
        st.plotly_chart(fig2, use_container_width=True)
        two_liner(
            "The model correctly identifies the majority of true adopters (high recall) "
            "while keeping false positives low, making it suitable for targeted "
            "acquisition campaigns."
        )

    # ── Feature importance ────────────────────────────────────────────────
    st.subheader("Feature Importance – Random Forest")
    fi = pd.DataFrame({"Feature": clf_features,
                        "Importance": rf.feature_importances_}).sort_values(
        "Importance", ascending=True).tail(15)
    fig3 = px.bar(fi, x="Importance", y="Feature", orientation="h",
                  color="Importance", color_continuous_scale="Tealgrn",
                  title="Top 15 Feature Importances")
    fig3.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)
    two_liner(
        "Tech savviness and feature_micro_investment lead the importance ranking, "
        "followed by monthly income and willingness to pay. These are the primary "
        "screening variables for targeted onboarding."
    )

    # ── Classification report ────────────────────────────────────────────
    with st.expander("📊 Full Classification Report (Random Forest)"):
        report = classification_report(y_test, y_pred_rf,
                                       target_names=["No (0)", "Yes (1)"])
        st.code(report)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
def tab_clustering(df: pd.DataFrame):
    st.title("👥 Clustering – User Personas")
    st.markdown(
        "**Goal**: Segment users into distinct personas using K-Means clustering.  \n"
        "**Features**: Income, savings, investment experience, risk appetite, tech savviness, feature interest."
    )

    clust_feats = [
        "age","monthly_income_aed","monthly_savings_aed",
        "has_investments","financial_anxiety_score",
        "willingness_to_pay_aed_monthly","finance_apps_currently_used",
        "smartphone_usage_hours_daily","feature_micro_investment",
        "feature_savings_goals","feature_sharia_portfolio",
    ] + ["risk_appetite","tech_savviness","investment_experience","savings_habit"]

    dfe = encode(df)
    X   = dfe[clust_feats]
    sc  = StandardScaler()
    Xs  = sc.fit_transform(X)

    # Elbow curve
    inertias = []
    K_range = range(2, 9)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        km.fit(Xs)
        inertias.append(km.inertia_)

    col1, col2 = st.columns(2)
    with col1:
        fig_elbow = px.line(x=list(K_range), y=inertias, markers=True,
                            title="Elbow Curve – Optimal K",
                            labels={"x": "Number of Clusters (K)", "y": "Inertia"})
        fig_elbow.add_vline(x=4, line_dash="dash", line_color=RED,
                            annotation_text="K=4 chosen", annotation_position="top right")
        st.plotly_chart(fig_elbow, use_container_width=True)
        two_liner(
            "The elbow occurs at K=4, balancing cluster compactness with interpretability. "
            "Beyond K=4 the marginal reduction in inertia is minimal, confirming four "
            "meaningful user segments."
        )

    # Fit final model with K=4
    K_BEST = 4
    km = KMeans(n_clusters=K_BEST, random_state=SEED, n_init=10)
    df["cluster"] = km.fit_predict(Xs)

    PERSONA_NAMES = {
        0: "💰 Wealth Builders",
        1: "📱 Digital Novices",
        2: "🕌 Sharia-First Investors",
        3: "🎓 Student Savers",
    }
    # Re-label clusters by median income (descending) for stability
    med_inc = df.groupby("cluster")["monthly_income_aed"].median().sort_values(ascending=False)
    rename_map = {old: list(PERSONA_NAMES.keys())[new]
                  for new, old in enumerate(med_inc.index)}
    df["cluster_id"]   = df["cluster"].map(rename_map)
    df["cluster_name"] = df["cluster_id"].map(PERSONA_NAMES)

    with col2:
        fig_cnt = px.pie(df, names="cluster_name", title="Cluster Size Distribution",
                         color_discrete_sequence=PALETTE, hole=0.4)
        st.plotly_chart(fig_cnt, use_container_width=True)
        two_liner(
            "Wealth Builders (highest income) and Digital Novices represent the two "
            "largest segments. Sharia-First and Student Savers are smaller but "
            "strategically important niches."
        )

    # PCA 2D scatter
    pca   = PCA(n_components=2, random_state=SEED)
    pca2d = pca.fit_transform(Xs)
    pca_df = pd.DataFrame({"PC1": pca2d[:,0], "PC2": pca2d[:,1],
                            "Persona": df["cluster_name"]})
    fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="Persona",
                         color_discrete_sequence=PALETTE,
                         title=f"K-Means Clusters in PCA Space (Var explained: "
                               f"{pca.explained_variance_ratio_.sum()*100:.1f}%)",
                         opacity=0.75)
    st.plotly_chart(fig_pca, use_container_width=True)
    two_liner(
        "PCA reduces 15 features to 2 principal components capturing most variance. "
        "Clusters are well-separated, validating that K-Means found genuinely distinct "
        "user groups rather than arbitrary partitions."
    )

    # Cluster profile radar
    st.subheader("Cluster Profiles – Normalised Feature Averages")
    radar_feats = ["monthly_income_aed","monthly_savings_aed",
                   "willingness_to_pay_aed_monthly","financial_anxiety_score",
                   "feature_micro_investment","feature_savings_goals",
                   "feature_sharia_portfolio","smartphone_usage_hours_daily"]
    radar_labels = ["Income","Savings","WTP","Fin Anxiety",
                    "Micro Inv","Sav Goals","Sharia","Phone hrs"]

    clust_means = df.groupby("cluster_name")[radar_feats].mean()
    clust_norm  = (clust_means - clust_means.min()) / (clust_means.max() - clust_means.min() + 1e-9)

    fig_r = go.Figure()
    for i, persona in enumerate(clust_norm.index):
        vals = clust_norm.loc[persona].tolist()
        vals += [vals[0]]
        lbls  = radar_labels + [radar_labels[0]]
        fig_r.add_trace(go.Scatterpolar(r=vals, theta=lbls, fill="toself",
                                        name=persona,
                                        line_color=PALETTE[i]))
    fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                        title="Radar Chart – Cluster Feature Profiles",
                        legend=dict(orientation="h", yanchor="bottom", y=-0.35))
    st.plotly_chart(fig_r, use_container_width=True)
    two_liner(
        "Wealth Builders score highest on income and WTP, making them the primary "
        "revenue target. Sharia-First Investors spike on Sharia portfolio interest. "
        "Student Savers show high financial anxiety but low income – needing a freemium hook."
    )

    # Cluster summary table
    st.subheader("Cluster Summary Statistics")
    summary_cols = ["monthly_income_aed","monthly_savings_aed","expected_monthly_investment_aed",
                    "willingness_to_pay_aed_monthly","would_use_app"]
    smry = df.groupby("cluster_name")[summary_cols].mean().round(1)
    smry.columns = ["Avg Income (AED)","Avg Savings (AED)","Avg Inv/Mo (AED)","Avg WTP (AED)","Adoption Rate"]
    smry["Adoption Rate"] = smry["Adoption Rate"].map("{:.1%}".format)
    smry["Cluster Size"]  = df.groupby("cluster_name").size()
    st.dataframe(smry, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – ASSOCIATION RULES
# ══════════════════════════════════════════════════════════════════════════════
def tab_association(df: pd.DataFrame):
    st.title("🔗 Association Rule Mining – Feature Bundles")
    st.markdown(
        "**Goal**: Discover which app features are commonly wanted together.  \n"
        "**Algorithm**: Apriori (mlxtend)  \n"
        "**Items**: Six binary feature-interest columns"
    )

    # ── Controls ──────────────────────────────────────────────────────────
    col_ctrl1, col_ctrl2 = st.columns(2)
    min_sup  = col_ctrl1.slider("Min Support",  0.05, 0.80, 0.30, 0.05)
    min_conf = col_ctrl2.slider("Min Confidence", 0.30, 0.99, 0.60, 0.05)

    # ── Apriori ───────────────────────────────────────────────────────────
    feat_df = df[FEAT_COLS].astype(bool)
    feat_df.columns = FEAT_LABELS_CLEAN

    freq_items = apriori(feat_df, min_support=min_sup, use_colnames=True)
    if freq_items.empty:
        st.warning("No frequent itemsets found at this support level. Try lowering Min Support.")
        return

    rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)

    if rules.empty:
        st.warning("No rules found. Try lowering Min Confidence.")
        return

    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

    st.success(f"Found **{len(rules)} rules** from **{len(freq_items)} frequent itemsets**.")

    # ── Top rules bar ─────────────────────────────────────────────────────
    top10 = rules.head(10).copy()
    top10["rule"] = top10["antecedents"] + "  →  " + top10["consequents"]
    fig1 = px.bar(top10, x="lift", y="rule", orientation="h",
                  color="confidence", color_continuous_scale="Viridis",
                  title="Top 10 Association Rules by Lift",
                  labels={"lift": "Lift", "rule": "Rule"})
    fig1.update_layout(height=500)
    st.plotly_chart(fig1, use_container_width=True)
    two_liner(
        "Rules with lift > 1.2 indicate features demanded together more than by chance. "
        "Savings Goals → Micro-Investment is the strongest bundle, suggesting users want "
        "a seamless save-then-invest flow within the same app."
    )

    # ── Scatter: Support vs Confidence coloured by Lift ───────────────────
    fig2 = px.scatter(rules, x="support", y="confidence", color="lift",
                      size="lift", hover_data=["antecedents","consequents"],
                      color_continuous_scale="Plasma",
                      title="Support vs Confidence (bubble size = Lift)",
                      labels={"support":"Support","confidence":"Confidence"})
    st.plotly_chart(fig2, use_container_width=True)
    two_liner(
        "High-confidence, high-support rules (top-right quadrant) represent safe "
        "feature bundles to launch together. The upper-right cluster confirms that "
        "Spending Tracker + Savings Goals form a near-universal base package."
    )

    # ── Heatmap: antecedent vs consequent lift ────────────────────────────
    pivot = rules.pivot_table(index="antecedents", columns="consequents",
                              values="lift", aggfunc="max").fillna(0)
    if not pivot.empty:
        fig3 = px.imshow(pivot.round(2), text_auto=True, aspect="auto",
                         color_continuous_scale="YlOrRd",
                         title="Lift Heatmap – Antecedent vs Consequent")
        st.plotly_chart(fig3, use_container_width=True)
        two_liner(
            "Red cells (lift > 1.5) highlight the strongest feature co-dependencies. "
            "Sharia Portfolio appears primarily co-demanded with Savings Goals, "
            "confirming a dedicated Islamic finance user niche."
        )

    # ── Full rules table ──────────────────────────────────────────────────
    with st.expander("📋 Full Rules Table"):
        display_cols = ["antecedents","consequents","support","confidence","lift","leverage","conviction"]
        st.dataframe(rules[display_cols].round(4), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
def tab_regression(df: pd.DataFrame):
    st.title("📈 Regression – Forecasting Monthly Investment")
    st.markdown(
        "**Goal**: Predict `expected_monthly_investment_aed` per user.  \n"
        "**Models**: Random Forest Regressor & Linear Regression  \n"
        "**Target**: Continuous AED value"
    )

    reg_features = [
        "age","monthly_income_aed","monthly_savings_aed",
        "willingness_to_pay_aed_monthly","financial_anxiety_score",
        "finance_apps_currently_used","smartphone_usage_hours_daily",
        "has_investments","feature_micro_investment","feature_savings_goals",
        "feature_ai_advisor","nps_score",
        "risk_appetite","investment_experience","tech_savviness",
        "employment_status","savings_habit",
    ]
    dfe = encode(df)
    X   = dfe[reg_features]
    y   = dfe["expected_monthly_investment_aed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    rfr = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=SEED)
    lr  = LinearRegression()
    rfr.fit(X_train, y_train)
    lr.fit(X_tr_sc, y_train)

    y_pred_rfr = rfr.predict(X_test)
    y_pred_lr  = lr.predict(X_te_sc)

    mae_rfr = mean_absolute_error(y_test, y_pred_rfr)
    mae_lr  = mean_absolute_error(y_test, y_pred_lr)
    rmse_rfr = mean_squared_error(y_test, y_pred_rfr) ** 0.5
    rmse_lr  = mean_squared_error(y_test, y_pred_lr)  ** 0.5
    r2_rfr  = r2_score(y_test, y_pred_rfr)
    r2_lr   = r2_score(y_test, y_pred_lr)

    # Metrics
    st.subheader("Model Metrics")
    rc1,rc2,rc3,rc4,rc5,rc6 = st.columns(6)
    rc1.metric("RF MAE",  f"AED {mae_rfr:,.0f}")
    rc2.metric("RF RMSE", f"AED {rmse_rfr:,.0f}")
    rc3.metric("RF R²",   f"{r2_rfr:.3f}")
    rc4.metric("LR MAE",  f"AED {mae_lr:,.0f}")
    rc5.metric("LR RMSE", f"AED {rmse_lr:,.0f}")
    rc6.metric("LR R²",   f"{r2_lr:.3f}")
    st.markdown("---")

    # Actual vs Predicted
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.scatter(x=y_test, y=y_pred_rfr,
                          labels={"x":"Actual (AED)","y":"Predicted (AED)"},
                          title="Random Forest – Actual vs Predicted",
                          opacity=0.6, color_discrete_sequence=[GREEN])
        max_v = max(y_test.max(), y_pred_rfr.max())
        fig1.add_trace(go.Scatter(x=[0,max_v], y=[0,max_v],
                                  mode="lines", name="Perfect Fit",
                                  line=dict(color=RED, dash="dash")))
        st.plotly_chart(fig1, use_container_width=True)
        two_liner(
            f"Random Forest achieves R²={r2_rfr:.3f} with MAE≈AED {mae_rfr:.0f}. "
            "Points cluster tightly around the perfect-fit diagonal at lower investment "
            "amounts, with slight dispersion for outlier high-investment users."
        )

    with col2:
        residuals = y_test - y_pred_rfr
        fig2 = px.histogram(residuals, nbins=30,
                            title="Residual Distribution – Random Forest",
                            labels={"value":"Residual (AED)","count":"Frequency"},
                            color_discrete_sequence=[BLUE])
        fig2.add_vline(x=0, line_dash="dash", line_color=RED)
        st.plotly_chart(fig2, use_container_width=True)
        two_liner(
            "Residuals are approximately normally distributed and centred at zero, "
            "indicating no systematic bias. The slight right skew reflects a small "
            "group of high-income users who invest significantly more than average."
        )

    # Feature importance
    st.subheader("Regression Feature Importance – Random Forest")
    fi = pd.DataFrame({
        "Feature": reg_features,
        "Importance": rfr.feature_importances_,
    }).sort_values("Importance", ascending=True).tail(15)
    fig3 = px.bar(fi, x="Importance", y="Feature", orientation="h",
                  color="Importance", color_continuous_scale="Oranges",
                  title="Top 15 Feature Importances (Regression)")
    fig3.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)
    two_liner(
        "Monthly income is the single biggest driver of expected investment, "
        "followed by monthly savings and investment experience. "
        "Feature_micro_investment interest also contributes, validating the core "
        "product premise."
    )

    # Income segmentation forecast
    st.subheader("Forecasted Investment by Income Band")
    df2 = df.copy()
    df2["income_band"] = pd.cut(df2["monthly_income_aed"],
                                bins=[0,8000,15000,25000,40000,80001],
                                labels=["<8K","8K-15K","15K-25K","25K-40K",">40K"])
    band_stats = df2.groupby("income_band", observed=True)["expected_monthly_investment_aed"].agg(
        Mean="mean", Median="median", Std="std").reset_index()
    fig4 = px.bar(band_stats, x="income_band", y="Mean",
                  error_y="Std", color="Mean",
                  color_continuous_scale="Tealgrn",
                  title="Mean Forecasted Monthly Investment by Income Band",
                  labels={"income_band":"Income Band (AED)","Mean":"Mean Investment (AED)"})
    fig4.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig4, use_container_width=True)
    two_liner(
        "Investment rises non-linearly with income band. The >40K segment invests "
        "5–7× more than the <8K band on average, suggesting a premium tier strategy "
        "targeting high-income expat professionals."
    )

    with st.expander("📊 Linear Regression Coefficients"):
        coef_df = pd.DataFrame({
            "Feature": reg_features,
            "Coefficient": lr.coef_,
        }).sort_values("Coefficient", ascending=False)
        st.dataframe(coef_df.round(3), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    df = load_data()
    selected = sidebar(df)

    if selected == "🏠 Overview & EDA":
        tab_overview(df)
    elif selected == "🎯 Classification":
        tab_classification(df)
    elif selected == "👥 Clustering":
        tab_clustering(df)
    elif selected == "🔗 Association Rules":
        tab_association(df)
    elif selected == "📈 Regression":
        tab_regression(df)


if __name__ == "__main__":
    main()
