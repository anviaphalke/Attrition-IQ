"""
Employee Attrition Prediction System
A comprehensive HR analytics platform built with Streamlit.
"""
# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import json
import logging
import io
import sys
from pathlib import Path
from datetime import datetime

def _generate_fallback_response(query: str, df: pd.DataFrame, trainer=None) -> str:
    """Rule-based fallback when API is unavailable."""
    q = query.lower()

    if "highest attrition" in q or "most attrition" in q:
        dept = df.groupby("Department")["Attrition"].apply(
            lambda x: (x == "Yes").mean() * 100
        ).idxmax()
        rate = df.groupby("Department")["Attrition"].apply(
            lambda x: (x == "Yes").mean() * 100
        ).max()
        return f"**{dept}** has the highest attrition rate at **{rate:.1f}%**. This department should be prioritized for retention initiatives."

    if "overtime" in q:
        ot_rate = df[df["OverTime"] == "Yes"]["Attrition"].eq("Yes").mean() * 100
        no_ot_rate = df[df["OverTime"] == "No"]["Attrition"].eq("Yes").mean() * 100
        return (f"Employees working overtime have a **{ot_rate:.1f}%** attrition rate vs "
                f"**{no_ot_rate:.1f}%** for those who don't — a {ot_rate/no_ot_rate:.1f}x higher risk. "
                "Consider reviewing workload distribution.")

    if "salary" in q or "income" in q:
        left_income = df[df["Attrition"] == "Yes"]["MonthlyIncome"].mean()
        stayed_income = df[df["Attrition"] == "No"]["MonthlyIncome"].mean()
        return (f"Employees who left earned on average **${left_income:,.0f}/month** vs "
                f"**${stayed_income:,.0f}/month** for those who stayed. "
                "Compensation appears to be a significant factor in attrition.")

    attrition_rate = (df["Attrition"] == "Yes").mean() * 100
    top_dept = df.groupby("Department")["Attrition"].apply(
        lambda x: (x == "Yes").mean() * 100
    ).idxmax()
    return (f"Based on our data ({len(df):,} employees, {attrition_rate:.1f}% attrition rate): "
            f"The highest-risk department is **{top_dept}**. "
            "Key risk factors include overtime, low job satisfaction, and years since last promotion.")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# pyrefly: ignore [missing-import]
from data_generator import generate_hr_dataset
# pyrefly: ignore [missing-import]
from preprocessor import HRDataPreprocessor
from models import AttritionModelTrainer, build_metrics_dataframe
# pyrefly: ignore [missing-import]
from explainer import AttritionExplainer
# pyrefly: ignore [missing-import]
from visualizations import (
    plot_attrition_overview, plot_department_attrition,
    plot_age_distribution, plot_salary_distribution,
    plot_satisfaction_heatmap, plot_overtime_impact,
    plot_tenure_attrition, plot_jobrole_attrition,
    plot_promotion_lag, plot_gender_attrition,
    plot_roc_curves, plot_confusion_matrix,
    plot_metrics_comparison, plot_feature_importance,
    plot_shap_summary, plot_shap_individual,
    plot_attrition_risk_gauge
)
# pyrefly: ignore [missing-import]
from report_generator import generate_attrition_summary_excel, generate_risk_report_csv

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AttritionIQ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Base ── */
  .stApp { background: #0F172A; color: #F1F5F9; }
  .main .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] { background: #1E293B; border-right: 1px solid rgba(255,255,255,0.06); }
  section[data-testid="stSidebar"] .stMarkdown { color: #94A3B8; }

  /* ── Metric Cards ── */
  div[data-testid="metric-container"] {
    background: #1E293B;
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 12px;
    padding: 1rem;
  }
  div[data-testid="metric-container"] label { color: #94A3B8 !important; font-size: 0.75rem !important; }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #F1F5F9 !important; font-size: 1.75rem !important; }

  /* ── Cards ── */
  .hr-card {
    background: #1E293B;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
  }
  .hr-card:hover {
    transform: translateY(-8px) scale(1.04);
    border-color: rgba(99,102,241,0.5);
    background: #24324D;
    box-shadow: 0 12px 24px -10px rgba(99,102,241,0.4), 0 4px 12px -5px rgba(99,102,241,0.2);
  }
  .hr-card:active {
    transform: translateY(-4px) scale(1.02);
    border-color: #6366F1;
    box-shadow: 0 8px 16px -8px rgba(99,102,241,0.6);
  }
  .risk-badge-high { background: rgba(239,68,68,0.15); color: #FCA5A5; border: 1px solid rgba(239,68,68,0.3); border-radius: 8px; padding: 0.35rem 0.75rem; font-size: 0.8rem; font-weight: 600; }
  .risk-badge-medium { background: rgba(245,158,11,0.15); color: #FCD34D; border: 1px solid rgba(245,158,11,0.3); border-radius: 8px; padding: 0.35rem 0.75rem; font-size: 0.8rem; font-weight: 600; }
  .risk-badge-low { background: rgba(16,185,129,0.15); color: #6EE7B7; border: 1px solid rgba(16,185,129,0.3); border-radius: 8px; padding: 0.35rem 0.75rem; font-size: 0.8rem; font-weight: 600; }

  /* ── Section Headers ── */
  .section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #F1F5F9;
    margin-bottom: 0.25rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .section-sub { font-size: 0.85rem; color: #64748B; margin-bottom: 1.5rem; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] { background: #1E293B; border-radius: 12px; padding: 0.25rem; gap: 4px; }
  .stTabs [data-baseweb="tab"] { background: transparent; color: #94A3B8; border-radius: 8px; font-size: 0.85rem; }
  .stTabs [aria-selected="true"] { background: #6366F1 !important; color: white !important; }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    color: white; border: none; border-radius: 10px;
    font-weight: 600; padding: 0.5rem 1.5rem;
    transition: all 0.2s;
  }
  .stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }

  /* ── Chat ── */
  .chat-msg-user { background: rgba(99,102,241,0.15); border-left: 3px solid #6366F1; padding: 0.75rem 1rem; border-radius: 0 10px 10px 0; margin: 0.5rem 0; }
  .chat-msg-assistant { background: rgba(30,41,59,0.8); border-left: 3px solid #10B981; padding: 0.75rem 1rem; border-radius: 0 10px 10px 0; margin: 0.5rem 0; }

  /* ── Data Table ── */
  .stDataFrame { border-radius: 10px; overflow: hidden; }

  /* ── Expander ── */
  .streamlit-expanderHeader { background: #1E293B; border-radius: 10px; }

  /* ── Hero ── */
  .hero-banner {
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 50%, rgba(99,102,241,0.1) 100%);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
  }
  .hero-title { font-size: 2rem; font-weight: 800; background: linear-gradient(135deg, #F1F5F9, #6366F1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .hero-sub { color: #64748B; font-size: 0.95rem; margin-top: 0.4rem; }

  /* ── Best model badge ── */
  .best-badge { background: linear-gradient(135deg, #6366F1, #8B5CF6); color: white; border-radius: 6px; padding: 0.2rem 0.6rem; font-size: 0.75rem; font-weight: 700; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #0F172A; }
  ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "df_raw": None,
        "df_clean": None,
        "X_processed": None,
        "y": None,
        "preprocessor": None,
        "trainer": None,
        "explainer": None,
        "metrics_df": None,
        "chat_history": [],
        "models_trained": False,
        "data_loaded": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
      <div style='font-size:2.5rem;'>🧠</div>
      <div style='font-weight:800; font-size:1.1rem; color:#F1F5F9;'>AttritionIQ</div>
      <div style='font-size:0.75rem; color:#64748B;'>Best-Fit Model Selection • SHAP Explainability</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📂 Data Source")

    data_source = st.radio(
        "Choose data source",
        ["🎲 Use Sample Dataset", "📤 Upload CSV/Excel"],
        label_visibility="collapsed"
    )

    if data_source == "📤 Upload CSV/Excel":
        uploaded_file = st.file_uploader(
            "Upload HR Data",
            type=["csv", "xlsx"],
            help="Upload employee data with attrition labels"
        )
        st.markdown(
            "<div style='font-size:0.8rem; color:#94A3B8; margin-top:-0.5rem; margin-bottom:0.5rem;'>"
            "Use the template below to ensure correct columns and format:"
            "</div>",
            unsafe_allow_html=True
        )
        try:
            with open("data/sample_hr_data.csv", "r", encoding="utf-8") as f:
                sample_csv = f.read()
            st.download_button(
                label="📥 Download CSV Template",
                data=sample_csv,
                file_name="sample_hr_data.csv",
                mime="text/csv",
                width="stretch"
            )
        except Exception as e:
            logger.error(f"Error loading sample CSV template: {e}")
    else:
        uploaded_file = None
        n_employees = st.slider("Sample size", 500, 3000, 1470, 100)

    if st.button("⚡ Load & Analyse", use_container_width=True):
        with st.spinner("Loading and preprocessing data..."):
            try:
                if uploaded_file:
                    if uploaded_file.name.endswith(".csv"):
                        df_raw = pd.read_csv(uploaded_file)
                    else:
                        df_raw = pd.read_excel(uploaded_file)
                else:
                    df_raw = generate_hr_dataset(n_employees)

                preprocessor = HRDataPreprocessor()
                X, y, df_clean = preprocessor.full_pipeline(df_raw, fit=True)

                st.session_state.df_raw = df_raw
                st.session_state.df_clean = df_clean
                st.session_state.X_processed = X
                st.session_state.y = y
                st.session_state.preprocessor = preprocessor
                st.session_state.data_loaded = True
                st.session_state.models_trained = False
                st.session_state.chat_history = []

                st.success(f"✅ {len(df_raw):,} employees loaded")
                logger.info(f"Data loaded: {df_raw.shape}")
            except Exception as e:
                st.error(f"❌ Error: please upload a file {e}")
                logger.error(f"Data loading failed: {e}", exc_info=True)

    if st.session_state.data_loaded:
        st.divider()
        st.markdown("### 🤖 Train Models")
        if st.button("🚀 Train All Models", use_container_width=True):
            with st.spinner("Training Logistic Regression, Random Forest & XGBoost..."):
                try:
                    from sklearn.model_selection import train_test_split

                    X = st.session_state.X_processed
                    y = st.session_state.y

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )

                    trainer = AttritionModelTrainer()
                    results = trainer.train_all(X_train, y_train, X_test, y_test)
                    metrics_df = build_metrics_dataframe(results)

                    # SHAP explainer
                    best_model = trainer.get_best_model()
                    explainer = AttritionExplainer(best_model, X.columns.tolist())
                    explainer.fit(X_train)

                    st.session_state.trainer = trainer
                    st.session_state.metrics_df = metrics_df
                    st.session_state.explainer = explainer
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.models_trained = True

                    st.success(f"✅ Best: **{trainer.best_model_name}**")
                    logger.info("Models trained successfully")
                except Exception as e:
                    st.error(f"❌ Training failed: try again later {e}")
                    logger.error(f"Training failed: try again later {e}", exc_info=True)

    st.divider()
    # Status
    if st.session_state.data_loaded:
        df_c = st.session_state.df_clean
        attrition_rate = (df_c["Attrition"] == "Yes").mean() * 100
        st.markdown(f"""
        <div style='background:#1E293B;border-radius:10px;padding:0.75rem 1rem;font-size:0.8rem;color:#94A3B8;'>
          📊 <b style='color:#F1F5F9;'>{len(df_c):,}</b> employees<br>
          ⚠️ Attrition: <b style='color:#EF4444;'>{attrition_rate:.1f}%</b><br>
          🏢 Departments: <b style='color:#F1F5F9;'>{df_c['Department'].nunique()}</b><br>
          💼 Job Roles: <b style='color:#F1F5F9;'>{df_c['JobRole'].nunique()}</b>
        </div>
        """, unsafe_allow_html=True)

# ── Main Content ───────────────────────────────────────────────────────────────

# Hero Banner
st.markdown("""
<div class='hero-banner'>
  <div class='hero-title'>🧠 Employee Attrition Prediction System</div>
  <div class='hero-sub'>
    AI-powered HR analytics — predict who's at risk, understand why, and act before they leave.
  </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.data_loaded:
    # Landing state
    cols = st.columns(3)
    cards = [
        ("📊", "EDA Dashboard", "Explore demographics, salary distributions, department trends, and satisfaction scores"),
        ("🤖", "ML Models", "Compare Logistic Regression, Random Forest & XGBoost with full metrics and ROC curves"),
        ("💡", "SHAP Explainability", "Understand exactly WHY each employee is at risk with individual SHAP explanations"),
    ]
    for col, (icon, title, desc) in zip(cols, cards):
        with col:
            st.markdown(f"""
            <div class='hr-card' style='text-align:center;'>
              <div style='font-size:2.5rem;'>{icon}</div>
              <div style='font-weight:700;color:#F1F5F9;margin:0.5rem 0;'>{title}</div>
              <div style='color:#64748B;font-size:0.85rem;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    cols2 = st.columns(3)
    cards2 = [
        ("🔮", "Individual Predictions", "Input any employee's data and get instant attrition risk with actionable HR recommendations"),
        ("📥", "Downloadable Reports", "Export full Excel reports with department analysis, risk lists, and model performance"),
    ]
    for col, (icon, title, desc) in zip(cols2, cards2):
        with col:
            st.markdown(f"""
            <div class='hr-card' style='text-align:center;'>
              <div style='font-size:2.5rem;'>{icon}</div>
              <div style='font-weight:700;color:#F1F5F9;margin:0.5rem 0;'>{title}</div>
              <div style='color:#64748B;font-size:0.85rem;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center; margin-top:2rem; padding:2rem; background:#1E293B; border-radius:16px;'>
      <div style='font-size:1.1rem; color:#94A3B8;'>
        👈 Select a data source from the sidebar and click <b style='color:#6366F1;'>Load & Analyse</b> to get started
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Main Tabs ─────────────────────────────────────────────────────────────────
df = st.session_state.df_clean

tab_eda, tab_models, tab_predict, tab_reports = st.tabs([
    "📊 EDA Dashboard",
    "🤖 Model Performance",
    "🔮 Predict & Explain",
    "📥 Reports"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown("<div class='section-header'>📊 Exploratory Data Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Understand workforce composition, attrition patterns, and key HR metrics</div>", unsafe_allow_html=True)

    # KPI Row
    total_emp = len(df)
    attrition_count = (df["Attrition"] == "Yes").sum()
    attrition_rate = attrition_count / total_emp * 100
    avg_income = df["MonthlyIncome"].mean() if "MonthlyIncome" in df.columns else 0
    avg_tenure = df["YearsAtCompany"].mean() if "YearsAtCompany" in df.columns else 0
    ot_rate = (df["OverTime"] == "Yes").mean() * 100 if "OverTime" in df.columns else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("👥 Total Employees", f"{total_emp:,}")
    k2.metric("⚠️ Attrition Count", f"{attrition_count:,}", f"{attrition_rate:.1f}%")
    k3.metric("💰 Avg Monthly Income", f"${avg_income:,.0f}")
    k4.metric("📅 Avg Tenure", f"{avg_tenure:.1f} yrs")
    k5.metric("🕐 Overtime Rate", f"{ot_rate:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1
    c1, c2 = st.columns([1, 2])
    with c1:
        st.plotly_chart(plot_attrition_overview(df), use_container_width=True)
    with c2:
        st.plotly_chart(plot_department_attrition(df), use_container_width=True)

    # Row 2
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(plot_age_distribution(df), use_container_width=True)
    with c4:
        st.plotly_chart(plot_salary_distribution(df), use_container_width=True)

    # Row 3
    st.plotly_chart(plot_satisfaction_heatmap(df), use_container_width=True)

    # Row 4
    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(plot_overtime_impact(df), use_container_width=True)
    with c6:
        st.plotly_chart(plot_tenure_attrition(df), use_container_width=True)

    # Row 5
    c7, c8 = st.columns(2)
    with c7:
        st.plotly_chart(plot_jobrole_attrition(df), use_container_width=True)
    with c8:
        st.plotly_chart(plot_promotion_lag(df), use_container_width=True)

    st.plotly_chart(plot_gender_attrition(df), use_container_width=True)

    # Raw data preview
    with st.expander("🔍 View Raw Data"):
        st.dataframe(df.head(100), use_container_width=True)
        st.caption(f"Showing first 100 of {len(df):,} rows")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Performance
# ═══════════════════════════════════════════════════════════════════════════════
with tab_models:
    st.markdown("<div class='section-header'>🤖 Model Training & Evaluation</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Compare Logistic Regression, Random Forest, and XGBoost across all key metrics</div>", unsafe_allow_html=True)

    if not st.session_state.models_trained:
        st.info("👈 Click **Train All Models** in the sidebar to begin model training.")
        st.stop()

    trainer = st.session_state.trainer
    metrics_df = st.session_state.metrics_df
    results = trainer.evaluation_results

    # Best model badge
    best = trainer.best_model_name
    st.markdown(f"<div style='margin-bottom:1rem;'>Best Model: <span class='best-badge'>🏆 {best}</span></div>", unsafe_allow_html=True)

    # Metrics table
    st.markdown("#### 📋 Performance Metrics")
    st.dataframe(
        metrics_df.style.highlight_max(
            subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            color="rgba(99,102,241,0.3)"
        ),
        use_container_width=True
    )

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_roc_curves(results), use_container_width=True)
    with c2:
        st.plotly_chart(plot_metrics_comparison(metrics_df), use_container_width=True)

    # Confusion matrices
    st.markdown("#### 🔲 Confusion Matrices")
    cm_cols = st.columns(len(results))
    for col, (name, metrics) in zip(cm_cols, results.items()):
        with col:
            st.plotly_chart(
                plot_confusion_matrix(metrics["confusion_matrix"], name),
                use_container_width=True
            )

    # Feature importance
    st.markdown("#### 🎯 Feature Importance")
    feat_imp = trainer.get_feature_importances(
        st.session_state.X_processed.columns.tolist()
    )
    if not feat_imp.empty:
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(plot_feature_importance(feat_imp), use_container_width=True)
        with c4:
            # SHAP global
            explainer = st.session_state.explainer
            X_test = st.session_state.X_test
            with st.spinner("Computing SHAP values..."):
                try:
                    shap_vals, shap_imp = explainer.explain_global(X_test)
                    st.plotly_chart(
                        plot_shap_summary(shap_vals, explainer.feature_names),
                        use_container_width=True
                    )
                except Exception as e:
                    st.warning(f"SHAP global explanation unavailable: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Predict & Explain
# ═══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("<div class='section-header'>🔮 Individual Attrition Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Enter employee details for instant risk assessment with SHAP-powered explanations</div>", unsafe_allow_html=True)

    if not st.session_state.models_trained:
        st.info("👈 Train models first using the sidebar button.")
        st.stop()

    # Employee selector OR manual input
    predict_mode = st.radio(
        "Input Method",
        ["📋 Select Existing Employee", "✏️ Enter Custom Data"],
        horizontal=True
    )

    if predict_mode == "📋 Select Existing Employee":
        df_raw = st.session_state.get("df_raw")
        preprocessor = st.session_state.get("preprocessor")
        emp_id_col = None

        if preprocessor and hasattr(preprocessor, "employee_id_column") and preprocessor.employee_id_column:
            emp_id_col = preprocessor.employee_id_column

        if df_raw is not None:
            if not emp_id_col or emp_id_col not in df_raw.columns:
                # pyrefly: ignore [missing-import]
                from preprocessor import HRDataPreprocessor
                emp_id_col = HRDataPreprocessor.find_employee_id_column(df_raw)

            if emp_id_col and emp_id_col in df_raw.columns:
                id_list = df_raw[emp_id_col].tolist()
                selected_id = st.selectbox(
                    "Choose Employee ID",
                    id_list,
                    index=0
                )
                matching_rows = df_raw[df_raw[emp_id_col] == selected_id]
                if not matching_rows.empty:
                    idx = matching_rows.index[0]
                    employee_data = df.loc[idx].to_dict() if idx in df.index else df.iloc[0].to_dict()
                else:
                    employee_data = df.iloc[0].to_dict()
            else:
                emp_id_col = df.columns[0]
                selected_id = st.selectbox(
                    "Choose Employee ID",
                    df[emp_id_col].tolist(),
                    index=0
                )
                employee_data = df[df[emp_id_col] == selected_id].iloc[0].to_dict()
        else:
            emp_id_col = df.columns[0]
            selected_id = st.selectbox(
                "Choose Employee ID",
                df[emp_id_col].tolist(),
                index=0
            )
            employee_data = df[df[emp_id_col] == selected_id].iloc[0].to_dict()
    else:
        employee_data = {}

    with st.form("predict_form"):
        st.markdown("#### 👤 Employee Profile")
        c1, c2, c3, c4 = st.columns(4)

        age = c1.number_input("Age", 18, 65,
            int(employee_data.get("Age", 35)))
        department = c2.selectbox("Department",
            df["Department"].unique().tolist(),
            index=list(df["Department"].unique()).index(
                employee_data.get("Department", df["Department"].iloc[0])) if "Department" in employee_data else 0)
        job_role_options = df["JobRole"].unique().tolist()
        job_role = c3.selectbox("Job Role",
            job_role_options,
            index=job_role_options.index(employee_data.get("JobRole", job_role_options[0])) if "JobRole" in employee_data else 0)
        monthly_income = c4.number_input("Monthly Income ($)", 1000, 25000,
            int(employee_data.get("MonthlyIncome", 5000)), step=100)

        c5, c6, c7, c8 = st.columns(4)
        overtime = c5.selectbox("Overtime", ["Yes", "No"],
            index=0 if employee_data.get("OverTime", "No") == "Yes" else 1)
        years_at_company = c6.number_input("Years at Company", 0, 40,
            int(employee_data.get("YearsAtCompany", 5)))
        years_since_promo = c7.number_input("Years Since Promotion", 0, 15,
            int(employee_data.get("YearsSinceLastPromotion", 2)))
        job_satisfaction = c8.slider("Job Satisfaction (1-4)", 1, 4,
            int(employee_data.get("JobSatisfaction", 3)))

        c9, c10, c11, c12 = st.columns(4)
        marital_status = c9.selectbox("Marital Status", ["Single", "Married", "Divorced"],
            index=["Single", "Married", "Divorced"].index(
                employee_data.get("MaritalStatus", "Married")))
        work_life_balance = c10.slider("Work-Life Balance (1-4)", 1, 4,
            int(employee_data.get("WorkLifeBalance", 3)))
        env_satisfaction = c11.slider("Environment Satisfaction (1-4)", 1, 4,
            int(employee_data.get("EnvironmentSatisfaction", 3)))
        distance_from_home = c12.number_input("Distance From Home (km)", 1, 30,
            int(employee_data.get("DistanceFromHome", 10)))

        c13, c14, c15, c16 = st.columns(4)
        num_companies = c13.number_input("Num Companies Worked", 0, 10,
            int(employee_data.get("NumCompaniesWorked", 2)))
        stock_option = c14.selectbox("Stock Option Level", [0, 1, 2, 3],
            index=int(employee_data.get("StockOptionLevel", 0)))
        business_travel = c15.selectbox(
            "Business Travel",
            ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
            index=["Non-Travel", "Travel_Rarely", "Travel_Frequently"].index(
                employee_data.get("BusinessTravel", "Travel_Rarely"))
        )
        total_working_years = c16.number_input("Total Working Years", 0, 40,
            int(employee_data.get("TotalWorkingYears", 8)))

        submitted = st.form_submit_button("🔮 Predict Attrition Risk", use_container_width=True)

    if submitted:
        # Build employee record
        emp_record = {
            "Age": age, "Department": department, "JobRole": job_role,
            "MonthlyIncome": monthly_income, "OverTime": overtime,
            "YearsAtCompany": years_at_company,
            "YearsSinceLastPromotion": years_since_promo,
            "JobSatisfaction": job_satisfaction,
            "MaritalStatus": marital_status,
            "WorkLifeBalance": work_life_balance,
            "EnvironmentSatisfaction": env_satisfaction,
            "DistanceFromHome": distance_from_home,
            "NumCompaniesWorked": num_companies,
            "StockOptionLevel": stock_option,
            "BusinessTravel": business_travel,
            "TotalWorkingYears": total_working_years,
            # Defaults for remaining columns
            "Gender": employee_data.get("Gender", "Male"),
            "Education": int(employee_data.get("Education", 3)),
            "EducationField": employee_data.get("EducationField", "Life Sciences"),
            "JobLevel": int(employee_data.get("JobLevel", 2)),
            "JobInvolvement": int(employee_data.get("JobInvolvement", 3)),
            "MonthlyRate": int(employee_data.get("MonthlyRate", 14000)),
            "PercentSalaryHike": int(employee_data.get("PercentSalaryHike", 14)),
            "PerformanceRating": int(employee_data.get("PerformanceRating", 3)),
            "RelationshipSatisfaction": int(employee_data.get("RelationshipSatisfaction", 3)),
            "TrainingTimesLastYear": int(employee_data.get("TrainingTimesLastYear", 3)),
            "YearsInCurrentRole": int(employee_data.get("YearsInCurrentRole", 3)),
            "YearsWithCurrManager": int(employee_data.get("YearsWithCurrManager", 4)),
            "Attrition": "No"  # Placeholder
        }

        try:
            preprocessor = st.session_state.preprocessor
            trainer = st.session_state.trainer
            explainer = st.session_state.explainer

            emp_df = pd.DataFrame([emp_record])
            emp_df_engineered = preprocessor.engineer_features(emp_df)
            X_emp, _ = preprocessor.encode_and_scale(emp_df_engineered, fit=False)

            pred, prob = trainer.predict_employee(X_emp)

            # Risk badge
            risk_class = (
                "risk-badge-high" if prob > 0.6
                else "risk-badge-medium" if prob > 0.3
                else "risk-badge-low"
            )
            risk_label = (
                "🚨 HIGH RISK" if prob > 0.6
                else "⚠️ MEDIUM RISK" if prob > 0.3
                else "✅ LOW RISK"
            )

            # Results layout
            r1, r2 = st.columns([1, 2])
            with r1:
                st.plotly_chart(plot_attrition_risk_gauge(prob), use_container_width=True)
                st.markdown(
                    f"<div style='text-align:center;margin-top:0.5rem;'>"
                    f"<span class='{risk_class}'>{risk_label}</span></div>",
                    unsafe_allow_html=True
                )

            with r2:
                # SHAP individual
                try:
                    shap_df = explainer.explain_individual(X_emp)
                    st.plotly_chart(
                        plot_shap_individual(shap_df),
                        use_container_width=True
                    )
                except Exception as e:
                    st.warning(f"SHAP explanation unavailable: {e}")

            # Text explanation
            try:
                explanation = explainer.generate_explanation_text(
                    X_emp, pred, prob, emp_record
                )
                st.markdown(
                    f"<div class='hr-card'>{explanation}</div>",
                    unsafe_allow_html=True
                )
            except Exception as e:
                logger.warning(f"Explanation text failed: {e}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            logger.error(f"Prediction error: {e}", exc_info=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Reports
# ═══════════════════════════════════════════════════════════════════════════════
with tab_reports:
    st.markdown("<div class='section-header'>📥 Downloadable HR Reports</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Export analysis results, risk lists, and model performance to Excel or CSV</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class='hr-card'>
          <div style='font-size:1.1rem;font-weight:700;color:#F1F5F9;margin-bottom:0.75rem;'>
            📊 Full Attrition Analysis Report
          </div>
          <div style='color:#94A3B8;font-size:0.85rem;'>
            Multi-sheet Excel report including:<br>
            • Executive Summary KPIs<br>
            • Department & Role Analysis<br>
            • High-Risk Employee List<br>
            • Model Performance Metrics<br>
            • Feature Importance Rankings
          </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.models_trained:
            trainer = st.session_state.trainer
            feat_imp = trainer.get_feature_importances(
                st.session_state.X_processed.columns.tolist()
            )
            report_bytes = generate_attrition_summary_excel(
                df, st.session_state.metrics_df, feat_imp
            )
            st.download_button(
                label="⬇️ Download Excel Report",
                data=report_bytes,
                file_name=f"hr_attrition_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        else:
            st.download_button(
                label="⬇️ Download Excel Report",
                data=generate_attrition_summary_excel(
                    df,
                    pd.DataFrame(columns=["Model", "Accuracy", "F1-Score", "ROC-AUC"])
                ),
                file_name=f"hr_attrition_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    with c2:
        st.markdown("""
        <div class='hr-card'>
          <div style='font-size:1.1rem;font-weight:700;color:#F1F5F9;margin-bottom:0.75rem;'>
            🚨 High-Risk Employees CSV
          </div>
          <div style='color:#94A3B8;font-size:0.85rem;'>
            Quick-reference CSV listing employees who left,<br>
            suitable for immediate HR action review.<br><br>
            Includes: Employee ID, Department, Job Role,<br>
            Age, Income, Tenure, Overtime status.
          </div>
        </div>
        """, unsafe_allow_html=True)

        risk_csv = generate_risk_report_csv(df)
        st.download_button(
            label="⬇️ Download Risk CSV",
            data=risk_csv,
            file_name=f"high_risk_employees_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Quick stats
    st.markdown("#### 📈 Quick Summary Statistics")
    summary_cols = st.columns(3)

    with summary_cols[0]:
        dept_df = df.groupby("Department").apply(
            lambda x: pd.Series({
                "Total": len(x),
                "Attrition Count": (x["Attrition"] == "Yes").sum(),
                "Attrition Rate %": round((x["Attrition"] == "Yes").mean() * 100, 1)
            })
        ).reset_index()
        st.markdown("**Department Summary**")
        st.dataframe(dept_df, use_container_width=True, hide_index=True)

    with summary_cols[1]:
        role_df = df.groupby("JobRole").apply(
            lambda x: pd.Series({
                "Count": len(x),
                "Attrition %": round((x["Attrition"] == "Yes").mean() * 100, 1)
            })
        ).sort_values("Attrition %", ascending=False).reset_index()
        st.markdown("**Job Role Summary**")
        st.dataframe(role_df, use_container_width=True, hide_index=True)

    with summary_cols[2]:
        if st.session_state.models_trained:
            st.markdown("**Model Performance**")
            st.dataframe(
                st.session_state.metrics_df[["Model", "ROC-AUC", "F1-Score", "Accuracy"]],
                use_container_width=True,
                hide_index=True
            )

