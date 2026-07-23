"""
Visualizations Module
All Plotly charts for EDA, model evaluation, and SHAP explanations.
"""

import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import plotly.express as px
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go
# pyrefly: ignore [missing-import]
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# ── Design Tokens ─────────────────────────────────────────────────────────────
COLORS = {
    "primary": "#6366F1",       # Indigo
    "secondary": "#EC4899",     # Pink
    "success": "#10B981",       # Emerald
    "danger": "#EF4444",        # Red
    "warning": "#F59E0B",       # Amber
    "info": "#3B82F6",          # Blue
    "bg": "#0F172A",            # Slate-900
    "surface": "#1E293B",       # Slate-800
    "text": "#F1F5F9",          # Slate-100
    "muted": "#94A3B8",         # Slate-400
    "attrition_yes": "#EF4444",
    "attrition_no": "#10B981",
}

CHART_PALETTE = [
    "#6366F1", "#EC4899", "#10B981", "#F59E0B",
    "#3B82F6", "#8B5CF6", "#06B6D4", "#F97316"
]

LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=COLORS["text"], size=12),
    margin=dict(t=50, b=40, l=50, r=20),
    legend=dict(
        bgcolor="rgba(30,41,59,0.8)",
        bordercolor="rgba(255,255,255,0.1)",
        borderwidth=1
    ),
)


def _apply_base(fig) -> go.Figure:
    fig.update_layout(**LAYOUT_BASE)
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.1)", linecolor="rgba(148,163,184,0.2)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.1)", linecolor="rgba(148,163,184,0.2)")
    return fig


# ── EDA Charts ────────────────────────────────────────────────────────────────

def plot_attrition_overview(df: pd.DataFrame) -> go.Figure:
    """Donut chart of overall attrition."""
    counts = df["Attrition"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.65,
        marker_colors=[COLORS["attrition_no"], COLORS["attrition_yes"]],
        textinfo="label+percent",
        textfont_size=14,
        pull=[0.02, 0.05]
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Overall Attrition Rate", font_size=16, x=0.5),
        showlegend=True
    )
    return fig


def plot_department_attrition(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: attrition count by department."""
    grp = (
        df.groupby(["Department", "Attrition"])
        .size()
        .reset_index(name="Count")
    )
    fig = px.bar(
        grp, x="Department", y="Count", color="Attrition",
        barmode="group",
        color_discrete_map={"Yes": COLORS["attrition_yes"], "No": COLORS["attrition_no"]},
        title="Attrition by Department"
    )
    return _apply_base(fig)


def plot_age_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of age split by attrition."""
    fig = px.histogram(
        df, x="Age", color="Attrition", nbins=25,
        barmode="overlay",
        color_discrete_map={"Yes": COLORS["attrition_yes"], "No": COLORS["attrition_no"]},
        title="Age Distribution by Attrition",
        opacity=0.75
    )
    return _apply_base(fig)


def plot_salary_distribution(df: pd.DataFrame) -> go.Figure:
    """Box plot of monthly income by department, colored by attrition."""
    fig = px.box(
        df, x="Department", y="MonthlyIncome", color="Attrition",
        color_discrete_map={"Yes": COLORS["attrition_yes"], "No": COLORS["attrition_no"]},
        title="Salary Distribution by Department & Attrition",
        points="outliers"
    )
    return _apply_base(fig)


def plot_satisfaction_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of attrition rate across satisfaction dimensions."""
    sat_cols = ["JobSatisfaction", "EnvironmentSatisfaction",
                "WorkLifeBalance", "RelationshipSatisfaction"]
    sat_cols = [c for c in sat_cols if c in df.columns]

    matrix = {}
    for col in sat_cols:
        rate = df.groupby(col)["Attrition"].apply(
            lambda x: (x == "Yes").mean() * 100
        )
        matrix[col] = rate

    heat_df = pd.DataFrame(matrix).T
    fig = go.Figure(go.Heatmap(
        z=heat_df.values,
        x=[str(c) for c in heat_df.columns],
        y=heat_df.index.tolist(),
        colorscale=[[0, COLORS["success"]], [0.5, COLORS["warning"]], [1, COLORS["danger"]]],
        text=np.round(heat_df.values, 1),
        texttemplate="%{text}%",
        textfont=dict(size=12),
        colorbar=dict(title="Attrition %")
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Attrition Rate by Satisfaction Scores", font_size=16, x=0.5),
        xaxis_title="Score (1=Low, 4=High)",
        yaxis_title=""
    )
    return fig


def plot_overtime_impact(df: pd.DataFrame) -> go.Figure:
    """Stacked bar: overtime vs attrition."""
    if "OverTime" not in df.columns:
        return go.Figure()
    grp = (
        df.groupby(["OverTime", "Attrition"])
        .size()
        .reset_index(name="Count")
    )
    fig = px.bar(
        grp, x="OverTime", y="Count", color="Attrition",
        barmode="stack",
        color_discrete_map={"Yes": COLORS["attrition_yes"], "No": COLORS["attrition_no"]},
        title="Overtime Impact on Attrition"
    )
    return _apply_base(fig)


def plot_tenure_attrition(df: pd.DataFrame) -> go.Figure:
    """Line chart: attrition rate by years at company."""
    if "YearsAtCompany" not in df.columns:
        return go.Figure()
    grp = df.groupby("YearsAtCompany").apply(
        lambda x: (x["Attrition"] == "Yes").mean() * 100
    ).reset_index()
    grp.columns = ["YearsAtCompany", "AttritionRate"]
    grp = grp[grp["YearsAtCompany"] <= 25]

    fig = go.Figure(go.Scatter(
        x=grp["YearsAtCompany"], y=grp["AttritionRate"],
        mode="lines+markers",
        line=dict(color=COLORS["primary"], width=2.5),
        marker=dict(size=7, color=COLORS["secondary"]),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.15)"
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title="Attrition Rate by Tenure",
        xaxis_title="Years at Company",
        yaxis_title="Attrition Rate (%)"
    )
    return fig


def plot_jobrole_attrition(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar: attrition rate per job role."""
    rate = (
        df.groupby("JobRole")["Attrition"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .sort_values(ascending=True)
        .reset_index()
    )
    rate.columns = ["JobRole", "AttritionRate"]
    fig = px.bar(
        rate, x="AttritionRate", y="JobRole",
        orientation="h",
        color="AttritionRate",
        color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
        title="Attrition Rate by Job Role"
    )
    fig.update_layout(**LAYOUT_BASE, coloraxis_showscale=False)
    return fig


def plot_promotion_lag(df: pd.DataFrame) -> go.Figure:
    """Violin of years since last promotion by attrition."""
    if "YearsSinceLastPromotion" not in df.columns:
        return go.Figure()
    fig = px.violin(
        df, x="Attrition", y="YearsSinceLastPromotion",
        color="Attrition",
        box=True,
        color_discrete_map={"Yes": COLORS["attrition_yes"], "No": COLORS["attrition_no"]},
        title="Years Since Last Promotion vs Attrition"
    )
    return _apply_base(fig)


def plot_gender_attrition(df: pd.DataFrame) -> go.Figure:
    """Pie charts by gender."""
    if "Gender" not in df.columns:
        return go.Figure()
    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]],
        subplot_titles=["Male", "Female"]
    )
    for i, gender in enumerate(["Male", "Female"], 1):
        sub = df[df["Gender"] == gender]["Attrition"].value_counts()
        fig.add_trace(go.Pie(
            labels=sub.index, values=sub.values,
            name=gender, hole=0.5,
            marker_colors=[COLORS["attrition_no"], COLORS["attrition_yes"]]
        ), row=1, col=i)
    fig.update_layout(**LAYOUT_BASE, title_text="Attrition by Gender")
    return fig


# ── Model Evaluation Charts ───────────────────────────────────────────────────

def plot_roc_curves(results: Dict) -> go.Figure:
    """Overlaid ROC curves for all models."""
    fig = go.Figure()
    colors_list = [COLORS["primary"], COLORS["secondary"], COLORS["success"]]

    for (name, metrics), color in zip(results.items(), colors_list):
        fig.add_trace(go.Scatter(
            x=metrics["roc_fpr"], y=metrics["roc_tpr"],
            name=f"{name} (AUC={metrics['roc_auc']:.3f})",
            mode="lines",
            line=dict(width=2.5, color=color)
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], name="Random Classifier",
        line=dict(dash="dash", color=COLORS["muted"], width=1.5)
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title="ROC Curves — Model Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )
    return fig


def plot_confusion_matrix(cm: list, model_name: str) -> go.Figure:
    """Annotated confusion matrix heatmap."""
    cm_arr = np.array(cm)
    labels = ["Stayed", "Left"]
    fig = go.Figure(go.Heatmap(
        z=cm_arr,
        x=labels, y=labels,
        colorscale=[[0, COLORS["surface"]], [1, COLORS["primary"]]],
        text=cm_arr, texttemplate="%{text}",
        textfont=dict(size=18, color="white"),
        showscale=False
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=f"Confusion Matrix — {model_name}",
        xaxis_title="Predicted",
        yaxis_title="Actual"
    )
    return fig


def plot_metrics_comparison(metrics_df: pd.DataFrame) -> go.Figure:
    """Radar chart comparing all models across all metrics."""
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    colors_list = [COLORS["primary"], COLORS["secondary"], COLORS["success"]]

    fig = go.Figure()
    for (_, row), color in zip(metrics_df.iterrows(), colors_list):
        vals = [row[m] for m in metric_cols] + [row[metric_cols[0]]]
        cats = metric_cols + [metric_cols[0]]

        fill_color = "rgba(99,102,241,0.2)"

        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill="toself",
            name=row["Model"],
            line_color=color,
            fillcolor=fill_color
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(148,163,184,0.2)"),
            angularaxis=dict(gridcolor="rgba(148,163,184,0.2)")
        ),
        title="Model Performance Comparison",
        showlegend=True,
        **LAYOUT_BASE
    )
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of top feature importances."""
    top = importance_df.head(top_n).sort_values("Importance", ascending=True)
    fig = go.Figure(go.Bar(
        x=top["Importance"], y=top["Feature"],
        orientation="h",
        marker=dict(
            color=top["Importance"],
            colorscale=[[0, COLORS["info"]], [1, COLORS["primary"]]]
        )
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=f"Top {top_n} Feature Importances",
        xaxis_title="Importance Score",
        yaxis_title=""
    )
    return fig


def plot_shap_summary(shap_values: np.ndarray, feature_names: List[str], top_n: int = 15) -> go.Figure:
    """SHAP mean absolute impact — horizontal bar."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    idx = np.argsort(mean_abs)[-top_n:]
    feats = [feature_names[i] for i in idx]
    vals = mean_abs[idx]

    fig = go.Figure(go.Bar(
        x=vals, y=feats,
        orientation="h",
        marker=dict(
            color=vals,
            colorscale=[[0, COLORS["success"]], [0.5, COLORS["warning"]], [1, COLORS["danger"]]]
        )
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title="SHAP Global Feature Importance",
        xaxis_title="Mean |SHAP Value|",
        yaxis_title=""
    )
    return fig


def plot_shap_individual(shap_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Waterfall-style bar for individual SHAP explanation."""
    top = shap_df.head(top_n).sort_values("SHAP Value")
    colors = [COLORS["danger"] if v > 0 else COLORS["success"] for v in top["SHAP Value"]]

    fig = go.Figure(go.Bar(
        x=top["SHAP Value"],
        y=top["Display Name"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in top["SHAP Value"]],
        textposition="outside"
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title="Individual Prediction Explanation (SHAP)",
        xaxis_title="SHAP Value (impact on prediction)",
        yaxis_title="",
        shapes=[dict(
            type="line", x0=0, x1=0, y0=-0.5, y1=len(top) - 0.5,
            line=dict(color=COLORS["muted"], width=1.5)
        )]
    )
    return fig


def plot_attrition_risk_gauge(probability: float) -> go.Figure:
    """Gauge chart showing attrition probability."""
    color = (
        COLORS["success"] if probability < 0.3
        else COLORS["warning"] if probability < 0.6
        else COLORS["danger"]
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(probability * 100, 1),
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Attrition Risk Score", "font": {"size": 16, "color": COLORS["text"]}},
        number={"suffix": "%", "font": {"size": 28, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": COLORS["muted"]},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": COLORS["surface"],
            "bordercolor": "rgba(255,255,255,0.1)",
            "steps": [
                {"range": [0, 30], "color": "rgba(16,185,129,0.15)"},
                {"range": [30, 60], "color": "rgba(245,158,11,0.15)"},
                {"range": [60, 100], "color": "rgba(239,68,68,0.15)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.75,
                "value": probability * 100
            }
        }
    ))
    fig.update_layout(**LAYOUT_BASE, height=280)
    return fig
