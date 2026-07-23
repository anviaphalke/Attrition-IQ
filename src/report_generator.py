"""
HR Report Generator
Creates downloadable Excel and CSV reports from analysis results.
"""

import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import io
import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def generate_attrition_summary_excel(
    df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    feature_importance_df: Optional[pd.DataFrame] = None
) -> bytes:
    """
    Generate a multi-sheet Excel report with attrition analysis.
    Returns bytes suitable for Streamlit download_button.
    """
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Sheet 1: Executive Summary
        summary_data = _build_executive_summary(df)
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name="Executive Summary", index=False)

        # Sheet 2: Department Analysis
        dept_df = _build_department_analysis(df)
        dept_df.to_excel(writer, sheet_name="Department Analysis", index=False)

        # Sheet 3: Job Role Analysis
        role_df = _build_role_analysis(df)
        role_df.to_excel(writer, sheet_name="Job Role Analysis", index=False)

        # Sheet 4: Risk Employees
        risk_df = _build_risk_employee_list(df)
        risk_df.to_excel(writer, sheet_name="High Risk Employees", index=False)

        # Sheet 5: Model Performance
        metrics_df.to_excel(writer, sheet_name="Model Performance", index=False)

        # Sheet 6: Feature Importance
        if feature_importance_df is not None and not feature_importance_df.empty:
            feature_importance_df.to_excel(
                writer, sheet_name="Feature Importance", index=False
            )

    output.seek(0)
    return output.read()


def generate_risk_report_csv(df: pd.DataFrame, risk_scores: Optional[pd.Series] = None) -> bytes:
    """Generate a CSV of high-risk employees."""
    risk_df = _build_risk_employee_list(df, risk_scores)
    return risk_df.to_csv(index=False).encode("utf-8")


# ── Internal builders ─────────────────────────────────────────────────────────

def _build_executive_summary(df: pd.DataFrame) -> list:
    total = len(df)
    attrition_count = (df["Attrition"] == "Yes").sum()
    attrition_rate = attrition_count / total * 100

    rows = [
        {"Metric": "Report Generated", "Value": datetime.now().strftime("%Y-%m-%d %H:%M")},
        {"Metric": "Total Employees", "Value": total},
        {"Metric": "Attrition Count", "Value": attrition_count},
        {"Metric": "Attrition Rate (%)", "Value": round(attrition_rate, 2)},
        {"Metric": "Retained Employees", "Value": total - attrition_count},
    ]

    if "MonthlyIncome" in df.columns:
        avg_income_left = df[df["Attrition"] == "Yes"]["MonthlyIncome"].mean()
        avg_income_stayed = df[df["Attrition"] == "No"]["MonthlyIncome"].mean()
        rows += [
            {"Metric": "Avg Income (Left)", "Value": round(avg_income_left, 0)},
            {"Metric": "Avg Income (Stayed)", "Value": round(avg_income_stayed, 0)},
        ]

    if "YearsAtCompany" in df.columns:
        avg_tenure_left = df[df["Attrition"] == "Yes"]["YearsAtCompany"].mean()
        rows.append({"Metric": "Avg Tenure of Leavers (years)", "Value": round(avg_tenure_left, 1)})

    if "OverTime" in df.columns:
        ot_attrition = (
            df[df["OverTime"] == "Yes"]["Attrition"]
            .eq("Yes").mean() * 100
        )
        rows.append({"Metric": "Overtime Workers Attrition Rate (%)", "Value": round(ot_attrition, 1)})

    return rows


def _build_department_analysis(df: pd.DataFrame) -> pd.DataFrame:
    dept_stats = df.groupby("Department").agg(
        Total=("Attrition", "count"),
        Attrition_Count=("Attrition", lambda x: (x == "Yes").sum()),
    ).reset_index()
    dept_stats["Attrition_Rate_%"] = (
        dept_stats["Attrition_Count"] / dept_stats["Total"] * 100
    ).round(2)

    if "MonthlyIncome" in df.columns:
        income = df.groupby("Department")["MonthlyIncome"].mean().round(0).reset_index()
        income.columns = ["Department", "Avg_Monthly_Income"]
        dept_stats = dept_stats.merge(income, on="Department")

    return dept_stats.sort_values("Attrition_Rate_%", ascending=False)


def _build_role_analysis(df: pd.DataFrame) -> pd.DataFrame:
    role_stats = df.groupby("JobRole").agg(
        Total=("Attrition", "count"),
        Attrition_Count=("Attrition", lambda x: (x == "Yes").sum()),
    ).reset_index()
    role_stats["Attrition_Rate_%"] = (
        role_stats["Attrition_Count"] / role_stats["Total"] * 100
    ).round(2)
    return role_stats.sort_values("Attrition_Rate_%", ascending=False)


def _build_risk_employee_list(
    df: pd.DataFrame,
    risk_scores: Optional[pd.Series] = None
) -> pd.DataFrame:
    """Return employees with actual attrition = Yes, or highest risk scores."""
    cols = ["EmployeeID", "Department", "JobRole", "Age",
            "MonthlyIncome", "YearsAtCompany", "OverTime", "Attrition"]
    cols = [c for c in cols if c in df.columns]

    if risk_scores is not None:
        result = df[cols].copy()
        result["Predicted_Risk_%"] = (risk_scores.values * 100).round(1)
        return result.sort_values("Predicted_Risk_%", ascending=False).head(50)

    return df[df["Attrition"] == "Yes"][cols].reset_index(drop=True)
