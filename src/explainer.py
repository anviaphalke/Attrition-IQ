"""
SHAP Explainability Module
Generates per-prediction and global explanations for attrition predictions.
"""

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import shap
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AttritionExplainer:
    """
    Wraps SHAP to explain model predictions for HR professionals.
    """

    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self._background_data = None

    def fit(self, X_background: pd.DataFrame, n_background: int = 100):
        """
        Fit SHAP explainer on background data.
        Uses TreeExplainer for tree-based models, LinearExplainer otherwise.
        """
        bg = shap.sample(X_background, min(n_background, len(X_background)))
        self._background_data = bg

        model_type = type(self.model).__name__
        try:
            if model_type in ("RandomForestClassifier", "XGBClassifier"):
                self.explainer = shap.TreeExplainer(self.model)
                logger.info(f"Using TreeExplainer for {model_type}")
            else:
                self.explainer = shap.LinearExplainer(
                    self.model, bg, feature_perturbation="interventional"
                )
                logger.info(f"Using LinearExplainer for {model_type}")
        except Exception as e:
            logger.warning(f"Falling back to KernelExplainer: {e}")
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, bg
            )

    def explain_global(self, X: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Compute global SHAP values and mean absolute importance.
        Returns (shap_values_array, importance_df).
        """
        if self.explainer is None:
            raise RuntimeError("Call fit() before explain_global()")

        sample = X.iloc[:min(300, len(X))]
        sv = self.explainer.shap_values(sample)

        # For binary classifiers, take class-1 SHAP values
        if isinstance(sv, list):
            sv = sv[1]
        # Handle 3D arrays from KernelExplainer (n_samples, n_features, n_classes)
        if sv.ndim == 3:
            sv = sv[:, :, 1]

        self.shap_values = sv
        self._X_sample = sample

        mean_abs = np.abs(sv).mean(axis=0)
        importance = pd.DataFrame({
            "Feature": self.feature_names,
            "Mean |SHAP|": mean_abs
        }).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)

        return sv, importance

    def explain_individual(
        self,
        X_employee: pd.DataFrame,
        feature_display_names: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Compute SHAP values for a single employee record.
        Returns a DataFrame with feature, value, SHAP contribution, and direction.
        """
        if self.explainer is None:
            raise RuntimeError("Call fit() before explain_individual()")

        sv = self.explainer.shap_values(X_employee)
        if isinstance(sv, list):
            sv = sv[1]
        # Handle 3D arrays (n_samples, n_features, n_classes)
        if sv.ndim == 3:
            sv = sv[:, :, 1]

        shap_vals = sv[0] if sv.ndim > 1 else sv
        feature_vals = X_employee.iloc[0].values

        rows = []
        for feat, val, sv_val in zip(self.feature_names, feature_vals, shap_vals):
            display = (feature_display_names or {}).get(feat, feat.replace("_", " ").title())
            rows.append({
                "Feature": feat,
                "Display Name": display,
                "Value": round(float(val), 4),
                "SHAP Value": round(float(sv_val), 4),
                "Direction": "↑ Risk" if sv_val > 0 else "↓ Risk",
                "Abs SHAP": abs(float(sv_val))
            })

        df = pd.DataFrame(rows).sort_values("Abs SHAP", ascending=False).reset_index(drop=True)
        return df

    def get_top_risk_factors(
        self,
        X_employee: pd.DataFrame,
        n: int = 5
    ) -> List[Dict]:
        """Return top N risk/protective factors as human-readable strings."""
        shap_df = self.explain_individual(X_employee)
        risk = shap_df[shap_df["SHAP Value"] > 0].head(n)
        protect = shap_df[shap_df["SHAP Value"] < 0].head(n)

        risk_factors = []
        for _, row in risk.iterrows():
            risk_factors.append({
                "factor": row["Display Name"],
                "impact": row["SHAP Value"],
                "type": "risk"
            })
        for _, row in protect.iterrows():
            risk_factors.append({
                "factor": row["Display Name"],
                "impact": row["SHAP Value"],
                "type": "protective"
            })
        return risk_factors

    def generate_explanation_text(
        self,
        X_employee: pd.DataFrame,
        prediction: int,
        probability: float,
        employee_info: Optional[Dict] = None
    ) -> str:
        """
        Generate a natural-language explanation for HR professionals.
        """
        shap_df = self.explain_individual(X_employee)
        top_risk = shap_df[shap_df["SHAP Value"] > 0].head(3)
        top_protect = shap_df[shap_df["SHAP Value"] < 0].head(2)

        status = "HIGH RISK" if prediction == 1 else "LOW RISK"
        risk_pct = round(probability * 100, 1)

        lines = [
            f"Attrition Assessment: {status}",
            f"This employee has a {risk_pct}% probability of leaving.",
            "",
            "Primary Risk Drivers:"
        ]

        for _, row in top_risk.iterrows():
            lines.append(f"• {row['Display Name']}: contributing +{row['SHAP Value']:.3f} to attrition risk")

        if not top_protect.empty:
            lines.append("")
            lines.append("Protective Factors:")
            for _, row in top_protect.iterrows():
                lines.append(f"• {row['Display Name']}: reducing attrition risk by {abs(row['SHAP Value']):.3f}")

        lines.append("")
        lines.append("Recommended HR Actions:")
        if probability > 0.7:
            lines.append("🚨Immediate retention conversation recommended")
            lines.append("• Consider salary review or career advancement discussion")
            lines.append("• Review overtime workload and work-life balance")
        elif probability > 0.4:
            lines.append("⚠️ Schedule engagement check-in within 30 days")
            lines.append("• Review promotion history and growth opportunities")
        else:
            lines.append("✅ Monitor in next quarterly review")
            lines.append("• Maintain current engagement programs")

        return "\n".join(lines)
