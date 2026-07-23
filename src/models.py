"""
Machine Learning Models Module
Trains, evaluates, and compares Logistic Regression, Random Forest, and XGBoost
for employee attrition prediction.
"""

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

# pyrefly: ignore [missing-import]
from imblearn.over_sampling import SMOTE
# pyrefly: ignore [missing-import]
import xgboost as xgb

logger = logging.getLogger(__name__)

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)



# ── Model Definitions ─────────────────────────────────────────────────────────

def get_model_configs() -> Dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
            C=0.1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=3,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0
        )
    }


# ── Training ──────────────────────────────────────────────────────────────────

class AttritionModelTrainer:

    def __init__(self):
        self.models = get_model_configs()
        self.trained_models: Dict[str, Any] = {}
        self.evaluation_results: Dict[str, Dict] = {}
        self.best_model_name: Optional[str] = None

    def balance_classes(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE oversampling to handle class imbalance."""
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        logger.info(
            f"SMOTE: {y.sum()} → {y_resampled.sum()} positive samples "
            f"({y_resampled.mean()*100:.1f}% attrition)"
        )
        return X_resampled, y_resampled

    def train_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        use_smote: bool = True
    ) -> Dict[str, Dict]:
        """
        Train all models and evaluate them.
        Returns dict of evaluation metrics per model.
        """
        if use_smote:
            X_train_bal, y_train_bal = self.balance_classes(X_train, y_train)
        else:
            X_train_bal, y_train_bal = X_train.values, y_train.values

        results = {}

        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            try:
                model.fit(X_train_bal, y_train_bal)
                self.trained_models[name] = model

                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

                metrics = self._compute_metrics(y_test, y_pred, y_prob)
                cv_scores = self._cross_validate(model, X_train, y_train)
                metrics["cv_roc_auc_mean"] = cv_scores.mean()
                metrics["cv_roc_auc_std"] = cv_scores.std()

                # ROC curve data
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                metrics["roc_fpr"] = fpr.tolist()
                metrics["roc_tpr"] = tpr.tolist()
                metrics["y_prob"] = y_prob.tolist()
                metrics["y_pred"] = y_pred.tolist()
                metrics["y_test"] = y_test.tolist()

                results[name] = metrics
                logger.info(
                    f"{name} — AUC: {metrics['roc_auc']:.3f}, "
                    f"F1: {metrics['f1']:.3f}"
                )

            except Exception as e:
                logger.error(f"Training failed for {name}: {e}")
                continue

        self.evaluation_results = results
        self.best_model_name = max(results, key=lambda k: results[k]["roc_auc"])
        logger.info(f"Best model: {self.best_model_name}")
        return results

    def _compute_metrics(
        self,
        y_true,
        y_pred,
        y_prob
    ) -> Dict:
        cm = confusion_matrix(y_true, y_pred)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_prob),
            "confusion_matrix": cm.tolist(),
        }

    def _cross_validate(self, model, X, y, cv: int = 5) -> np.ndarray:
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        return cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)

    def get_best_model(self):
        """Return the best trained model object."""
        if not self.best_model_name:
            raise RuntimeError("No models have been trained yet.")
        return self.trained_models[self.best_model_name]

    def predict_employee(
        self,
        X_employee: pd.DataFrame,
        model_name: Optional[str] = None
    ) -> Tuple[int, float]:
        """
        Predict attrition for a single employee.
        Returns (prediction, probability).
        """
        name = model_name or self.best_model_name
        model = self.trained_models[name]
        pred = model.predict(X_employee)[0]
        prob = model.predict_proba(X_employee)[0, 1]
        return int(pred), float(prob)

    def get_feature_importances(
        self,
        feature_names: list,
        model_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Return feature importances as a sorted DataFrame."""
        name = model_name or self.best_model_name
        model = self.trained_models[name]

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            return pd.DataFrame()

        df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        return df

    def save_models(self):
        """Persist trained models to disk."""
        for name, model in self.trained_models.items():
            path = MODELS_DIR / f"{name.replace(' ', '_').lower()}.pkl"
            with open(path, "wb") as f:
                pickle.dump(model, f)
        logger.info(f"Saved {len(self.trained_models)} models to {MODELS_DIR}")

    def load_model(self, model_name: str):
        """Load a model from disk."""
        path = MODELS_DIR / f"{model_name.replace(' ', '_').lower()}.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)


def build_metrics_dataframe(results: Dict[str, Dict]) -> pd.DataFrame:
    """Convert evaluation results dict to a clean comparison DataFrame."""
    rows = []
    for model_name, metrics in results.items():
        rows.append({
            "Model": model_name,
            "Accuracy": round(metrics["accuracy"], 4),
            "Precision": round(metrics["precision"], 4),
            "Recall": round(metrics["recall"], 4),
            "F1-Score": round(metrics["f1"], 4),
            "ROC-AUC": round(metrics["roc_auc"], 4),
            "CV AUC (mean)": round(metrics.get("cv_roc_auc_mean", 0), 4),
            "CV AUC (std)": round(metrics.get("cv_roc_auc_std", 0), 4),
        })
    return pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
