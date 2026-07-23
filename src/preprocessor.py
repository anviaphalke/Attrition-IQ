"""
HR Data Preprocessor — Production-Ready
========================================
Features:
  • Automatic column mapping  (fuzzy alias resolution, 60+ known variants)
  • Automatic target detection (finds attrition/left/churn column)
  • Automatic Employee-ID removal
  • Currency detection & cleaning  ($1,200 -> 1200.0)
  • Missing value handling          (median / mode imputation per type)
  • Unknown / extra column handling (logged and kept as numeric if possible)
  • Feature engineering             (8 derived signals that boost accuracy)
  • Train / test compatibility      (fit=True on train, fit=False on test/new rows)
  • Dataset summary generation      (structured dict with stats & warnings)
  • Proper logging throughout
  • Comprehensive error handling
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

# == Canonical column aliases ===================================================
# Maps any known variant spelling -> canonical name used downstream.
COLUMN_ALIASES: Dict[str, str] = {
    # Target
    "attrition": "Attrition",
    "left": "Attrition",
    "churned": "Attrition",
    "turnover": "Attrition",
    "employee_left": "Attrition",
    "left_company": "Attrition",
    # Demographics
    "age": "Age",
    "employee_age": "Age",
    "gender": "Gender",
    "sex": "Gender",
    "marital_status": "MaritalStatus",
    "maritalstatus": "MaritalStatus",
    # Job info
    "department": "Department",
    "dept": "Department",
    "division": "Department",
    "job_role": "JobRole",
    "jobrole": "JobRole",
    "role": "JobRole",
    "position": "JobRole",
    "job_title": "JobRole",
    "job_level": "JobLevel",
    "joblevel": "JobLevel",
    "job_involvement": "JobInvolvement",
    "jobinvolvement": "JobInvolvement",
    "job_satisfaction": "JobSatisfaction",
    "jobsatisfaction": "JobSatisfaction",
    "education": "Education",
    "education_level": "Education",
    "education_field": "EducationField",
    "educationfield": "EducationField",
    # Compensation
    "monthly_income": "MonthlyIncome",
    "monthlyincome": "MonthlyIncome",
    "salary": "MonthlyIncome",
    "monthly_salary": "MonthlyIncome",
    "monthly_rate": "MonthlyRate",
    "monthlyrate": "MonthlyRate",
    "percent_salary_hike": "PercentSalaryHike",
    "percentsalaryhike": "PercentSalaryHike",
    "salary_hike": "PercentSalaryHike",
    "stock_option_level": "StockOptionLevel",
    "stockoptionlevel": "StockOptionLevel",
    "stock_options": "StockOptionLevel",
    # Work conditions
    "overtime": "OverTime",
    "over_time": "OverTime",
    "overwork": "OverTime",
    "business_travel": "BusinessTravel",
    "businesstravel": "BusinessTravel",
    "travel": "BusinessTravel",
    "distance_from_home": "DistanceFromHome",
    "distancefromhome": "DistanceFromHome",
    "commute_distance": "DistanceFromHome",
    "environment_satisfaction": "EnvironmentSatisfaction",
    "environmentsatisfaction": "EnvironmentSatisfaction",
    "work_life_balance": "WorkLifeBalance",
    "worklifebalance": "WorkLifeBalance",
    "performance_rating": "PerformanceRating",
    "performancerating": "PerformanceRating",
    "relationship_satisfaction": "RelationshipSatisfaction",
    "relationshipsatisfaction": "RelationshipSatisfaction",
    "training_times_last_year": "TrainingTimesLastYear",
    "trainingtimeslastyear": "TrainingTimesLastYear",
    # Tenure
    "years_at_company": "YearsAtCompany",
    "yearsatcompany": "YearsAtCompany",
    "tenure": "YearsAtCompany",
    "total_working_years": "TotalWorkingYears",
    "totalworkingyears": "TotalWorkingYears",
    "years_in_current_role": "YearsInCurrentRole",
    "yearsincurrentrole": "YearsInCurrentRole",
    "years_since_last_promotion": "YearsSinceLastPromotion",
    "yearssincelastpromotion": "YearsSinceLastPromotion",
    "years_with_curr_manager": "YearsWithCurrManager",
    "yearswithcurrmanager": "YearsWithCurrManager",
    "num_companies_worked": "NumCompaniesWorked",
    "numcompaniesworked": "NumCompaniesWorked",
    "number_of_companies": "NumCompaniesWorked",
}

# == Column type schemas ========================================================
CATEGORICAL_COLUMNS: List[str] = [
    "BusinessTravel", "Department", "EducationField",
    "Gender", "JobRole", "MaritalStatus", "OverTime",
]

ORDINAL_COLUMNS: List[str] = [
    "Education", "EnvironmentSatisfaction", "JobInvolvement",
    "JobLevel", "JobSatisfaction", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "WorkLifeBalance",
]

NUMERIC_COLUMNS: List[str] = [
    "Age", "DistanceFromHome", "MonthlyIncome", "MonthlyRate",
    "NumCompaniesWorked", "PercentSalaryHike", "TotalWorkingYears",
    "TrainingTimesLastYear", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "YearsWithCurrManager",
]

# Column-name regex patterns that indicate ID / meta columns to drop
ID_COLUMN_PATTERNS: List[str] = [
    r".*employee.*id.*", r".*emp.*id.*",
    r".*employee.*number.*", r".*employee.*count.*",
    r".*over.*18.*", r".*standard.*hours.*",
]

# Strings treated as the positive (attrition) class
POSITIVE_LABELS = {"yes", "1", "true", "left", "churned", "1.0"}


# == Module-level helpers =======================================================

def _normalise_key(name: str) -> str:
    """Lower-case and replace non-alphanumeric runs with underscores."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _is_currency_column(series: pd.Series) -> bool:
    """Return True if >30 % of the first 50 non-null values look like currency."""
    sample = series.dropna().astype(str).head(50)
    if len(sample) == 0:
        return False
    currency_re = re.compile(r"[$£€₹,]")
    hits = sample.apply(lambda v: bool(currency_re.search(v))).sum()
    return hits > len(sample) * 0.30


def _clean_currency(series: pd.Series) -> pd.Series:
    """Strip currency symbols and thousand separators, return float series."""
    return (
        series.astype(str)
        .str.replace(r"[$£€₹,\s]", "", regex=True)
        .replace("", np.nan)
        .pipe(pd.to_numeric, errors="coerce")
    )


def _is_id_column(name: str) -> bool:
    """Return True if the column name matches a known ID/meta pattern."""
    key = _normalise_key(name)
    for pat in ID_COLUMN_PATTERNS:
        if re.fullmatch(pat, key):
            return True
    return False


# == Main class ================================================================

class HRDataPreprocessor:
    """
    End-to-end HR data preprocessing pipeline.

    Training usage:
        prep = HRDataPreprocessor()
        X, y, df_clean = prep.full_pipeline(df_raw, fit=True)

    Inference usage (single employee or test set):
        emp_df_eng = prep.engineer_features(prep.clean_data(df_new))
        X_emp, _   = prep.encode_and_scale(emp_df_eng, fit=False)
    """

    def __init__(self) -> None:
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.target_column: Optional[str] = None
        self.employee_id_column: Optional[str] = None  # auto-detected raw ID column
        self.is_fitted: bool = False
        self.preprocessing_report: Dict = {}
        self._col_mapping: Dict[str, str] = {}

    # -- Public API ------------------------------------------------------------

    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate minimum viability of the DataFrame.
        Returns (is_valid, list_of_issues).
        """
        issues: List[str] = []

        if len(df) < 10:
            issues.append("Dataset too small — minimum 10 rows required.")

        if df.empty or df.shape[1] < 2:
            issues.append("Dataset must have at least 2 columns.")

        if self._detect_target(df) is None:
            issues.append(
                "Could not detect a target (attrition/churn/left) column. "
                "Please ensure one exists or rename it to 'Attrition'."
            )

        all_null = df.columns[df.isnull().mean() == 1.0].tolist()
        if all_null:
            issues.append(f"Columns with 100% missing values: {all_null}")

        return len(issues) == 0, issues

    def generate_dataset_summary(self, df: pd.DataFrame) -> Dict:
        """
        Return a structured summary dict with schema stats and data-quality warnings.
        Intended to be called on the RAW DataFrame before any cleaning.
        """
        summary: Dict = {
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "column_types": {},
            "missing_pct": {},
            "target_column": self._detect_target(df),
            "id_columns_detected": [],
            "currency_columns_detected": [],
            "warnings": [],
        }

        for col in df.columns:
            summary["column_types"][col] = str(df[col].dtype)
            pct = round(df[col].isnull().mean() * 100, 2)
            summary["missing_pct"][col] = pct
            if pct > 20:
                summary["warnings"].append(
                    f"Column '{col}' has {pct:.1f}% missing values."
                )
            if _is_id_column(col):
                summary["id_columns_detected"].append(col)
            if df[col].dtype == object and _is_currency_column(df[col]):
                summary["currency_columns_detected"].append(col)

        if summary["target_column"] is None:
            summary["warnings"].append(
                "No attrition/target column detected automatically."
            )

        logger.info(
            "Dataset summary: %d rows x %d cols | target=%s | warnings=%d",
            summary["n_rows"], summary["n_cols"],
            summary["target_column"], len(summary["warnings"]),
        )
        return summary

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 1: Clean raw DataFrame.
          1. Column alias mapping
          2. Employee-ID removal (auto-detected)
          3. Currency column cleaning
          4. Target standardisation -> 'Attrition' Yes/No
          5. Duplicate removal
          6. Unknown column handling
          7. Missing value imputation
          8. Numeric dtype coercion
          9. Outlier clipping [1st-99th percentile]
        """
        df = df.copy()
        report: Dict = {
            "original_shape": df.shape,
            "steps": [],
            "warnings": [],
        }

        # 1. Column alias mapping
        mapping = self._build_column_mapping(df)
        if mapping:
            df.rename(columns=mapping, inplace=True)
            report["steps"].append(f"Column aliases resolved: {mapping}")
            logger.info("Column mapping applied: %s", mapping)
            self._col_mapping.update(mapping)

        # 2. Auto-detect & drop ID / constant columns
        id_cols = self._detect_id_columns(df)
        if id_cols:
            df.drop(columns=id_cols, inplace=True)
            report["steps"].append(f"Dropped ID/constant columns: {id_cols}")
            logger.info("Dropped ID columns: %s", id_cols)
            # Keep the best employee-ID column name so the UI can reference df_raw
            if self.employee_id_column is None and id_cols:
                self.employee_id_column = id_cols[0]
                logger.info("Employee ID column recorded: %s", self.employee_id_column)

        # 3. Currency cleaning
        currency_cols = [
            c for c in df.columns
            if df[c].dtype == object and _is_currency_column(df[c])
        ]
        for col in currency_cols:
            df[col] = _clean_currency(df[col])
            report["steps"].append(f"Currency cleaned: '{col}'")
            logger.info("Currency cleaned column: %s", col)

        # 4. Standardise target column
        target = self._detect_target(df)
        if target is None:
            raise ValueError(
                "Could not detect a target column. "
                "Expected a column named 'Attrition', 'Left', 'Churned', etc."
            )
        self.target_column = target

        if target != "Attrition":
            df.rename(columns={target: "Attrition"}, inplace=True)
            report["steps"].append(f"Renamed target '{target}' -> 'Attrition'")
            logger.info("Target column renamed: %s -> Attrition", target)

        df["Attrition"] = df["Attrition"].apply(
            lambda x: "Yes" if str(x).strip().lower() in POSITIVE_LABELS else "No"
        )

        # 5. Duplicate removal
        n_dupes = int(df.duplicated().sum())
        if n_dupes:
            df.drop_duplicates(inplace=True)
            report["steps"].append(f"Removed {n_dupes} duplicate rows.")
            logger.info("Removed %d duplicate rows.", n_dupes)

        # 6. Unknown column handling
        known_cols = (
            {"Attrition"}
            | set(CATEGORICAL_COLUMNS)
            | set(ORDINAL_COLUMNS)
            | set(NUMERIC_COLUMNS)
        )
        extra_cols = [c for c in df.columns if c not in known_cols]
        if extra_cols:
            coerced, dropped = [], []
            for col in extra_cols:
                try:
                    coerced_col = pd.to_numeric(df[col], errors="coerce")
                    non_null = coerced_col.notna().mean()
                    if non_null >= 0.5:
                        df[col] = coerced_col
                        coerced.append(col)
                    else:
                        df.drop(columns=[col], inplace=True)
                        dropped.append(col)
                except Exception:
                    df.drop(columns=[col], inplace=True)
                    dropped.append(col)

            if coerced:
                report["steps"].append(f"Unknown columns kept as numeric: {coerced}")
                logger.info("Unknown columns kept as numeric: %s", coerced)
            if dropped:
                msg = f"Unknown columns dropped (insufficient signal): {dropped}"
                report["warnings"].append(msg)
                logger.warning(msg)

        # 7. Missing value imputation
        n_missing_before = int(df.isnull().sum().sum())
        if n_missing_before > 0:
            df = self._impute_missing(df)
            n_missing_after = int(df.isnull().sum().sum())
            report["steps"].append(
                f"Imputed {n_missing_before - n_missing_after} missing values "
                f"({n_missing_before} -> {n_missing_after} NaNs remaining)."
            )
            logger.info("Imputation: %d -> %d NaNs.", n_missing_before, n_missing_after)

        # 8. Numeric dtype coercion
        num_cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
        for col in num_cols:
            median_val = df[col].median() if df[col].notna().any() else 0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(median_val)

        ord_cols = [c for c in ORDINAL_COLUMNS if c in df.columns]
        for col in ord_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(1).astype(int)

        # 9. Outlier clipping [1st-99th pct]
        for col in num_cols:
            if col in df.columns:
                lo, hi = df[col].quantile([0.01, 0.99])
                df[col] = df[col].clip(lo, hi)

        report["final_shape"] = df.shape
        self.preprocessing_report = report

        logger.info(
            "clean_data: %s -> %s | steps=%d | warnings=%d",
            report["original_shape"], df.shape,
            len(report["steps"]), len(report["warnings"]),
        )
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 2: Add derived features that boost predictive power.
        Safe to call on single-row DataFrames (inference mode).
        """
        df = df.copy()
        added: List[str] = []

        # Income-to-level ratio — is employee paid fairly for their level?
        if "MonthlyIncome" in df.columns and "JobLevel" in df.columns:
            if len(df) > 1:
                level_median = df.groupby("JobLevel")["MonthlyIncome"].transform("median")
            else:
                level_median = df["MonthlyIncome"]
            df["IncomeToLevelRatio"] = (
                df["MonthlyIncome"] / level_median.replace(0, np.nan)
            ).fillna(1.0)
            added.append("IncomeToLevelRatio")

        # Promotion lag — time without promotion relative to total tenure
        if "YearsAtCompany" in df.columns and "YearsSinceLastPromotion" in df.columns:
            df["PromotionLag"] = (
                df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
            )
            added.append("PromotionLag")

        # Tenured flag — proxy for institutional loyalty
        if "YearsAtCompany" in df.columns:
            df["IsTenured"] = (df["YearsAtCompany"] >= 5).astype(int)
            added.append("IsTenured")

        # Burnout risk — overtime penalised by poor work-life balance
        if "OverTime" in df.columns and "WorkLifeBalance" in df.columns:
            ot_binary = (df["OverTime"] == "Yes").astype(int)
            df["BurnoutRisk"] = ot_binary * (5 - df["WorkLifeBalance"])
            added.append("BurnoutRisk")

        # Overall satisfaction — average of job + environment
        if "JobSatisfaction" in df.columns and "EnvironmentSatisfaction" in df.columns:
            df["OverallSatisfaction"] = (
                df["JobSatisfaction"] + df["EnvironmentSatisfaction"]
            ) / 2
            added.append("OverallSatisfaction")

        # Career mobility — companies per working year
        if "NumCompaniesWorked" in df.columns and "TotalWorkingYears" in df.columns:
            df["CareerMobility"] = df["NumCompaniesWorked"] / (
                df["TotalWorkingYears"] + 1
            )
            added.append("CareerMobility")

        # Loyalty index — fraction of career spent at current employer
        if "YearsAtCompany" in df.columns and "TotalWorkingYears" in df.columns:
            df["LoyaltyIndex"] = df["YearsAtCompany"] / (
                df["TotalWorkingYears"] + 1
            )
            added.append("LoyaltyIndex")

        # Distance + overtime compound risk
        if "DistanceFromHome" in df.columns and "OverTime" in df.columns:
            ot = (df["OverTime"] == "Yes").astype(int)
            df["DistanceOvertimeRisk"] = df["DistanceFromHome"] * (1 + ot)
            added.append("DistanceOvertimeRisk")

        if added:
            logger.info("Feature engineering: +%d columns: %s", len(added), added)

        return df

    def encode_and_scale(
        self,
        df: pd.DataFrame,
        fit: bool = True,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Stage 3: Label-encode categoricals and standard-scale numerics.

        Args:
            df:  Engineered DataFrame (output of engineer_features).
            fit: True on training data; False for test/inference (transform only).

        Returns:
            (X, y): X is the scaled feature matrix, y is the binary target series
                    (or None if 'Attrition' column is absent).
        """
        df = df.copy()

        # Extract target
        y: Optional[pd.Series] = None
        if "Attrition" in df.columns:
            y = (df["Attrition"] == "Yes").astype(int)
            y.name = "Attrition"
            df.drop(columns=["Attrition"], inplace=True)
        else:
            logger.debug("encode_and_scale: 'Attrition' not present (inference mode).")

        # Label-encode categorical columns
        cat_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
        for col in cat_cols:
            df[col] = df[col].astype(str)
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    known = set(le.classes_)
                    df[col] = df[col].apply(
                        lambda v, _le=le, _k=known: (
                            _le.transform([v])[0] if v in _k else -1
                        )
                    )
                else:
                    logger.warning("Column '%s' unseen at fit time — encoding as -1.", col)
                    df[col] = -1

        # Scale all numeric features
        num_cols = list(df.select_dtypes(include=[np.number]).columns)

        if fit:
            if num_cols:
                df[num_cols] = self.scaler.fit_transform(df[num_cols])
            self.feature_names = list(df.columns)
            self.is_fitted = True
            logger.info(
                "Preprocessor fitted. Features=%d | Encoded categoricals=%d",
                len(self.feature_names), len(cat_cols),
            )
        else:
            if not self.is_fitted:
                raise RuntimeError(
                    "Preprocessor is not fitted yet. Call full_pipeline(fit=True) first."
                )

            # Add any missing columns (fill with 0)
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
                    logger.debug("Added missing column '%s' with fill=0 for inference.", col)

            # Drop extra columns not seen at fit time
            extra = [c for c in df.columns if c not in self.feature_names]
            if extra:
                logger.warning("Dropping unseen columns at inference: %s", extra)
                df.drop(columns=extra, inplace=True)

            # Reorder to match training schema
            df = df.reindex(columns=self.feature_names, fill_value=0)

            num_cols_inf = list(df.select_dtypes(include=[np.number]).columns)
            if num_cols_inf:
                df[num_cols_inf] = self.scaler.transform(df[num_cols_inf])

        return df, y

    def full_pipeline(
        self,
        df: pd.DataFrame,
        fit: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Run the complete end-to-end preprocessing pipeline.

        Returns:
            X_processed : pd.DataFrame  — feature matrix ready for ML
            y           : pd.Series     — binary target (1=attrition, 0=retained)
            df_clean    : pd.DataFrame  — human-readable cleaned DataFrame (pre-encoding)
        """
        logger.info("full_pipeline started: shape=%s | fit=%s", df.shape, fit)

        valid, issues = self.validate_dataframe(df)
        if not valid:
            raise ValueError(
                "Data validation failed:\n" + "\n".join(f"  - {i}" for i in issues)
            )

        # Log summary (non-blocking)
        try:
            summary = self.generate_dataset_summary(df)
            self.preprocessing_report["dataset_summary"] = summary
        except Exception as exc:
            logger.warning("Dataset summary generation failed (non-fatal): %s", exc)

        df_clean = self.clean_data(df)
        df_engineered = self.engineer_features(df_clean)
        X, y = self.encode_and_scale(df_engineered, fit=fit)

        if y is None:
            raise ValueError("Target column could not be extracted after preprocessing.")

        logger.info(
            "full_pipeline complete: X=%s | attrition rate=%.1f%%",
            X.shape, float(y.mean()) * 100,
        )
        return X, y, df_clean

    # -- Private helpers -------------------------------------------------------

    def _build_column_mapping(self, df: pd.DataFrame) -> Dict[str, str]:
        """Build {original_col -> canonical_col} for known alias columns."""
        mapping: Dict[str, str] = {}
        for col in df.columns:
            key = _normalise_key(col)
            canonical = COLUMN_ALIASES.get(key)
            if canonical and col != canonical and canonical not in df.columns:
                mapping[col] = canonical
        return mapping

    def _detect_target(self, df: pd.DataFrame) -> Optional[str]:
        """
        Return the name of the attrition/target column, or None if undetectable.
        Strategy: exact alias match first, then binary-value heuristic.
        """
        target_keys = {
            "attrition", "left", "churned", "turnover",
            "employee_left", "left_company",
        }
        for col in df.columns:
            if _normalise_key(col) in target_keys:
                return col

        # Fallback: binary column containing at least one positive label
        for col in df.columns:
            vals = set(df[col].dropna().astype(str).str.lower().unique())
            if 1 <= len(vals) <= 2 and vals & POSITIVE_LABELS:
                logger.info(
                    "Auto-detected target column by value pattern: '%s' (values=%s)",
                    col, vals,
                )
                return col
        return None

    def _detect_id_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Detect Employee-ID and constant/meta columns to drop.
        Returns columns in priority order:
          1. Name-pattern matches (EmployeeID, EmpID, etc.)
          2. Constant columns (only 1 unique value)
          3. All-unique numeric columns (row-index proxies)

        The first name-pattern match is saved as ``self.employee_id_column``
        so the UI can still reference it in df_raw for employee selection.
        """
        name_based: List[str] = []
        constant: List[str] = []
        row_index: List[str] = []
        n = len(df)

        for col in df.columns:
            if col == "Attrition":
                continue
            if _is_id_column(col):
                name_based.append(col)
                continue
            if df[col].nunique(dropna=False) <= 1:
                constant.append(col)
                continue
            if df[col].dtype in (np.int64, np.int32, np.float64) and df[col].nunique() == n:
                row_index.append(col)

        # The best employee-ID candidate is the first name-based hit; fall back
        # to the first all-unique numeric column if no name match was found.
        if name_based and self.employee_id_column is None:
            self.employee_id_column = name_based[0]
        elif row_index and self.employee_id_column is None:
            self.employee_id_column = row_index[0]

        return name_based + constant + row_index

    @staticmethod
    def find_employee_id_column(df: pd.DataFrame) -> Optional[str]:
        """
        Public static helper — returns the best employee-ID column in *df*
        without needing a fitted preprocessor instance.

        Useful for determining which column to use in a UI selector before
        the preprocessor has been run (e.g., on df_raw).
        """
        n = len(df)
        # 1. Name-pattern match
        for col in df.columns:
            if _is_id_column(col):
                return col
        # 2. All-unique numeric column
        for col in df.columns:
            if df[col].dtype in (np.int64, np.int32, np.float64) and df[col].nunique() == n:
                return col
        return None

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute column-by-column with the appropriate strategy:
          - Numeric / ordinal -> median
          - Categorical / object -> most_frequent
        """
        cat_cols = [
            c for c in CATEGORICAL_COLUMNS
            if c in df.columns and df[c].isnull().any()
        ]
        ord_cols = [
            c for c in ORDINAL_COLUMNS
            if c in df.columns and df[c].isnull().any()
        ]
        num_cols = [
            c for c in df.columns
            if c not in cat_cols
            and c != "Attrition"
            and pd.api.types.is_numeric_dtype(df[c])
            and df[c].isnull().any()
        ]
        obj_cols = [
            c for c in df.select_dtypes(include="object").columns
            if c not in cat_cols
            and c != "Attrition"
            and df[c].isnull().any()
        ]

        if num_cols + ord_cols:
            med_imp = SimpleImputer(strategy="median")
            df[num_cols + ord_cols] = med_imp.fit_transform(df[num_cols + ord_cols])

        if cat_cols + obj_cols:
            mode_imp = SimpleImputer(strategy="most_frequent")
            df[cat_cols + obj_cols] = mode_imp.fit_transform(df[cat_cols + obj_cols])

        return df


# == Module-level utility ======================================================

def get_feature_importance_names(preprocessor: HRDataPreprocessor) -> List[str]:
    """Return human-readable feature names from a fitted preprocessor."""
    return preprocessor.feature_names
