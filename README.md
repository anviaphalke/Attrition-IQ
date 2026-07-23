# 🧠 Employee Attrition Prediction System

A production-ready HR analytics platform that predicts employee attrition using Machine Learning and Explainable AI. Built for HR managers to make data-driven retention decisions.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange)
![SHAP](https://img.shields.io/badge/SHAP-0.44+-green)

---

## ✨ Features

### 📊 EDA Dashboard
- Interactive workforce demographics visualizations
- Department-wise attrition analysis
- Salary distribution by role and department
- Satisfaction score heatmaps
- Overtime impact analysis
- Tenure vs attrition trends

### 🤖 Machine Learning Models
- **Logistic Regression** — interpretable baseline
- **Random Forest** — robust ensemble method
- **XGBoost** — best-performing gradient boosted trees
- SMOTE oversampling to handle class imbalance
- 5-fold stratified cross-validation
- Full metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- ROC curves and confusion matrices

### 🔮 Individual Predictions + SHAP
- Input any employee's profile for instant risk scoring
- Attrition probability gauge (0–100%)
- SHAP waterfall chart explaining each prediction
- Natural language HR recommendation text

### 🤝 AI HR Assistant (Claude)
- Natural language Q&A about your workforce data
- "Which department has the highest attrition?"
- "What salary range is at highest risk?"
- Context-aware responses grounded in your actual data

### 📥 Downloadable Reports
- Multi-sheet Excel: Executive Summary, Department Analysis, Risk Employee List, Model Performance
- CSV export of high-risk employees

---

## 🚀 Quick Start

### Local Development
```bash
git clone https://github.com/your-org/hr-attrition-ai
cd hr-attrition-ai
pip install -r requirements.txt
streamlit run app.py
```

### Deploy to Streamlit Cloud
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file: `app.py`
5. Deploy!

---

## 📁 Project Structure

```
employee_attrition/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md
├── src/
│   ├── data_generator.py     # Sample HR dataset generator
│   ├── preprocessor.py       # Data cleaning & feature engineering
│   ├── models.py             # ML training & evaluation
│   ├── explainer.py          # SHAP explainability
│   ├── visualizations.py     # All Plotly charts
│   ├── hr_assistant.py       # AI HR assistant (Claude)
│   └── report_generator.py   # Excel/CSV report generation
├── models/                   # Saved model artifacts
├── data/                     # Sample datasets
└── reports/                  # Generated reports
```

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit 1.35+ |
| Data | Pandas, NumPy |
| ML Models | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Visualization | Plotly |
| Class Balancing | imbalanced-learn (SMOTE) |
| AI Assistant | Anthropic Claude API |
| Reports | openpyxl (Excel) |

---

## 📊 Dataset

The system works with any HR dataset containing these key columns:

| Column | Type | Description |
|--------|------|-------------|
| Attrition | categorical | Yes/No — target variable |
| Age | numeric | Employee age |
| Department | categorical | Business department |
| MonthlyIncome | numeric | Monthly salary |
| JobSatisfaction | ordinal | 1–4 scale |
| OverTime | categorical | Yes/No |
| YearsAtCompany | numeric | Tenure |
| ... | ... | See preprocessor.py for full schema |

**Built-in sample data** generates 1,470 realistic IBM-style HR records if no file is uploaded.

---

## 🔬 Model Performance (Sample Data)

| Model | ROC-AUC | F1-Score | Accuracy |
|-------|---------|----------|----------|
| XGBoost | ~0.87 | ~0.68 | ~0.88 |
| Random Forest | ~0.85 | ~0.65 | ~0.87 |
| Logistic Regression | ~0.80 | ~0.58 | ~0.83 |

---

## 🧩 Architecture

```
Upload CSV/Excel
      ↓
HRDataPreprocessor
  ├── validate_dataframe()
  ├── clean_data()          # impute, dedupe, clip outliers
  ├── engineer_features()   # IncomeToLevelRatio, BurnoutRisk, etc.
  └── encode_and_scale()    # LabelEncoder + StandardScaler
      ↓
AttritionModelTrainer
  ├── SMOTE balancing
  ├── train_all()           # LR + RF + XGBoost
  └── evaluate()            # metrics, ROC, confusion matrix
      ↓
AttritionExplainer (SHAP)
  ├── explain_global()      # feature importance
  └── explain_individual()  # per-employee waterfall
      ↓
Streamlit UI
  ├── EDA Dashboard
  ├── Model Comparison
  ├── Individual Predictions
  ├── AI HR Assistant
  └── Excel/CSV Reports
```

---

## 🤝 Contributing

Pull requests welcome! Please read `CONTRIBUTING.md` and open an issue first.

---

## 📄 License

MIT License — see `LICENSE` for details.
