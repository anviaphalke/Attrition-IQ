"""
Sample HR Dataset Generator
Generates realistic IBM-style HR attrition data for demo purposes.
"""

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_hr_dataset(n_employees: int = 1470, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic HR attrition dataset inspired by the IBM HR Analytics dataset.
    
    Args:
        n_employees: Number of employee records to generate
        random_state: Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Complete HR dataset
    """
    np.random.seed(random_state)
    
    departments = ["Sales", "Research & Development", "Human Resources"]
    dept_weights = [0.35, 0.55, 0.10]
    
    job_roles = {
        "Sales": ["Sales Executive", "Sales Representative", "Manager"],
        "Research & Development": [
            "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative", "Manager"
        ],
        "Human Resources": ["Human Resources", "Manager"]
    }
    
    education_fields = ["Life Sciences", "Medical", "Marketing",
                        "Technical Degree", "Human Resources", "Other"]
    
    # Generate base employee data
    department = np.random.choice(departments, n_employees, p=dept_weights)
    
    job_role = []
    for dept in department:
        roles = job_roles[dept]
        job_role.append(np.random.choice(roles))
    
    age = np.random.normal(37, 9, n_employees).clip(18, 60).astype(int)
    
    # Correlated features
    years_at_company = np.random.exponential(7, n_employees).clip(0, 40).astype(int)
    years_in_current_role = np.minimum(
        years_at_company,
        np.random.exponential(4, n_employees).clip(0, 18).astype(int)
    )
    years_since_last_promotion = np.minimum(
        years_at_company,
        np.random.exponential(3, n_employees).clip(0, 15).astype(int)
    )
    years_with_curr_manager = np.minimum(
        years_at_company,
        np.random.exponential(4, n_employees).clip(0, 17).astype(int)
    )
    
    # Monthly income based on job role and years
    base_income = {
        "Sales Executive": 6000, "Sales Representative": 3000, "Manager": 17000,
        "Research Scientist": 5000, "Laboratory Technician": 3500,
        "Manufacturing Director": 14000, "Healthcare Representative": 6000,
        "Human Resources": 4000,
    }
    
    monthly_income = np.array([
        base_income.get(role, 5000) * np.random.uniform(0.7, 1.5) + years_at_company[i] * 100
        for i, role in enumerate(job_role)
    ]).clip(1009, 20000).astype(int)
    
    # Satisfaction scores (1-4 scale)
    job_satisfaction = np.random.choice([1, 2, 3, 4], n_employees, p=[0.12, 0.20, 0.30, 0.38])
    environment_satisfaction = np.random.choice([1, 2, 3, 4], n_employees, p=[0.10, 0.18, 0.35, 0.37])
    relationship_satisfaction = np.random.choice([1, 2, 3, 4], n_employees, p=[0.08, 0.20, 0.37, 0.35])
    work_life_balance = np.random.choice([1, 2, 3, 4], n_employees, p=[0.05, 0.22, 0.60, 0.13])
    
    # Performance and training
    performance_rating = np.random.choice([1, 2, 3, 4], n_employees, p=[0.01, 0.03, 0.85, 0.11])
    training_times_last_year = np.random.choice(range(7), n_employees,
                                                 p=[0.06, 0.20, 0.29, 0.26, 0.11, 0.06, 0.02])
    
    # Other features
    education = np.random.choice([1, 2, 3, 4, 5], n_employees, p=[0.12, 0.19, 0.32, 0.28, 0.09])
    education_field = np.random.choice(education_fields, n_employees,
                                        p=[0.41, 0.27, 0.11, 0.09, 0.07, 0.05])
    
    gender = np.random.choice(["Male", "Female"], n_employees, p=[0.60, 0.40])
    marital_status = np.random.choice(["Single", "Married", "Divorced"], n_employees,
                                       p=[0.32, 0.46, 0.22])
    
    num_companies_worked = np.random.choice(range(10), n_employees,
                                             p=[0.10, 0.30, 0.20, 0.15, 0.10, 0.07, 0.04, 0.02, 0.01, 0.01])
    
    total_working_years = (years_at_company + np.random.randint(0, 10, n_employees)).clip(0, 40)
    
    distance_from_home = np.random.exponential(9, n_employees).clip(1, 29).astype(int)
    
    percent_salary_hike = np.random.normal(15, 3, n_employees).clip(11, 25).astype(int)
    
    over_time = np.random.choice(["Yes", "No"], n_employees, p=[0.29, 0.71])
    
    business_travel = np.random.choice(
        ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
        n_employees, p=[0.19, 0.71, 0.10]
    )
    
    stock_option_level = np.random.choice([0, 1, 2, 3], n_employees, p=[0.47, 0.36, 0.12, 0.05])
    
    # Compute attrition probability based on risk factors
    attrition_score = np.zeros(n_employees)
    
    # High risk factors
    attrition_score += (over_time == "Yes") * 0.30
    attrition_score += (job_satisfaction == 1) * 0.25
    attrition_score += (marital_status == "Single") * 0.15
    attrition_score += (years_at_company < 3) * 0.20
    attrition_score += (years_since_last_promotion > 5) * 0.10
    attrition_score += (business_travel == "Travel_Frequently") * 0.10
    attrition_score += (environment_satisfaction == 1) * 0.15
    attrition_score += (work_life_balance == 1) * 0.10
    attrition_score += (stock_option_level == 0) * 0.08
    attrition_score += (distance_from_home > 20) * 0.05
    attrition_score += (num_companies_worked > 5) * 0.08
    attrition_score += (department == "Sales") * 0.05
    
    # Protective factors
    attrition_score -= (years_at_company > 10) * 0.15
    attrition_score -= (stock_option_level > 1) * 0.10
    attrition_score -= (job_satisfaction == 4) * 0.10
    
    # Convert to probability
    attrition_prob = 1 / (1 + np.exp(-2 * (attrition_score - 0.5)))
    attrition = (np.random.random(n_employees) < attrition_prob).astype(int)
    
    # Build DataFrame
    df = pd.DataFrame({
        "EmployeeID": range(1001, 1001 + n_employees),
        "Age": age,
        "Attrition": ["Yes" if a == 1 else "No" for a in attrition],
        "BusinessTravel": business_travel,
        "Department": department,
        "DistanceFromHome": distance_from_home,
        "Education": education,
        "EducationField": education_field,
        "EnvironmentSatisfaction": environment_satisfaction,
        "Gender": gender,
        "JobInvolvement": np.random.choice([1, 2, 3, 4], n_employees, p=[0.06, 0.20, 0.59, 0.15]),
        "JobLevel": np.random.choice([1, 2, 3, 4, 5], n_employees, p=[0.26, 0.37, 0.22, 0.10, 0.05]),
        "JobRole": job_role,
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_status,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": np.random.randint(2094, 26999, n_employees),
        "NumCompaniesWorked": num_companies_worked,
        "OverTime": over_time,
        "PercentSalaryHike": percent_salary_hike,
        "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times_last_year,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager,
    })
    
    logger.info(f"Generated dataset: {len(df)} employees, "
                f"{df['Attrition'].eq('Yes').sum()} attrition cases "
                f"({df['Attrition'].eq('Yes').mean()*100:.1f}%)")
    
    return df


if __name__ == "__main__":
    df = generate_hr_dataset()
    DATA_DIR = Path(__file__).resolve().parent.parent / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / "sample_hr_data.csv", index=False)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Attrition rate: {df['Attrition'].eq('Yes').mean()*100:.1f}%")
