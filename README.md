# \# PayEquity AI — Compensation Gap Intelligence Platform

# 

# An end-to-end data analytics project that investigates whether a statistically significant, unexplained gender pay gap exists within an organization, after controlling for legitimate business factors such as role, department, education, age, and seniority.

# 

# \## The Problem

# 

# Raw pay gaps between genders are often reported at face value, but a raw average difference can be misleading — it may simply reflect that one group is more concentrated in certain roles, departments, or seniority levels, rather than unequal pay for equal work. This project builds a statistically grounded methodology to separate the two, and presents the result through an interactive dashboard aimed at HR leadership.

# 

# \## Dataset

# 

# \*\*Source:\*\* \[Glassdoor Gender Pay Gap dataset (Kaggle)](https://www.kaggle.com/datasets/nilimajauhari/glassdoor-analyze-gender-pay-gap)

# 

# Public, anonymized dataset containing: `JobTitle`, `Gender`, `Age`, `PerfEval`, `Education`, `Dept`, `Seniority`, `BasePay`, `Bonus`. No real individuals are represented.

# 

# \## Methodology

# 

# The analysis proceeds in layers, each adding more statistical control:

# 

# 1\. \*\*Raw Pay Gap\*\* — simple mean difference in `BasePay` between genders, no adjustment.

# 2\. \*\*Partial Adjustment\*\* — OLS regression controlling only for `Age` and `Seniority`.

# 3\. \*\*Full Adjustment\*\* — OLS regression controlling for `Age`, `Seniority`, `Education`, `Department`, and `Job Title`.

# 4\. \*\*Robustness Check\*\* — a Random Forest model is trained on the same features to rank which variables most strongly predict `BasePay`, as a cross-check against the linear regression result.

# 5\. \*\*Traffic-Light Risk Scoring\*\* — each result is classified as:

# &#x20;  - 🟢 \*\*Green\*\* — not statistically significant (p ≥ 0.05)

# &#x20;  - 🟡 \*\*Amber\*\* — borderline significance (0.05 ≤ p < 0.10)

# &#x20;  - 🔴 \*\*Red\*\* — statistically significant unexplained gap (p < 0.05)

# 6\. \*\*Privacy Suppression\*\* — any gender subgroup with fewer than 5 employees is automatically masked in the dashboard, to prevent re-identification in small teams (a k-anonymity style safeguard).

# 

# \## Key Findings (full dataset)

# 

# | Stage | Estimated Gap | p-value | Significant? |

# |---|---|---|---|

# | Raw gap | \~$8,514.73 (men higher) | — | — |

# | Adjusted for Age + Seniority only | \~$10,112.41 (men higher) | < 0.001 | Yes |

# | Fully adjusted (+ Education, Dept, Job Title) | \~$777.79 (men higher) | 0.28 | No |

# 

# \*\*Interpretation:\*\* The raw gap initially appears to widen once age and seniority are controlled for, because women in this dataset have slightly more favorable baseline seniority/age profiles. However, once job title and department are also controlled for, the remaining gap becomes small and not statistically significant.

# 

# This suggests the raw pay gap in this dataset is better explained by \*representation\* — men are disproportionately concentrated in higher-paying departments and roles — rather than unequal pay for the same role. This is a hypothesis suggested by this specific dataset, not a general claim about pay equity in all organizations.

# 

# \## Tech Stack

# 

# \- \*\*Python\*\* — Pandas, NumPy

# \- \*\*Statistics\*\* — SciPy, statsmodels (OLS regression, hypothesis testing)

# \- \*\*Machine Learning\*\* — scikit-learn (Random Forest)

# \- \*\*Dashboard\*\* — Streamlit

# \- \*\*Version control\*\* — Git / GitHub

# 

# \## How to Run Locally

# 

# ```bash

# git clone https://github.com/vkshilpa106-blip/payequity-AI.git

# cd payequity-AI

# pip install -r requirements.txt

# streamlit run app.py

# ```

# 

# The app will open at `http://localhost:8501`.

# 

# \## Project Structure

# 

# ```

# payequity-AI/

# ├── data/

# │   └── Glassdoor\_Gender\_Pay\_Gap.csv

# ├── Notebook/

# │   └── 01\_eda.ipynb

# ├── app.py

# ├── requirements.txt

# └── README.md

# ```

# 

# \## Ethical Considerations

# 

# \- This project uses a \*\*public, anonymized dataset\*\* and does not process any real employee data.

# \- Findings describe \*\*statistical association\*\*, not causation — the analysis cannot determine intent or legal liability.

# \- The dashboard suppresses any group smaller than 5 employees to protect against re-identification.

# \- The "AI Executive Summary" in the current version is a \*\*rule-based text template\*\*, not an LLM-generated output. This is disclosed transparently in the app itself.

# 

# \## MVP vs. Planned Extensions

# 

# \*\*Delivered (MVP):\*\*

# \- Cleaned dataset, EDA

# \- Raw and adjusted pay gap analysis via OLS regression

# \- Random Forest robustness check

# \- Traffic-light significance scoring

# \- Interactive Streamlit dashboard with department-level filtering

# \- Privacy suppression for small subgroups

# 

# \*\*Planned for Version 2:\*\*

# \- SQL backend (structured Employees/Salaries tables with real queries)

# \- Additional ML model comparison (e.g. Gradient Boosting)

# \- Real LLM-generated plain-language executive summaries

# \- Power BI executive dashboard

# \- Cloud deployment

# 

# \## Author

# 

# Built as a final project for \[Bootcamp Name], applying Python, statistics, and machine learning to a real-world HR analytics problem.

