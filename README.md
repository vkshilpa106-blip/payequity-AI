# PayEquity AI
### Compensation Gap Analysis for HR Decision-Making

A data analytics project that investigates whether a company's gender pay gap comes from unequal pay, or from something else entirely — and shows the difference using statistics, not assumptions.

🔗 **Live app:** https://payequity-ai-fkqnqqebhlappxqp787yjov.streamlit.app/

---

## The Question

Pay gap headlines usually report one number: the average difference between what men and women earn. But that single number hides an important distinction.

There are two very different explanations for a raw pay gap:

- **Unequal pay** — the same role, the same experience, paid differently.
- **Unequal representation** — different groups simply concentrated in different roles, at different pay levels.

These two problems need completely different fixes. This project uses regression and statistical testing to work out which one is actually happening.

---

## Dataset

**Source:** [Glassdoor Gender Pay Gap dataset](https://www.kaggle.com/datasets/nilimajauhari/glassdoor-analyze-gender-pay-gap) (Kaggle)

- 1,000 employee records
- 9 features: Gender, Base Pay, Bonus, Age, Seniority, Performance Rating, Education, Department, Job Title
- Public and fully anonymized — no real individuals involved
- Clean: no missing values, no placeholder entries

A privacy rule is also applied throughout the analysis: any group smaller than 5 employees is automatically excluded from department-level results, to avoid indirectly exposing individual pay.

---

## Method

The analysis moves through five stages, each one adding more control for legitimate factors:

1. **Raw gap** — simple average difference in pay between genders, no adjustments
2. **Significance test** — a Welch's t-test to check whether that raw gap is statistically real, or just noise
3. **Partial regression** — adjusting for age and seniority only
4. **Full regression** — adjusting for age, seniority, education, department, and job title
5. **Random Forest cross-check** — a second model used to confirm which factors actually predict pay, as a sanity check on the regression result

Each department is also tested separately, not just the company as a whole, so the conclusion isn't hidden behind an overall average.

---

## What the Data Showed

| Stage | Gap | P-value | Statistically significant? |
|---|---|---|---|
| Raw gap | $8,514.73 | 8.72 × 10⁻⁸ | Yes |
| Adjusted for age + seniority | $10,112.42 | 7.85 × 10⁻²⁴ | Yes |
| Fully adjusted (+ department, job title, education) | $777.79 | 0.28 | No |

The gap didn't shrink right away — it grew after the first adjustment. That happened because women in this dataset happened to have slightly better age and seniority profiles on average, which was quietly masking the underlying gap in the raw numbers.

Once department and job title were added, though, the gap dropped sharply and stopped being statistically significant. In plain terms: once you compare people doing similar work, in similar roles, the pay difference mostly disappears.

This pattern held up separately in every department tested — Operations, Management, Administration, Sales, and Engineering all returned the same result.

**Takeaway:** the raw gap in this dataset looks like it comes from where men and women are concentrated across roles and departments, not from unequal pay for the same job.

---

## Tools Used

| Purpose | Tools |
|---|---|
| Data cleaning & analysis | Python, Pandas, NumPy |
| Statistics | SciPy, statsmodels |
| Machine learning cross-check | scikit-learn (Random Forest) |
| Charts | Plotly |
| Interactive app | Streamlit |
| Executive dashboard | Power BI |

---

## What's in This Repository

```
payequity-AI/
├── Notebook/
│   └── PE_eda_1.ipynb          full analysis: cleaning, testing, regression
├── data/
│   └── Glassdoor_Gender_Pay_Gap.csv
├── app.py                       Streamlit app (deployed live)
├── Pay_equity.pbix               Power BI dashboard
├── requirements.txt
└── README.md
```

---

## Running It Yourself

```bash
git clone https://github.com/vkshilpa106-blip/payequity-AI.git
cd payequity-AI
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. Or just use the live version linked at the top of this file.

---

## The Streamlit App

The app lets you explore the same analysis interactively:

- Filter by department and watch every number update
- See the raw gap, adjusted gap, and significance result side by side
- A traffic-light status (pass / warning / fail) based on the p-value, not just the raw number
- A "what-if" simulator estimating the cost of closing part of the pay gap
- A Random Forest chart showing which factors actually predict pay
- Export a short audit report as a text file

---

## Limitations, Honestly

- 1,000 employees is a small sample compared to a real company audit — enough to detect a pattern this size, but not enough to catch smaller, subtler gaps
- The dataset only has 9 features. Things like performance history over time, negotiation outcomes, or promotion history aren't included, and could tell a different part of the story
- This shows *association* after controlling for known factors — it doesn't prove cause and effect
- This is a portfolio project built on public data, not a certified compliance audit

---

## What I'd Add Next

- A companion tool for quick, ad-hoc questions about the data
- Plain-language summaries generated automatically per department
- An always-live cloud version of the dashboard, updated automatically as data changes

---

## About

Built as a final project for the AI Data Analytics Bootcamp, applying statistics and Python to a real HR analytics question.

**Shilpa Vellore Krishnmurthy**
