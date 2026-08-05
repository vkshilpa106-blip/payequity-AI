import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

st.set_page_config(page_title="PayEquity AI Dashboard", layout="wide")

st.title("🟢 PayEquity AI — Compensation Gap Intelligence Platform")
st.subheader("Interactive Internal Audit Dashboard for HR Leadership")

base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "data", "Glassdoor Gender Pay Gap.csv")

@st.cache_data
def load_project_data(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return pd.read_csv("data/Glassdoor Gender Pay Gap.csv")

try:
    df = load_project_data(data_path)
    
    st.sidebar.header("Filter Analytics Layer")
    departments = ["All Corporate"] + list(df['Dept'].unique())
    selected_dept = st.sidebar.selectbox("Select Corporate Department", departments)

    # 1. Filter structural data frame slice based on selection
    if selected_dept == "All Corporate":
        filtered_df = df
    else:
        filtered_df = df[df['Dept'] == selected_dept]

    # 2. CALCULATION A: Raw Pay Gap
    male_avg = filtered_df[filtered_df['Gender'] == 'Male']['BasePay'].mean()
    female_avg = filtered_df[filtered_df['Gender'] == 'Female']['BasePay'].mean()
    raw_gap = (male_avg - female_avg) if (not np.isnan(male_avg) and not np.isnan(female_avg)) else 0.0

    # 3. CALCULATION B: Dynamic Live Regression
    adjusted_gap_val = 0.0
    p_value_val = 1.0
    
    # Check safety threshold to make sure we have enough data points to run regression
    if len(filtered_df) > 10 and filtered_df['Gender'].nunique() > 1:
        # One-hot encode our text variables within the selected slice
        # We explicitly drop first column to isolate pure baseline references
        df_encoded = pd.get_dummies(filtered_df, columns=['Gender', 'JobTitle', 'Education', 'Dept'], drop_first=True)
        
        # Pull targets and isolate valid predictors
        y = df_encoded['BasePay']
        predictor_cols = [col for col in df_encoded.columns if col not in ['BasePay', 'Bonus']]
        
        X = df_encoded[predictor_cols].astype(float)
        X = sm.add_constant(X)
        
        # Fit our localized OLS pipeline model
        model = sm.OLS(y, X).fit()
        
        # Check if Gender_Male exists in our sliced columns and isolate it
        if 'Gender_Male' in model.params:
            adjusted_gap_val = model.params['Gender_Male']
            p_value_val = model.pvalues['Gender_Male']

    # 4. CALCULATION C: Dynamic Risk Scoring Decisions
    if p_value_val < 0.05 and adjusted_gap_val > 0:
        status_lbl = "FAIL (Red)"
        status_color = "normal"
        risk_caption = f"Significant liability risk observed (p={p_value_val:.4f})"
    elif p_value_val < 0.10 and adjusted_gap_val > 0:
        status_lbl = "WARNING (Amber)"
        status_color = "off"
        risk_caption = "Marginal or borderline significance profile discovered."
    else:
        status_lbl = "PASS (Green)"
        status_color = "inverse"
        risk_caption = "No statistically significant unexplained gap found."

    # 5. UI Card Matrix Integration Layer
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label=f"Raw Pay Gap ({selected_dept})", 
            value=f"${raw_gap:,.2f}", 
            delta=f"{(raw_gap / male_avg * 100):.1f}% Variance" if male_avg > 0 else "0%"
        )
        st.caption("Unadjusted absolute average difference between genders")

    with col2:
        st.metric(
            label=f"Fully-Adjusted Pay Gap", 
            value=f"${adjusted_gap_val:,.2f}", 
            delta="Significant" if p_value_val < 0.05 else "Not Significant",
            delta_color="normal" if p_value_val < 0.05 else "inverse"
        )
        st.caption("Controlled for all concurrent background attributes via live OLS")

    with col3:
        st.metric(label="Platform Audit Status", value=status_lbl, delta=status_lbl, delta_color=status_color)
        st.caption(risk_caption)

    st.divider()
    st.write(f"Showing localized metric matrix breakdown for exactly **{len(filtered_df)} employees** in {selected_dept}.")
    st.dataframe(filtered_df.head(10), use_container_width=True)

except Exception as e:
    st.error(f"Error initializing data pipeline hook: {e}")
# Temporary statistical transparency check for learning validation
st.subheader("🔬 Underlying Model Math (Verification Layer)")
st.write(f"Current Department Slice: **{selected_dept}**")
st.write(f"Isolated Model P-Value Matrix Variable: `{p_value_val:.6f}`")
st.write("*(Note: If this P-value is larger than 0.05, the status will mathematically lock to Green PASS)*")

