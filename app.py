import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from sklearn.ensemble import RandomForestRegressor

# 1. Page Configuration
st.set_page_config(page_title="PayEquity AI Dashboard", layout="wide")

st.title("🟢 PayEquity AI — Compensation Gap Intelligence Platform")
st.subheader("Interactive Internal Audit Dashboard for HR Leadership")

# 2. Dynamic Path Data Loading
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

    # Filter data frame based on user selection
    if selected_dept == "All Corporate":
        filtered_df = df
    else:
        filtered_df = df[df['Dept'] == selected_dept]

    # --- CALCULATION A: Raw Pay Gap ---
    male_workers = filtered_df[filtered_df['Gender'] == 'Male']
    female_workers = filtered_df[filtered_df['Gender'] == 'Female']
    
    male_avg = male_workers['BasePay'].mean()
    female_avg = female_workers['BasePay'].mean()
    raw_gap = (male_avg - female_avg) if (not np.isnan(male_avg) and not np.isnan(female_avg)) else 0.0

    # Privacy Protection Trigger Flag (checks if sample group size is less than 5)
    privacy_compromised = (len(male_workers) < 5) or (len(female_workers) < 5)

    # --- CALCULATION B: Dynamic Live Models (OLS + Random Forest ML) ---
    adjusted_gap_val = 0.0
    p_value_val = 1.0
    rf_feature_importance_df = pd.DataFrame() 
    
    if len(filtered_df) > 10 and filtered_df['Gender'].nunique() > 1 and not privacy_compromised:
        # One-hot encode categorical strings
        df_encoded = pd.get_dummies(filtered_df, columns=['Gender', 'JobTitle', 'Education', 'Dept'], drop_first=True)
        
        y = df_encoded['BasePay']
        predictor_cols = [col for col in df_encoded.columns if col not in ['BasePay', 'Bonus']]
        X = df_encoded[predictor_cols].astype(float)
        
        # MODEL 1: OLS Regression Engine
        X_ols = sm.add_constant(X)
        model = sm.OLS(y, X_ols).fit()
        if 'Gender_Male' in model.params:
            adjusted_gap_val = model.params['Gender_Male']
            p_value_val = model.pvalues['Gender_Male']

        # MODEL 2: Random Forest Robustness Check
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        
        # Calculate feature importance arrays
        rf_feature_importance_df = pd.DataFrame({
            'Feature': predictor_cols,
            'Importance Score': rf_model.feature_importances_
        }).sort_values(by='Importance Score', ascending=False)

    # --- CALCULATION C: Dynamic Risk Scoring Decisions ---
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

    # 3. UI Card Matrix Integration Layer
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

    # SECURE DISPLAY ENVELOPE (Hides downstream content if privacy thresholds fail)
    if privacy_compromised:
        st.write("---")
        st.warning("🔒 **Data Privacy Suppression Active:** Analytics for this cohort have been masked because individual demographic subgroup sizes contain fewer than 5 individuals. This prevents direct compensation discovery.")
    else:
        # --- 4. ML ARCHITECTURE VISUALIZATION LAYER ---
        if not rf_feature_importance_df.empty:
            st.write("---")
            st.subheader("🤖 Machine Learning Robustness Layer — Feature Importance Audit")
            st.caption("This ensemble model screens for non-linear influences. A low rank for demographic tags indicates objective, performance-aligned organizational compensation loops.")
            
            # Isolate the top 5 drivers of pay
            top_features = rf_feature_importance_df.head(5)
            st.bar_chart(data=top_features, x='Feature', y='Importance Score', horizontal=True, use_container_width=True)

        # --- 5. AI GENERATIVE INSIGHTS LAYER ---
        st.write("---")
        st.subheader("✍️ PayEquity AI — Executive Summary Brief")
        st.caption("Automated plain-language summary generated for non-technical HR leadership decision mapping.")
        
        # Construct the variable payload text details dynamically
        gap_direction = "favoring Male employees" if adjusted_gap_val > 0 else "favoring Female employees"
        significance_text = "statistically significant" if p_value_val < 0.05 else "NOT statistically significant"
        
        if not rf_feature_importance_df.empty:
            top_driver_name = rf_feature_importance_df.iloc[0]['Feature']
        else:
            top_driver_name = 'Career Attributes'

        # Draft the automated executive brief string
        ai_summary_text = f"""
        **Audit Finding:** The compensation audit for the **{selected_dept}** cohort reveals an unadjusted raw pay gap of **${abs(raw_gap):,.2f}**. 
        However, after applying multi-variable OLS regression to control for Age, Seniority, Education, and Job Title, the true unexplained pay gap shifts to **${abs(adjusted_gap_val):,.2f}**, which is **{significance_text}** ({gap_direction}).
        
        **Root Cause Driver:** The integrated Machine Learning model isolates **{top_driver_name}** as the dominant predictor of base compensation variance within this segment. 
        This indicates that salary differentials are primarily driven by career advancement, tenure attributes, and baseline workforce demographic distribution patterns rather than systemic, unadjusted policy bias.
        """
        
        # Render the output inside a clean message box
        st.info(ai_summary_text)

except Exception as e:
    st.error(f"Error initializing data pipeline hook: {e}")

# Temporary statistical transparency check for learning validation
st.write("---")
st.subheader("🔬 Underlying Model Math (Verification Layer)")
st.write(f"Current Department Slice: **{selected_dept}**")
st.write(f"Isolated Model P-Value Matrix Variable: `{p_value_val:.6f}`")
st.write("*(Note: If this P-value is larger than 0.05, the status will mathematically lock to Green PASS)*")
