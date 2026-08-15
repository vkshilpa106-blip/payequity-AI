import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="PayEquity AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. THEME — AI / TECH DARK MODE (violet → cyan gradient system)
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {
        --bg: #0B0E17;
        --bg-soft: #10141F;
        --card: #151A27;
        --card-border: rgba(139, 92, 246, 0.18);
        --violet: #8B5CF6;
        --cyan: #22D3EE;
        --text: #E8EAF0;
        --muted: #8B93A8;
        --green: #34D399;
        --amber: #FBBF24;
        --red: #F87171;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: radial-gradient(circle at 15% 0%, #161233 0%, var(--bg) 45%),
                    radial-gradient(circle at 100% 100%, #0E2233 0%, var(--bg) 55%),
                    var(--bg);
        color: var(--text);
    }

    section[data-testid="stSidebar"] {
        background: var(--bg-soft);
        border-right: 1px solid rgba(139, 92, 246, 0.15);
    }
    section[data-testid="stSidebar"] * { color: var(--text) !important; }

    .hero-title {
        font-size: 2.4rem; font-weight: 800; margin-bottom: 0.1rem;
        background: linear-gradient(90deg, #A78BFA, #22D3EE);
        -webkit-background-clip: text; background-clip: text; color: transparent;
        letter-spacing: -0.5px;
    }
    .hero-sub { color: var(--muted); font-size: 1rem; font-weight: 500; margin-bottom: 1.2rem; }
    .badge-row { display:flex; gap:8px; margin-bottom: 1.4rem; flex-wrap: wrap; }
    .tech-badge {
        display:inline-flex; align-items:center; gap:6px;
        background: rgba(139,92,246,0.10); border: 1px solid rgba(139,92,246,0.3);
        color: #C4B5FD; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.3px;
        padding: 4px 12px; border-radius: 20px;
    }

    .kpi-card {
        background: linear-gradient(160deg, var(--card) 0%, rgba(21,26,39,0.6) 100%);
        border: 1px solid var(--card-border);
        border-radius: 16px; padding: 20px 22px; height: 100%;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        position: relative; overflow: hidden;
    }
    .kpi-card::before {
        content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, var(--violet), var(--cyan));
    }
    .kpi-card.status-red::before { background: var(--red); }
    .kpi-card.status-amber::before { background: var(--amber); }
    .kpi-card.status-green::before { background: var(--green); }
    .kpi-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.6px; color: var(--muted); font-weight: 600; margin-bottom: 8px; }
    .kpi-value { font-size: 1.9rem; font-weight: 700; color: var(--text); line-height: 1.1; }
    .kpi-caption { font-size: 0.78rem; color: var(--muted); margin-top: 8px; }
    .kpi-delta { font-size: 0.82rem; font-weight: 700; margin-top: 4px; display:inline-block; }
    .delta-red { color: var(--red); }
    .delta-amber { color: var(--amber); }
    .delta-green { color: var(--green); }

    .status-pill { display:inline-flex; align-items:center; gap:6px; font-size: 0.95rem; font-weight: 700; padding: 6px 14px; border-radius: 20px; }
    .pill-red { background: rgba(248,113,113,0.12); color: var(--red); border: 1px solid rgba(248,113,113,0.35); }
    .pill-amber { background: rgba(251,191,36,0.12); color: var(--amber); border: 1px solid rgba(251,191,36,0.35); }
    .pill-green { background: rgba(52,211,153,0.12); color: var(--green); border: 1px solid rgba(52,211,153,0.35); }

    .section-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1.2px; color: var(--cyan); font-weight: 700; margin: 1.4rem 0 0.5rem 0; }

    .ai-panel {
        background: linear-gradient(160deg, rgba(139,92,246,0.08), rgba(34,211,238,0.05));
        border: 1px solid rgba(139,92,246,0.25);
        border-radius: 16px; padding: 22px 26px; margin-top: 6px;
        font-size: 0.92rem; line-height: 1.65; color: var(--text);
    }
    .ai-panel b { color: #C4B5FD; }

    .sim-panel {
        background: linear-gradient(160deg, rgba(52,211,153,0.07), rgba(34,211,238,0.05));
        border: 1px solid rgba(52,211,153,0.25);
        border-radius: 16px; padding: 22px 26px; margin-top: 6px;
        font-size: 0.92rem; line-height: 1.65; color: var(--text);
    }

    div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid var(--card-border); }
    .stAlert { border-radius: 12px; }
    hr { border-color: rgba(139,92,246,0.15) !important; }

    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background: var(--card); border-radius: 10px 10px 0 0; padding: 8px 18px;
        border: 1px solid var(--card-border); border-bottom: none; color: var(--muted);
    }
    .stTabs [aria-selected="true"] { color: var(--cyan) !important; background: var(--bg-soft) !important; }

    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-thumb { background: rgba(139,92,246,0.3); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 3. HERO HEADER
# ============================================================
st.markdown('<div class="hero-title">PayEquity AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Compensation Gap Intelligence Platform — Interactive Audit Dashboard for HR Leadership</div>', unsafe_allow_html=True)
st.markdown("""
<div class="badge-row">
    <span class="tech-badge"> OLS Regression</span>
    <span class="tech-badge"> Random Forest Cross-Check</span>
    <span class="tech-badge"> Privacy-Safe (k ≥ 5)</span>
    <span class="tech-badge"> Live Statistical Audit</span>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 4. DATA LOADING
# ============================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "data", "Glassdoor_Gender_Pay_Gap.csv")


@st.cache_data
def load_project_data(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return pd.read_csv("data/Glassdoor_Gender_Pay_Gap.csv")


def kpi_card(label, value, caption, delta_text=None, delta_class="", status_class=""):
    delta_html = f'<div class="kpi-delta {delta_class}">{delta_text}</div>' if delta_text else ""
    st.markdown(f"""
    <div class="kpi-card {status_class}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
        <div class="kpi-caption">{caption}</div>
    </div>
    """, unsafe_allow_html=True)


# Defaults defined up front so later code never breaks on an early failure
selected_dept = "All Corporate"
p_value_val = 1.0
adjusted_gap_val = 0.0
raw_gap = 0.0
male_avg = 0.0
female_avg = 0.0

try:
    df = load_project_data(data_path)

    st.sidebar.markdown("###  Filter Analytics Layer")
    departments = ["All Corporate"] + list(df['Dept'].unique())
    selected_dept = st.sidebar.selectbox("Select Corporate Department", departments)

    st.sidebar.markdown("###  What-If Simulator")
    close_pct = st.sidebar.slider("Close the pay gap by:", min_value=0, max_value=100, value=50, step=5, format="%d%%")

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
    rf_feature_importance_df = pd.DataFrame()
    regression_error = None

    if len(filtered_df) > 10 and filtered_df['Gender'].nunique() > 1 and not privacy_compromised:
        try:
            df_encoded = pd.get_dummies(filtered_df, columns=['Gender', 'JobTitle', 'Education', 'Dept'], drop_first=True)

            y = df_encoded['BasePay']
            predictor_cols = [col for col in df_encoded.columns if col not in ['BasePay', 'Bonus']]
            X = df_encoded[predictor_cols].astype(float)

            X_ols = sm.add_constant(X)
            model = sm.OLS(y, X_ols).fit()
            if 'Gender_Male' in model.params:
                adjusted_gap_val = model.params['Gender_Male']
                p_value_val = model.pvalues['Gender_Male']

            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X, y)

            rf_feature_importance_df = pd.DataFrame({
                'Feature': predictor_cols,
                'Importance Score': rf_model.feature_importances_
            }).sort_values(by='Importance Score', ascending=False)

        except Exception as reg_err:
            regression_error = str(reg_err)

    # --- CALCULATION C: Traffic-Light Risk Scoring ---
    # Significance is checked regardless of direction (a significant female-favoring gap
    # deserves the same flag as a significant male-favoring one). Direction is reported
    # separately in the caption, not baked into the red/amber/green threshold itself.
    gap_direction_word = "Male" if adjusted_gap_val > 0 else "Female"
    significant = p_value_val < 0.05

    if p_value_val < 0.05:
        status_lbl = "FAIL"
        pill_class = "pill-red"
        card_status_class = "status-red"
        delta_class = "delta-red"
        risk_caption = f"Statistically significant gap favoring {gap_direction_word} employees (p={p_value_val:.4f})"
    elif p_value_val < 0.10:
        status_lbl = "WARNING"
        pill_class = "pill-amber"
        card_status_class = "status-amber"
        delta_class = "delta-amber"
        risk_caption = f"Borderline significance, {gap_direction_word}-favoring (p={p_value_val:.4f})"
    else:
        status_lbl = "PASS"
        pill_class = "pill-green"
        card_status_class = "status-green"
        delta_class = "delta-green"
        risk_caption = "No statistically significant unexplained gap found."

    significance_text = "statistically significant" if significant else "NOT statistically significant"
    gap_direction = "favoring Male employees" if adjusted_gap_val > 0 else "favoring Female employees"
    top_driver_name = rf_feature_importance_df.iloc[0]['Feature'] if not rf_feature_importance_df.empty else "career attributes"

    # ============================================================
    # 5. KPI CARD ROW
    # ============================================================
    st.markdown('<div class="section-label">Audit Summary</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        variance_pct = f"{(raw_gap / male_avg * 100):.1f}% variance" if male_avg > 0 else "0%"
        kpi_card(
            f"Raw Pay Gap · {selected_dept}",
            f"${raw_gap:,.2f}",
            "Unadjusted average difference between genders",
            delta_text=variance_pct, delta_class="delta-amber"
        )

    with col2:
        sig_text = "Significant" if significant else "Not Significant"
        kpi_card(
            "Fully-Adjusted Pay Gap",
            f"${adjusted_gap_val:,.2f}",
            "Controlled for role, seniority, education & department (live OLS)",
            delta_text=sig_text, delta_class=delta_class
        )

    with col3:
        st.markdown(f"""
        <div class="kpi-card {card_status_class}">
            <div class="kpi-label">Platform Audit Status</div>
            <div style="margin-top:4px;"><span class="status-pill {pill_class}">{status_lbl}</span></div>
            <div class="kpi-caption" style="margin-top:12px;">{risk_caption}</div>
        </div>
        """, unsafe_allow_html=True)

    # ============================================================
    # 6. VISUAL BREAKDOWN (Plotly)
    # ============================================================
    st.markdown('<div class="section-label">Visual Breakdown</div>', unsafe_allow_html=True)
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        gender_avg = filtered_df.groupby('Gender')['BasePay'].mean().reset_index()
        fig_gender = go.Figure(data=[
            go.Bar(
                x=gender_avg['Gender'], y=gender_avg['BasePay'],
                marker_color=['#22D3EE' if g == 'Male' else '#8B5CF6' for g in gender_avg['Gender']],
                text=[f"${v:,.0f}" for v in gender_avg['BasePay']],
                textposition='outside', textfont=dict(color='#E8EAF0', size=13)
            )
        ])
        fig_gender.update_layout(
            title=dict(text="Average Base Pay by Gender", font=dict(color='#E8EAF0', size=14)),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8B93A8'), height=320, margin=dict(t=50, b=30, l=10, r=10),
            xaxis=dict(showgrid=False, color='#E8EAF0'),
            yaxis=dict(showgrid=True, gridcolor='rgba(139,92,246,0.12)', tickprefix="$"),
            showlegend=False
        )
        st.plotly_chart(fig_gender, use_container_width=True)

    with chart_col2:
        if selected_dept == "All Corporate":
            dept_gap = df.groupby(['Dept', 'Gender'])['BasePay'].mean().reset_index()
            fig_dept = go.Figure()
            for gender, color in [('Male', '#22D3EE'), ('Female', '#8B5CF6')]:
                sub = dept_gap[dept_gap['Gender'] == gender]
                fig_dept.add_trace(go.Bar(name=gender, x=sub['Dept'], y=sub['BasePay'], marker_color=color))
            fig_dept.update_layout(
                title=dict(text="Average Pay by Department & Gender", font=dict(color='#E8EAF0', size=14)),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#8B93A8'), height=320, margin=dict(t=50, b=30, l=10, r=10),
                barmode='group', legend=dict(font=dict(color='#E8EAF0')),
                xaxis=dict(showgrid=False, color='#E8EAF0'),
                yaxis=dict(showgrid=True, gridcolor='rgba(139,92,246,0.12)', tickprefix="$")
            )
            st.plotly_chart(fig_dept, use_container_width=True)
        else:
            age_avg = filtered_df.groupby('Gender')['Age'].mean().reset_index()
            fig_age = go.Figure(data=[
                go.Bar(x=age_avg['Gender'], y=age_avg['Age'],
                       marker_color=['#22D3EE' if g == 'Male' else '#8B5CF6' for g in age_avg['Gender']],
                       text=[f"{v:.1f}" for v in age_avg['Age']], textposition='outside',
                       textfont=dict(color='#E8EAF0', size=13))
            ])
            fig_age.update_layout(
                title=dict(text=f"Average Age by Gender — {selected_dept}", font=dict(color='#E8EAF0', size=14)),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#8B93A8'), height=320, margin=dict(t=50, b=30, l=10, r=10),
                xaxis=dict(showgrid=False, color='#E8EAF0'),
                yaxis=dict(showgrid=True, gridcolor='rgba(139,92,246,0.12)'),
                showlegend=False
            )
            st.plotly_chart(fig_age, use_container_width=True)

    # ============================================================
    # 7. TABS — Records / Statistics / ML / What-If
    # ============================================================
    tab_records, tab_stats, tab_ml, tab_sim = st.tabs(
        ["Records", "Statistics", "ML Insights", "What-If Simulator"]
    )

    # ---------------- TAB: RECORDS ----------------
    with tab_records:
        st.write(f"Showing **{len(filtered_df)} employees** in **{selected_dept}**.")
        st.dataframe(filtered_df.head(10), use_container_width=True)

        if regression_error:
            st.warning(f"Regression could not be computed for this slice (likely too few employees or "
                       f"a job title/department combination with no variation). Details: {regression_error}")
        elif privacy_compromised:
            st.warning("**Data Privacy Suppression Active:** subgroup sizes are below 5 individuals, "
                       "so detailed statistics are hidden to prevent direct compensation discovery.")

        report_text = f"""PAYEQUITY AI — AUDIT REPORT
================================
Department: {selected_dept}
Employees analyzed: {len(filtered_df)}

RAW PAY GAP: ${raw_gap:,.2f}
FULLY-ADJUSTED PAY GAP: ${adjusted_gap_val:,.2f}
P-VALUE: {p_value_val:.6f}
STATUS: {status_lbl} — {risk_caption}

TOP ML PREDICTOR OF PAY: {top_driver_name}

--
Generated by PayEquity AI. Dataset: Glassdoor Gender Pay Gap (Kaggle), public & anonymized.
This report is a portfolio project output and not a substitute for a formal legal or HR compliance audit.
"""
        st.download_button(
            "Export Audit Report",
            data=report_text,
            file_name=f"payequity_audit_{selected_dept.replace(' ', '_')}.txt",
            mime="text/plain"
        )

    # ---------------- TAB: STATISTICS ----------------
    with tab_stats:
        if regression_error or privacy_compromised:
            st.info("Statistics are unavailable for this selection — see the Records tab for details.")
        else:
            st.markdown('<div class="section-label">Executive Summary</div>', unsafe_allow_html=True)
            st.caption("Automated, rule-based plain-language summary generated from the statistical model above "
                       "(template-based, not LLM-generated in this version).")

            if significant:
                conclusion_text = (
                    "This gap remains after controlling for role, department, education, age, and seniority, "
                    "and is statistically significant. This warrants a closer, human-led review of compensation "
                    "practices in this segment — it should not be dismissed as explained by other factors."
                )
            else:
                conclusion_text = (
                    "After controlling for role, department, education, age, and seniority, the remaining gap is "
                    "small enough that it could plausibly be due to random variation rather than a structural pattern. "
                    "This suggests — but does not prove — that observed raw pay differences are more associated with "
                    "role and tenure distribution than with gender itself."
                )

            st.markdown(f"""
            <div class="ai-panel">
            <b>Audit Finding:</b> The compensation audit for the <b>{selected_dept}</b> cohort reveals an unadjusted raw pay gap of
            <b>${abs(raw_gap):,.2f}</b>. After applying multi-variable OLS regression to control for Age, Seniority, Education,
            Department, and Job Title, the estimated adjusted pay gap is <b>${abs(adjusted_gap_val):,.2f}</b>, which is
            <b>{significance_text}</b> ({gap_direction}, p={p_value_val:.4f}).
            <br><br>
            <b>Interpretation:</b> The Random Forest model ranks <b>{top_driver_name}</b> as the strongest predictor of base pay
            in this segment. {conclusion_text}
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-label">🔬 Underlying Model Math</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.85rem; color:var(--muted); line-height:1.9;">
            Current Department Slice: <span style="color:var(--text);">{selected_dept}</span><br>
            Isolated Model P-Value Variable: <span style="color:var(--cyan);">{p_value_val:.6f}</span><br>
            Sample size: <span style="color:var(--text);">{len(filtered_df)} employees ({len(male_workers)} male, {len(female_workers)} female)</span><br>
            <span style="opacity:0.7;">If this P-value is ≥ 0.05, the status locks to PASS regardless of gap direction.</span>
            </div>
            """, unsafe_allow_html=True)

    # ---------------- TAB: ML INSIGHTS ----------------
    with tab_ml:
        if regression_error or privacy_compromised:
            st.info("ML insights are unavailable for this selection — see the Records tab for details.")
        elif not rf_feature_importance_df.empty:
            st.markdown('<div class="section-label">Machine Learning Robustness Layer</div>', unsafe_allow_html=True)
            st.caption("A Random Forest model cross-checks the regression result by ranking which features most "
                       "strongly predict base pay. A low rank for demographic tags suggests compensation tracks "
                       "role and tenure more than gender — though this doesn't rule out gender effects operating "
                       "through those same factors.")
            top_features = rf_feature_importance_df.head(8)
            st.bar_chart(data=top_features, x='Feature', y='Importance Score', horizontal=True, use_container_width=True)
        else:
            st.info("Not enough data in this selection to run the Random Forest robustness check.")

    # ---------------- TAB: WHAT-IF SIMULATOR ----------------
    with tab_sim:
        st.markdown('<div class="section-label">Simulate Closing the Pay Gap</div>', unsafe_allow_html=True)
        st.caption("Uses the slider in the sidebar to model the cost of closing a percentage of the raw pay gap "
                   "for the underpaid group in this selection.")

        if raw_gap == 0 or np.isnan(male_avg) or np.isnan(female_avg):
            st.info("Not enough data to run the simulator for this selection.")
        else:
            if raw_gap > 0:
                underpaid_group, underpaid_count, underpaid_avg = "Female", len(female_workers), female_avg
            else:
                underpaid_group, underpaid_count, underpaid_avg = "Male", len(male_workers), male_avg

            gap_amount = abs(raw_gap)
            per_person_raise = gap_amount * (close_pct / 100)
            total_cost = per_person_raise * underpaid_count
            new_avg = underpaid_avg + per_person_raise

            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                kpi_card("Employees Affected", f"{underpaid_count}", f"{underpaid_group} employees in {selected_dept}")
            with sc2:
                kpi_card("Raise per Employee", f"${per_person_raise:,.2f}", f"Closing {close_pct}% of the ${gap_amount:,.2f} raw gap")
            with sc3:
                kpi_card("Total Annual Cost", f"${total_cost:,.2f}", "Combined cost across affected employees")

            st.markdown(f"""
            <div class="sim-panel">
            Closing <b>{close_pct}%</b> of the raw pay gap would raise the average {underpaid_group} salary in
            <b>{selected_dept}</b> from <b>${underpaid_avg:,.2f}</b> to <b>${new_avg:,.2f}</b>,
            at a total estimated cost of <b>${total_cost:,.2f}</b> per year across {underpaid_count} employees.
            <br><br>
            <span style="color:var(--muted); font-size:0.85rem;">This is a simplified raw-gap simulation for illustration —
            a full compensation adjustment would typically target the fully-adjusted gap on a per-role basis, not a flat raise.</span>
            </div>
            """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error initializing data pipeline hook: {e}")
