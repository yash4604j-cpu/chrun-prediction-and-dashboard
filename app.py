"""
Customer Churn Risk Dashboard
------------------------------
Loads the trained XGBoost (+SMOTE) pipeline produced by train_pipeline.py
and turns raw customer records into risk-tiered, actionable output for a
non-technical business user, per Section 4 of the dissertation report.

Run with:  streamlit run app.py
Expects churn_pipeline.joblib and shap_background.joblib in the same folder.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Customer Churn Risk Dashboard", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------
# Load the trained pipeline (cached so it only loads once per session)
# ---------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    pipe = joblib.load(os.path.join(HERE, "churn_pipeline.joblib"))
    return pipe

@st.cache_resource
def load_explainer(_model):
    # Tree explainer needs no background sample for the fast tree_path_dependent method
    return shap.TreeExplainer(_model)

pipe = load_pipeline()
model = pipe['model']
scaler = pipe['scaler']
feature_cols = pipe['feature_cols']
numeric_cols = pipe['numeric_cols']
categorical_cols = pipe['categorical_cols']
meta = pipe['preproc_meta']
explainer = load_explainer(model)

RISK_ACTIONS = {
    "High":   "Immediate intervention required — high-touch retention effort (personal outreach, priority offer).",
    "Medium": "Proactive monitoring and targeted communications (check-in email, engagement nudge).",
    "Low":    "Maintain engagement — low-cost retention activities (newsletter, loyalty content).",
}

# ---------------------------------------------------------------
# Preprocessing: mirrors train_pipeline.py exactly, using the
# medians/modes/mappings learned at training time (never recomputed
# from the new data, to avoid leaking new-data statistics into scoring).
# ---------------------------------------------------------------
def preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Keep originals for display/filtering before they get engineered away
    display_cols = {}
    for c in ['gender', 'region_category', 'membership_category', 'feedback',
              'medium_of_operation', 'preferred_offer_types']:
        if c in df.columns:
            display_cols[c] = df[c].astype(str)

    # days_since_last_login anomaly
    if 'days_since_last_login' in df.columns:
        df['days_since_last_login'] = df['days_since_last_login'].replace(
            -999, meta['valid_login_median'])

    # avg_frequency_login_days
    if 'avg_frequency_login_days' in df.columns:
        raw_series = pd.to_numeric(df['avg_frequency_login_days'], errors='coerce')
        df['avg_frequency_login_days'] = raw_series.fillna(meta['median_login_days'])

    # points_in_wallet
    if 'points_in_wallet' in df.columns:
        df['points_in_wallet'] = df['points_in_wallet'].fillna(meta['median_points'])

    # region_category
    if 'region_category' in df.columns:
        df['region_category'] = df['region_category'].fillna(meta['mode_region'])

    # '?' placeholders
    for c in ['joined_through_referral', 'medium_of_operation']:
        if c in df.columns:
            df[c] = df[c].replace('?', 'Unknown')

    if 'preferred_offer_types' in df.columns:
        df['preferred_offer_types'] = df['preferred_offer_types'].fillna(meta['preferred_offer_mode'])

    # tenure_months (relative to the training-time max joining date, for consistency
    # with how the model was trained — see Section 3.2 discussion in the report)
    if 'joining_date' in df.columns:
        joining_dt = pd.to_datetime(df['joining_date'], format='%d-%m-%Y', errors='coerce')
        max_date = pd.to_datetime(meta['max_date'])
        df['tenure_months'] = ((max_date - joining_dt).dt.days / 30.4).fillna(0)
        df = df.drop(columns=['joining_date'])

    # visit_hour
    if 'last_visit_time' in df.columns:
        df['visit_hour'] = pd.to_datetime(df['last_visit_time'], format='%H:%M:%S',
                                           errors='coerce').dt.hour.fillna(12)
        df = df.drop(columns=['last_visit_time'])

    # feedback_score
    if 'feedback' in df.columns:
        df['feedback_score'] = df['feedback'].map(meta['feedback_map']).fillna(0)
        df = df.drop(columns=['feedback'])

    # active_complaint
    if 'past_complaint' in df.columns and 'complaint_status' in df.columns:
        unresolved = meta['unresolved_complaint_statuses']
        df['active_complaint'] = ((df['past_complaint'] == 'Yes') &
                                   (df['complaint_status'].isin(unresolved))).astype(int)
        df = df.drop(columns=['past_complaint', 'complaint_status'])

    # membership_score
    if 'membership_category' in df.columns:
        df['membership_score'] = df['membership_category'].map(meta['membership_map']).fillna(0)
        df = df.drop(columns=['membership_category'])

    # binary Yes/No columns
    for col in ['used_special_discount', 'offer_application_preference', 'joined_through_referral']:
        if col in df.columns:
            df[col] = (df[col] == 'Yes').astype(int)

    # drop identifiers
    df = df.drop(columns=['security_no', 'referral_id'], errors='ignore')

    # one-hot encode, then align to the exact training feature columns
    df = pd.get_dummies(df, columns=[c for c in categorical_cols if c in df.columns], drop_first=True)
    df = df.reindex(columns=feature_cols, fill_value=0)

    # scale numeric columns with the fitted scaler
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    for k, v in display_cols.items():
        df[f"__display_{k}"] = v.values

    return df

def score(df_processed: pd.DataFrame) -> pd.DataFrame:
    X = df_processed[feature_cols]
    probs = model.predict_proba(X)[:, 1]
    out = df_processed.copy()
    out['churn_probability'] = probs
    out['risk_tier'] = pd.cut(probs, bins=[-0.01, 0.4, 0.7, 1.01],
                               labels=['Low', 'Medium', 'High'])
    out['recommended_action'] = out['risk_tier'].map(RISK_ACTIONS)
    return out

# ---------------------------------------------------------------
# Sidebar — data input
# ---------------------------------------------------------------
st.sidebar.title("Customer Churn Risk Dashboard")
st.sidebar.caption(f"Champion model: {pipe['model_name']} · AUC-ROC 0.9755 on held-out test set")

data_source = st.sidebar.radio("Data source", ["Use bundled sample (300 customers)", "Upload a CSV"])

if data_source == "Upload a CSV":
    uploaded = st.sidebar.file_uploader("Upload raw customer CSV (same columns as the training dataset, minus churn_risk_score)", type="csv")
    raw_df = pd.read_csv(uploaded) if uploaded is not None else None
else:
    sample_path = os.path.join(HERE, "sample_customers.csv")
    raw_df = pd.read_csv(sample_path) if os.path.exists(sample_path) else None

if raw_df is None:
    st.info("Upload a CSV in the sidebar, or select the bundled sample, to see risk scores.")
    st.stop()

processed = preprocess(raw_df)
scored = score(processed)

# ---------------------------------------------------------------
# Filters
# ---------------------------------------------------------------
st.sidebar.header("Filters")
tier_filter = st.sidebar.multiselect("Risk tier", ['High', 'Medium', 'Low'], default=['High', 'Medium', 'Low'])

filt = scored[scored['risk_tier'].isin(tier_filter)]

if '__display_membership_category' in filt.columns:
    mem_opts = sorted(filt['__display_membership_category'].dropna().unique().tolist())
    mem_filter = st.sidebar.multiselect("Membership level", mem_opts, default=mem_opts)
    filt = filt[filt['__display_membership_category'].isin(mem_filter)]

if '__display_region_category' in filt.columns:
    reg_opts = sorted(filt['__display_region_category'].dropna().unique().tolist())
    reg_filter = st.sidebar.multiselect("Region", reg_opts, default=reg_opts)
    filt = filt[filt['__display_region_category'].isin(reg_filter)]

if '__display_feedback' in filt.columns:
    fb_opts = sorted(filt['__display_feedback'].dropna().unique().tolist())
    fb_filter = st.sidebar.multiselect("Feedback category", fb_opts, default=fb_opts)
    filt = filt[filt['__display_feedback'].isin(fb_filter)]

# ---------------------------------------------------------------
# Summary KPIs
# ---------------------------------------------------------------
st.title("Customer Churn Risk Dashboard")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers shown", len(filt))
c2.metric("High-risk customers", int((filt['risk_tier'] == 'High').sum()))
c3.metric("Average risk score", f"{filt['churn_probability'].mean():.2f}" if len(filt) else "—")
c4.metric("% High risk", f"{(filt['risk_tier'] == 'High').mean()*100:.1f}%" if len(filt) else "—")

# ---------------------------------------------------------------
# Risk distribution
# ---------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Risk Distribution")
    tier_counts = filt['risk_tier'].value_counts().reindex(['High', 'Medium', 'Low']).fillna(0)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(tier_counts, labels=tier_counts.index, autopct='%1.0f%%',
           colors=['#d62728', '#ff7f0e', '#2ca02c'])
    st.pyplot(fig)
with col2:
    st.subheader("Feature Importance (Champion Model)")
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10)
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    importances.sort_values().plot(kind='barh', ax=ax2, color='steelblue')
    ax2.set_xlabel("Importance")
    st.pyplot(fig2)

# ---------------------------------------------------------------
# Customer risk ranking table
# ---------------------------------------------------------------
st.subheader("Customer Risk Ranking")
display_cols_order = [c for c in filt.columns if c.startswith('__display_')]
show_df = filt[display_cols_order + ['churn_probability', 'risk_tier', 'recommended_action']].copy()
show_df.columns = [c.replace('__display_', '') for c in show_df.columns]
show_df = show_df.sort_values('churn_probability', ascending=False)
st.dataframe(show_df, use_container_width=True)

st.download_button("Download filtered results as CSV",
                    show_df.to_csv(index=True).encode('utf-8'),
                    file_name="churn_risk_export.csv", mime="text/csv")

# ---------------------------------------------------------------
# Individual customer explainability (SHAP)
# ---------------------------------------------------------------
st.subheader("Individual Customer Explanation (SHAP)")
if len(show_df):
    selected_idx = st.selectbox("Select a customer (row index) to explain", show_df.index.tolist())
    row = filt.loc[[selected_idx], feature_cols]
    sv = explainer.shap_values(row)
    contrib = pd.Series(sv[0], index=feature_cols).sort_values(key=abs, ascending=False).head(6)
    fig3, ax3 = plt.subplots(figsize=(6, 3.5))
    colors = ['#d62728' if v > 0 else '#1f77b4' for v in contrib.values]
    contrib.sort_values().plot(kind='barh', ax=ax3, color=colors[::-1])
    ax3.set_xlabel("SHAP value (red = pushes risk up, blue = pushes risk down)")
    st.pyplot(fig3)
    top_driver = contrib.index[0]
    direction = "increasing" if contrib.iloc[0] > 0 else "decreasing"
    st.caption(f"For this customer, **{top_driver}** is the strongest driver, {direction} their predicted churn risk.")
else:
    st.write("No customers match the current filters.")
