"""
Customer Churn Risk Console (v3)
------------------------------
Structural redesign: no sidebar (the classic "Streamlit app" tell), no
equal-width KPI card row (the other classic tell). Instead: a single
asymmetric headline statement, a horizontal control strip, and tabbed
content areas instead of one long vertical scroll.

Run with:  streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
import os

st.set_page_config(page_title="Churn Risk Console", page_icon="\U0001F4D2", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))

# =================================================================
# DESIGN TOKENS
# =================================================================
INK = "#22271F"
PAPER = "#EEF1EC"
CARD = "#FFFFFF"
BORDER = "#D9D6C9"
MUTED = "#8C7B6B"
SAGE = "#5C7A63"
OXBLOOD = "#7A3B32"
TAUPE_BAR = "#B9AF9C"

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "axes.edgecolor": BORDER,
    "axes.linewidth": 0.8,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

#MainMenu {{visibility: hidden;}}
header {{visibility: hidden;}}
footer {{visibility: hidden;}}
.stDeployButton {{display:none;}}

html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; color: {INK}; }}
.stApp {{ background-color: {PAPER}; }}

.block-container {{ max-width: 1100px; padding-top: 1.5rem; }}

.console-header {{
    display: flex; justify-content: space-between; align-items: baseline;
    border-bottom: 2px solid {INK}; padding-bottom: 8px; margin-bottom: 26px;
}}
.console-title {{
    font-family: 'Space Grotesk', sans-serif; font-weight: 700;
    font-size: 22px; letter-spacing: -0.02em; color: {INK};
}}
.console-sub {{
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: {MUTED};
    text-transform: uppercase; letter-spacing: 0.06em;
}}

/* Headline block */
.headline-number {{
    font-family: 'Space Grotesk', sans-serif; font-weight: 700;
    font-size: 76px; line-height: 1; color: {OXBLOOD}; letter-spacing: -0.03em;
}}
.headline-label {{
    font-family: 'IBM Plex Sans', sans-serif; font-size: 15px; color: {INK};
    margin-top: 2px; max-width: 260px;
}}
.headline-sub {{
    font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: {MUTED};
    margin-top: 14px; letter-spacing: 0.02em;
}}
.headline-sub b {{ color: {INK}; }}

/* Tabs restyle -- make them look like a signal strip, not default pills */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px; border-bottom: 1px solid {BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    font-family: 'IBM Plex Mono', monospace; font-size: 12.5px;
    text-transform: uppercase; letter-spacing: 0.06em; color: {MUTED};
    padding: 8px 4px; background-color: transparent;
}}
.stTabs [aria-selected="true"] {{
    color: {INK} !important; border-bottom: 2px solid {OXBLOOD} !important;
}}

.card-label {{ font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 13.5px; color: {INK}; }}
.card-caption {{ font-family: 'IBM Plex Sans', sans-serif; font-size: 11.5px; color: {MUTED}; margin-bottom: 8px; }}

div[data-testid="stVerticalBlockBorderWrapper"] {{
    background-color: {CARD}; border: 1px solid {BORDER} !important; border-radius: 6px;
}}
div[data-testid="stExpander"] {{
    border: 1px solid {BORDER} !important; border-radius: 6px; background-color: {CARD};
}}
div[data-testid="stExpander"] summary {{
    font-family: 'IBM Plex Mono', monospace; font-size: 11.5px; text-transform: uppercase;
    letter-spacing: 0.06em; color: {MUTED};
}}

span[data-baseweb="tag"] {{ background-color: {INK} !important; }}
span[data-baseweb="tag"] span {{ color: {PAPER} !important; }}
.stRadio label p {{
    font-family: 'IBM Plex Mono', monospace !important; font-size: 11px !important;
    text-transform: uppercase; letter-spacing: 0.04em; color: {MUTED} !important;
}}
[data-testid="stWidgetLabel"] p {{
    font-family: 'IBM Plex Mono', monospace !important; font-size: 10.5px !important;
    text-transform: uppercase; letter-spacing: 0.06em; color: {MUTED} !important;
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

RISK_ACTIONS = {
    "High":   "Immediate intervention \u2014 high-touch outreach, priority offer.",
    "Medium": "Proactive monitoring \u2014 check-in email, engagement nudge.",
    "Low":    "Maintain \u2014 newsletter, low-cost loyalty content.",
}
TIER_COLOR = {"High": OXBLOOD, "Medium": TAUPE_BAR, "Low": SAGE}

@st.cache_resource
def load_pipeline():
    return joblib.load(os.path.join(HERE, "churn_pipeline.joblib"))

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

pipe = load_pipeline()
model = pipe['model']
scaler = pipe['scaler']
feature_cols = pipe['feature_cols']
numeric_cols = pipe['numeric_cols']
categorical_cols = pipe['categorical_cols']
meta = pipe['preproc_meta']
explainer = load_explainer(model)

def preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    display_cols = {}
    for c in ['gender', 'region_category', 'membership_category', 'feedback',
              'medium_of_operation', 'preferred_offer_types']:
        if c in df.columns:
            display_cols[c] = df[c].astype(str)

    if 'days_since_last_login' in df.columns:
        df['days_since_last_login'] = df['days_since_last_login'].replace(-999, meta['valid_login_median'])
    if 'avg_frequency_login_days' in df.columns:
        raw_series = pd.to_numeric(df['avg_frequency_login_days'], errors='coerce')
        df['avg_frequency_login_days'] = raw_series.fillna(meta['median_login_days'])
    if 'points_in_wallet' in df.columns:
        df['points_in_wallet'] = df['points_in_wallet'].fillna(meta['median_points'])
    if 'region_category' in df.columns:
        df['region_category'] = df['region_category'].fillna(meta['mode_region'])
    for c in ['joined_through_referral', 'medium_of_operation']:
        if c in df.columns:
            df[c] = df[c].replace('?', 'Unknown')
    if 'preferred_offer_types' in df.columns:
        df['preferred_offer_types'] = df['preferred_offer_types'].fillna(meta['preferred_offer_mode'])
    if 'joining_date' in df.columns:
        joining_dt = pd.to_datetime(df['joining_date'], format='%d-%m-%Y', errors='coerce')
        max_date = pd.to_datetime(meta['max_date'])
        df['tenure_months'] = ((max_date - joining_dt).dt.days / 30.4).fillna(0)
        df = df.drop(columns=['joining_date'])
    if 'last_visit_time' in df.columns:
        df['visit_hour'] = pd.to_datetime(df['last_visit_time'], format='%H:%M:%S',
                                           errors='coerce').dt.hour.fillna(12)
        df = df.drop(columns=['last_visit_time'])
    if 'feedback' in df.columns:
        df['feedback_score'] = df['feedback'].map(meta['feedback_map']).fillna(0)
        df = df.drop(columns=['feedback'])
    if 'past_complaint' in df.columns and 'complaint_status' in df.columns:
        unresolved = meta['unresolved_complaint_statuses']
        df['active_complaint'] = ((df['past_complaint'] == 'Yes') &
                                   (df['complaint_status'].isin(unresolved))).astype(int)
        df = df.drop(columns=['past_complaint', 'complaint_status'])
    if 'membership_category' in df.columns:
        df['membership_score'] = df['membership_category'].map(meta['membership_map']).fillna(0)
        df = df.drop(columns=['membership_category'])
    for col in ['used_special_discount', 'offer_application_preference', 'joined_through_referral']:
        if col in df.columns:
            df[col] = (df[col] == 'Yes').astype(int)
    df = df.drop(columns=['security_no', 'referral_id'], errors='ignore')
    df = pd.get_dummies(df, columns=[c for c in categorical_cols if c in df.columns], drop_first=True)
    df = df.reindex(columns=feature_cols, fill_value=0)
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    for k, v in display_cols.items():
        df[f"__display_{k}"] = v.values
    return df

def score(df_processed: pd.DataFrame) -> pd.DataFrame:
    X = df_processed[feature_cols]
    probs = model.predict_proba(X)[:, 1]
    out = df_processed.copy()
    out['churn_probability'] = probs
    out['risk_tier'] = pd.cut(probs, bins=[-0.01, 0.4, 0.7, 1.01], labels=['Low', 'Medium', 'High'])
    out['recommended_action'] = out['risk_tier'].map(RISK_ACTIONS)
    return out

# =================================================================
# HEADER
# =================================================================
st.markdown(f"""
<div class='console-header'>
    <div class='console-title'>Churn Risk Console</div>
    <div class='console-sub'>{pipe['model_name']} \u00b7 AUC-ROC 0.9755</div>
</div>
""", unsafe_allow_html=True)

# =================================================================
# HORIZONTAL CONTROL STRIP (replaces the sidebar entirely)
# =================================================================
with st.expander("Data source & filters", expanded=False):
    c0, c1, c2, c3 = st.columns([1.3, 1, 1, 1])
    with c0:
        data_source = st.radio("Data source", ["Bundled sample (300)", "Upload CSV"], horizontal=True)
        if data_source == "Upload CSV":
            uploaded = st.file_uploader("Raw customer CSV", type="csv", label_visibility="collapsed")
            raw_df = pd.read_csv(uploaded) if uploaded is not None else None
        else:
            sample_path = os.path.join(HERE, "sample_customers.csv")
            raw_df = pd.read_csv(sample_path) if os.path.exists(sample_path) else None

    if raw_df is not None:
        processed = preprocess(raw_df)
        scored = score(processed)
        with c1:
            tier_filter = st.multiselect("Risk tier", ['High', 'Medium', 'Low'], default=['High', 'Medium', 'Low'])
        filt = scored[scored['risk_tier'].isin(tier_filter)]
        with c2:
            if '__display_membership_category' in filt.columns:
                opts = sorted(filt['__display_membership_category'].dropna().unique().tolist())
                sel = st.multiselect("Membership", opts, default=opts)
                filt = filt[filt['__display_membership_category'].isin(sel)]
        with c3:
            if '__display_region_category' in filt.columns:
                opts = sorted(filt['__display_region_category'].dropna().unique().tolist())
                sel = st.multiselect("Region", opts, default=opts)
                filt = filt[filt['__display_region_category'].isin(sel)]

if raw_df is None:
    st.info("Open 'Data source & filters' above to upload a CSV or use the bundled sample.")
    st.stop()

# =================================================================
# HEADLINE STATEMENT (replaces the 4-box KPI row)
# =================================================================
n = len(filt)
n_high = int((filt['risk_tier'] == 'High').sum())
avg_risk = filt['churn_probability'].mean() if n else 0
counts = filt['risk_tier'].value_counts().reindex(['High', 'Medium', 'Low']).fillna(0)
pct = (counts / counts.sum() * 100) if counts.sum() else counts

hcol1, hcol2 = st.columns([1, 2.2])
with hcol1:
    st.markdown(f"""
    <div class='headline-number'>{n_high}</div>
    <div class='headline-label'>of {n} customers shown need retention action now</div>
    <div class='headline-sub'>avg. risk score <b>{avg_risk:.2f}</b> &nbsp;\u00b7&nbsp; {pct.get('Medium',0):.0f}% medium &nbsp;\u00b7&nbsp; {pct.get('Low',0):.0f}% low</div>
    """, unsafe_allow_html=True)

with hcol2:
    fig, ax = plt.subplots(figsize=(6.2, 1.3))
    left = 0
    for tier in ['High', 'Medium', 'Low']:
        w = pct.get(tier, 0)
        ax.barh(0, w, left=left, color=TIER_COLOR[tier], height=0.4, edgecolor='white', linewidth=1.5)
        if w > 5:
            ax.text(left + w/2, 0, f"{w:.0f}%", ha='center', va='center', fontsize=9, color='white')
        left += w
    ax.set_xlim(0, 100)
    ax.axis('off')
    # Tier legend, one label per segment, colored to match, positioned above the bar
    lx = 0
    for tier in ['High', 'Medium', 'Low']:
        w = pct.get(tier, 0)
        if w > 5:
            ax.text(lx + w/2, 0.62, tier.upper(), fontsize=7, color=TIER_COLOR[tier],
                    ha='center', family='monospace', fontweight='medium')
        lx += w
    plt.subplots_adjust(left=0.02, right=0.98, top=0.68, bottom=0.3)
    st.pyplot(fig, width='content')
    plt.close(fig)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# =================================================================
# TABBED CONTENT (replaces the long vertical stack of cards)
# =================================================================
tab_overview, tab_customers, tab_explain = st.tabs(["OVERVIEW", "CUSTOMERS", "EXPLAIN"])

with tab_overview:
    st.markdown("<div class='card-label'>Top risk drivers</div>", unsafe_allow_html=True)
    st.markdown("<div class='card-caption'>Champion model \u00b7 global importance, top 6</div>", unsafe_allow_html=True)
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(6)
    imp_sorted = importances.sort_values()
    fig2, ax2 = plt.subplots(figsize=(7, 2.4))
    ax2.barh(imp_sorted.index, imp_sorted.values, color=TAUPE_BAR, height=0.55)
    for spine in ['top', 'right', 'left']:
        ax2.spines[spine].set_visible(False)
    ax2.spines['bottom'].set_color(BORDER)
    ax2.tick_params(left=False, labelsize=9)
    ax2.set_xticks([])
    for i, v in enumerate(imp_sorted.values):
        ax2.text(v + 0.01, i, f"{v:.2f}", va='center', fontsize=8, color=MUTED)
    plt.tight_layout()
    st.pyplot(fig2, width='stretch')
    plt.close(fig2)

with tab_customers:
    st.markdown("<div class='card-label'>Customer risk ranking</div>", unsafe_allow_html=True)
    st.markdown("<div class='card-caption'>Sorted by churn probability, highest first</div>", unsafe_allow_html=True)

    display_cols_order = [c for c in filt.columns if c.startswith('__display_')]
    show_df = filt[display_cols_order + ['churn_probability', 'risk_tier', 'recommended_action']].copy()
    show_df.columns = [c.replace('__display_', '') for c in show_df.columns]
    show_df = show_df.sort_values('churn_probability', ascending=False)

    st.dataframe(
        show_df, width='stretch', height=340, hide_index=True,
        column_config={
            "churn_probability": st.column_config.ProgressColumn("Risk score", format="%.2f", min_value=0, max_value=1),
            "risk_tier": st.column_config.TextColumn("Tier"),
            "recommended_action": st.column_config.TextColumn("Recommended action", width="large"),
        },
    )
    st.download_button("Download filtered results (.csv)",
                        show_df.to_csv(index=True).encode('utf-8'),
                        file_name="churn_risk_export.csv", mime="text/csv")

with tab_explain:
    st.markdown("<div class='card-label'>Why this customer?</div>", unsafe_allow_html=True)
    st.markdown("<div class='card-caption'>Per-customer SHAP explanation</div>", unsafe_allow_html=True)

    display_cols_order = [c for c in filt.columns if c.startswith('__display_')]
    show_df2 = filt[display_cols_order + ['churn_probability', 'risk_tier']].copy()
    show_df2 = show_df2.sort_values('churn_probability', ascending=False)

    if len(show_df2):
        pick_col, info_col = st.columns([1, 2])
        with pick_col:
            selected_idx = st.selectbox("Customer row", show_df2.index.tolist(), label_visibility="collapsed")
        row = filt.loc[[selected_idx], feature_cols]
        sv = explainer.shap_values(row)
        contrib = pd.Series(sv[0], index=feature_cols).sort_values(key=abs, ascending=False).head(6)
        contrib_sorted = contrib.sort_values()
        colors_sorted = [OXBLOOD if v > 0 else SAGE for v in contrib_sorted.values]

        prob = filt.loc[selected_idx, 'churn_probability']
        with info_col:
            st.markdown(
                f"<div class='headline-sub' style='margin-top:8px;'>risk score "
                f"<b style='font-size:15px;'>{prob:.2f}</b></div>", unsafe_allow_html=True)

        fig3, ax3 = plt.subplots(figsize=(7.5, 2.4))
        ax3.barh(contrib_sorted.index, contrib_sorted.values, color=colors_sorted, height=0.55)
        for spine in ['top', 'right', 'left']:
            ax3.spines[spine].set_visible(False)
        ax3.spines['bottom'].set_color(BORDER)
        ax3.set_xticks([])
        ax3.tick_params(left=False, labelsize=9)
        plt.tight_layout()
        st.pyplot(fig3, width='stretch')
        plt.close(fig3)
    else:
        st.write("No customers match the current filters.")
