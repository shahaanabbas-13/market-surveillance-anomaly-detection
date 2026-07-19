import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt

# --- Page config (must be the first Streamlit command) ---
st.set_page_config(page_title="Market Surveillance Dashboard", layout="wide")

# --- Load data (cached so it doesn't reload on every interaction) ---
@st.cache_data
def load_data():
    df = pd.read_pickle('data/processed/full_events.pkl')
    combined = pd.read_pickle('data/processed/features_labeled.pkl')
    iso_results = pd.read_pickle('data/processed/iso_forest_results.pkl')
    with open('data/processed/model_comparison.json') as f:
        comparison = json.load(f)
    return df, combined, iso_results, comparison

df, combined, iso_results, comparison = load_data()

# --- Title ---
st.title("📊 Market Surveillance — Anomaly Detection Dashboard")
st.caption("AAPL, 2012-06-21 — LOBSTER Level-10 order book data")


# --- Introduction / context ---
with st.expander("ℹ️ About this dashboard", expanded=True):
    st.markdown("""
    This dashboard detects potential market manipulation patterns (spoofing, layering, quote-stuffing) 
    in real limit order book data, using a combination of rule-based signals and machine learning.

    **How to use it:**
    1. **Order Book chart** — see the day's price action with flagged anomalous orders overlaid.
    2. **Anomaly Rate chart** — see *when* anomalies cluster in time; select any spike from the 
       dropdown below it to inspect that specific incident in detail.
    3. **Incident Detail** — a second-by-second reconstruction of the selected window: new orders, 
       cancellations, and how the best bid/ask moved.
    4. **Model Comparison** — how a supervised model (trained on synthetic labeled examples) compares 
       to unsupervised anomaly detection (no labels needed).
    5. **Feature Importance & Limitations** — what actually drives the model's decisions, and honest 
       caveats about this analysis.

    *Built on LOBSTER Level-10 order book data (AAPL, 2012-06-21) as a market surveillance 
    portfolio project.*
    """)


# --- Overview metrics row ---
col1, col2, col3, col4 = st.columns(4)

total_orders = (df['event_label'] == 'new_order').sum()
total_cancels = (df['event_label'] == 'full_cancel').sum()
total_executions = (df['event_label'] == 'visible_execution').sum()
anomaly_rate = (combined['label'] == 0).mean()  # placeholder, we'll refine below

col1.metric("Total New Orders", f"{total_orders:,}")
col2.metric("Total Cancellations", f"{total_cancels:,}")
col3.metric("Total Executions", f"{total_executions:,}")
col4.metric("Overall Order-to-Trade Ratio", f"{total_orders / max(total_executions,1):.1f}")

import plotly.graph_objects as go

st.divider()
st.subheader("📈 Order Book Price Action with Flagged Anomalies")

# Build a lookup: which orders (by order_id) were flagged anomalous by Isolation Forest
combined_with_iso = combined.copy()
combined_with_iso['iso_anomaly'] = pd.NA  # default for synthetic rows, which have no IsoForest score
combined_with_iso.loc[combined_with_iso['label'] == 0, 'iso_anomaly'] = iso_results['is_anomaly'].values

fig = go.Figure()

# Background: best bid/ask lines
fig.add_trace(go.Scattergl(
    x=df['time'], y=df['bid_price_1'],
    mode='lines', name='Best Bid',
    line=dict(color='lightgray', width=1)
))
fig.add_trace(go.Scattergl(
    x=df['time'], y=df['ask_price_1'],
    mode='lines', name='Best Ask',
    line=dict(color='darkgray', width=1)
))

# Overlay: flagged anomalous orders (real orders only, label==0, iso_anomaly==-1)
anomalies = combined_with_iso[
    (combined_with_iso['label'] == 0) & (combined_with_iso['iso_anomaly'] == -1)
]
fig.add_trace(go.Scattergl(
    x=anomalies['time_submit'], y=anomalies['price'],
    mode='markers', name='Flagged Anomaly',
    marker=dict(color='red', size=4, opacity=0.5)
))

fig.update_layout(
    xaxis_title="Time (seconds since midnight)",
    yaxis_title="Price ($)",
    height=500,
    hovermode='closest'
)

st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("🔥 Anomaly Rate Over Time (10-second windows)")
st.caption("Reveals whether anomalies cluster in discrete bursts, or spread uniformly through the day")

# Recompute per-bucket anomaly rate (same logic as Step 13.5)
real_orders_with_iso = combined_with_iso[combined_with_iso['label'] == 0].copy()
real_orders_with_iso['time_bucket'] = (real_orders_with_iso['time_submit'] // 10).astype(int)

anomalies_per_bucket = real_orders_with_iso.groupby('time_bucket')['iso_anomaly'].apply(lambda x: (x == -1).sum())
total_per_bucket = real_orders_with_iso.groupby('time_bucket').size()
anomaly_rate_per_bucket = (anomalies_per_bucket / total_per_bucket).fillna(0)

fig2 = go.Figure()
fig2.add_trace(go.Scattergl(
    x=anomaly_rate_per_bucket.index * 10,  # convert bucket index back to seconds
    y=anomaly_rate_per_bucket.values,
    mode='lines',
    line=dict(color='crimson', width=1)
))
fig2.update_layout(
    xaxis_title="Time (seconds since midnight)",
    yaxis_title="Fraction of orders flagged anomalous",
    height=350
)
st.plotly_chart(fig2, use_container_width=True)

# Let the user pick a bucket to drill into
st.write("**Select a time window to inspect:**")
selected_bucket = st.selectbox(
    "Time bucket (10-sec window)",
    options=sorted(anomaly_rate_per_bucket[total_per_bucket >= 5].sort_values(ascending=False).index[:20]),
    format_func=lambda b: f"{b*10}s–{b*10+10}s  (anomaly rate: {anomaly_rate_per_bucket[b]:.0%}, orders: {total_per_bucket[b]})"
)

st.divider()
window_start = selected_bucket * 10
window_end = window_start + 10

st.subheader(f"🔍 Incident Detail — {window_start}s to {window_end}s")

# Pull raw events in this window from the full event log
case_study = df[(df['time'] >= window_start) & (df['time'] < window_end)].copy()

# --- Summary stats for this window ---
n_new = (case_study['event_label'] == 'new_order').sum()
n_exec = (case_study['event_label'] == 'visible_execution').sum()
n_cancel = (case_study['event_label'] == 'full_cancel').sum()
ratio = n_new / max(n_exec, 1)

c1, c2, c3, c4 = st.columns(4)
c1.metric("New Orders", n_new)
c2.metric("Cancellations", n_cancel)
c3.metric("Executions", n_exec)
c4.metric("Order-to-Trade Ratio", f"{ratio:.1f}")

# --- Microscope chart: price action + individual order/cancel events ---
fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=case_study['time'], y=case_study['bid_price_1'],
    mode='lines', name='Best Bid', line=dict(color='green', width=1.5)
))
fig3.add_trace(go.Scatter(
    x=case_study['time'], y=case_study['ask_price_1'],
    mode='lines', name='Best Ask', line=dict(color='red', width=1.5)
))

new_orders_window = case_study[case_study['event_label'] == 'new_order']
cancels_window = case_study[case_study['event_label'] == 'full_cancel']

fig3.add_trace(go.Scatter(
    x=new_orders_window['time'], y=new_orders_window['price'],
    mode='markers', name='New Order',
    marker=dict(color='blue', size=7, symbol='triangle-up')
))
fig3.add_trace(go.Scatter(
    x=cancels_window['time'], y=cancels_window['price'],
    mode='markers', name='Cancel',
    marker=dict(color='black', size=7, symbol='x')
))

fig3.update_layout(
    xaxis_title="Time (sec since midnight)",
    yaxis_title="Price ($)",
    height=450
)
st.plotly_chart(fig3, use_container_width=True)

# --- Raw event table, for full transparency ---
with st.expander("View raw event log for this window"):
    st.dataframe(
        case_study[['time', 'event_label', 'order_id', 'size', 'price', 'direction', 'bid_price_1', 'ask_price_1']],
        use_container_width=True
    )

st.divider()
st.subheader("🤖 Model Comparison: Supervised vs Unsupervised Detection")

st.caption(
    "Evaluated against synthetically injected spoofing episodes (ground truth), "
    "since real-world manipulation labels are not publicly available."
)

comp_df = pd.DataFrame({
    'Model': ['XGBoost (supervised)', 'Isolation Forest (unsupervised)'],
    'Recall': [comparison['xgboost']['recall'], comparison['isolation_forest']['recall']],
    'Precision': [comparison['xgboost']['precision'], comparison['isolation_forest']['precision']]
})

col_a, col_b = st.columns([1, 1])

with col_a:
    st.dataframe(
        comp_df.style.format({'Recall': '{:.1%}', 'Precision': '{:.1%}'}),
        use_container_width=True, hide_index=True
    )

with col_b:
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(name='Recall', x=comp_df['Model'], y=comp_df['Recall'], marker_color='steelblue'))
    fig4.add_trace(go.Bar(name='Precision', x=comp_df['Model'], y=comp_df['Precision'], marker_color='indianred'))
    fig4.update_layout(barmode='group', height=300, yaxis_title="Score", yaxis_tickformat='.0%')
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("""
**Interpretation:** The supervised model substantially outperforms unsupervised detection when 
ground-truth labels are available — but still, Isolation Forest recovers meaningfully more true 
positives than random chance (~5% baseline under the contamination assumption), validating it as 
a reasonable first-line detector in real-world settings where labels don't exist.
""")

st.divider()
st.subheader("📊 Feature Importance (XGBoost)")

importance_df = pd.DataFrame({
    'Feature': ['lifetime_sec', 'size_vs_avg', 'order_to_trade_ratio', 'fast_cancels_at_this_price',
                'is_high_ratio_window', 'is_ultra_fast_cancel', 'dual_flag', 'log_lifetime'],
    'Gain': [615.4, 77.4, 18.4, 32.8, 57.9, 41.5, 38.3, 0.0]
}).sort_values('Gain', ascending=True)

fig5 = go.Figure(go.Bar(
    x=importance_df['Gain'], y=importance_df['Feature'],
    orientation='h', marker_color='teal'
))
fig5.update_layout(xaxis_title="Importance Score (Gain)", height=350)
st.plotly_chart(fig5, use_container_width=True)

st.markdown("""
**Interpretation:** `lifetime_sec` and `size_vs_avg` dominate model decisions — consistent with the 
synthetic spoofing design (short-lived, larger-than-average orders). Hand-engineered binary flags 
(`is_ultra_fast_cancel`, `dual_flag`) contributed minimally beyond their continuous counterparts, 
suggesting they were valuable for interpretable EDA but not essential to the model itself.
""")

st.divider()
st.subheader("⚠️ Methodology & Limitations")
st.markdown("""
- **Data scope:** Single trading day, single ticker (AAPL, 2012-06-21), LOBSTER Level-10 sample data.
  Patterns found here may not generalize to other stocks, days, or market regimes.
- **No account-level data:** LOBSTER data is anonymized at the order level with no trader/account ID, 
  so features like `fast_cancels_at_this_price` use price level as a proxy for repeated behavior — 
  real surveillance systems would use account linkage, which is a stronger signal.
- **Synthetic ground truth:** Supervised model evaluation uses synthetically injected spoofing 
  episodes, not confirmed real-world manipulation. Strong performance here demonstrates the model 
  can learn *this specific injected pattern* well — it does not prove real-world manipulation would 
  look identical.
- **Threshold choices:** Percentile-based thresholds (95th percentile for order-to-trade ratio, 
  1ms for ultra-fast cancellation) were calibrated on this single day's data and would need 
  re-calibration for deployment across different stocks or time periods.
""")