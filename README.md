# Market Surveillance — Anomaly Detection in Limit Order Book Data

A portfolio project detecting potential market manipulation patterns (spoofing, layering, quote-stuffing) in real limit order book data, combining rule-based signals, unsupervised anomaly detection, and a supervised model trained on synthetically injected ground truth.

Built as part of preparation for the LSEG Quantitative Analytics Early Careers Programme.

---

## Overview

Exchanges and surveillance systems monitor order book activity for patterns consistent with manipulation — orders placed with no intent to execute, used to create false impressions of supply/demand. This project builds an end-to-end detection pipeline on real historical order book data, evaluates it against synthetic ground truth (since real manipulation labels are not publicly available), and packages the results into an interactive dashboard.

**Data:** [LOBSTER](https://lobsterdata.com) Level-10 limit order book sample data — AAPL, 2012-06-21 (~400k order book events).

---

## Methodology

### 1. Data reconstruction

LOBSTER provides a `message` file (every order event: submission, cancellation, execution) and an `orderbook` file (resulting book state after each event). These were merged into a single event-level table with real dollar prices and human-readable timestamps.

### 2. Exploratory analysis

- **Order lifetimes** (time between submission and cancellation) were found to be heavily right-skewed, with a distinct, separate cluster of ultra-fast cancellations (<1ms) — 6.64% of all cancelled orders — sitting outside the main distribution of normal cancellation behavior.
- **Order-to-trade ratio** (new orders per execution, in 10-second windows) showed a similarly skewed distribution; the 95th percentile ratio was 50:1.

### 3. Rule-based baseline features

Eight features were engineered from this analysis:

- `is_ultra_fast_cancel` — order cancelled within 1ms
- `order_to_trade_ratio` / `is_high_ratio_window` — window-level order flow imbalance
- `dual_flag` — both conditions met simultaneously
- `size_vs_avg` — order size relative to the average size at that price level
- `fast_cancels_at_this_price` — repeated ultra-fast-cancel activity at a given price (a proxy for repeated behavior, in the absence of account-level data)
- `lifetime_sec` / `log_lifetime` — raw and log-scaled order lifetime

### 4. Unsupervised detection (Isolation Forest)

Trained on the 8 engineered features with `contamination=0.05`. Validated two ways:

- **100% of orders** flagging both rule-based conditions (`dual_flag`) were also flagged by the model — strong agreement between independent methods.
- Anomaly rate was checked **temporally**: rather than uniform noise, anomalies cluster in sharp, discrete bursts (some 10-second windows reaching 100% anomaly rate against a near-zero baseline) — consistent with genuine, isolated incidents rather than model noise.

### 5. Case study

The highest-intensity window (10 seconds, 37090s–37100s) was reconstructed in full: **158 new orders against 1 execution** (order-to-trade ratio 158:1), with **46% of matched orders cancelled within 1ms** — roughly 7x the day's baseline rate — concentrated at price levels above a best ask that remained flat throughout the burst.

### 6. Supervised model (synthetic ground truth)

Since true manipulation labels don't exist in public data, 987 synthetic spoofing episodes were injected into the real dataset (large orders, priced near the real touch, with sub-3ms lifetimes), and an XGBoost classifier was trained with class weighting (`scale_pos_weight`) to handle the ~0.6% positive rate.

### 7. Model comparison

| Model                           | Recall | Precision |
| ------------------------------- | ------ | --------- |
| XGBoost (supervised)            | 78.0%  | 49.0%     |
| Isolation Forest (unsupervised) | 20.9%  | 2.5%      |

The supervised model substantially outperforms unsupervised detection when labels are available. Isolation Forest still recovers meaningfully more true positives than random chance (~5% baseline under the contamination assumption), supporting its use as a first-line, no-label-required detector in realistic deployment settings.

### 8. Feature importance

`lifetime_sec` and `size_vs_avg` dominate the supervised model's decisions (via both gain and SHAP analysis) — consistent with the synthetic spoofing design. Hand-engineered binary flags contributed minimally beyond their continuous counterparts, indicating they were valuable for interpretable EDA but not essential to the model itself.

---

## Dashboard

An interactive Streamlit dashboard (`dashboard/app.py`) presents the full pipeline:

- Day overview metrics
- Interactive price chart with flagged anomalies overlaid
- Temporal anomaly-rate chart with a drill-down selector for any 10-second window
- Dynamic incident reconstruction (raw event table + microscope chart) for the selected window
- Model comparison (supervised vs unsupervised)
- Feature importance
- Methodology & limitations

**To run:**

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

---

## Limitations

- **Single day, single ticker.** Patterns found here may not generalize to other stocks or market regimes.
- **No account-level data.** LOBSTER data is anonymized at the order level; `fast_cancels_at_this_price` uses price level as a proxy for repeated behavior. Real surveillance systems use account/entity linkage, a materially stronger signal this project cannot replicate.
- **Synthetic ground truth.** Supervised evaluation uses injected synthetic spoofing episodes, not confirmed real manipulation. Strong performance demonstrates the model can learn this specific injected pattern well — it does not prove real-world manipulation would look identical.
- **Threshold calibration.** Percentile-based thresholds (95th percentile order-to-trade ratio, 1ms cancellation cutoff) were calibrated on this single day and would need re-calibration for deployment across different stocks or periods.

---

## Project Structure

```
lseg-surveillance-project/
├── data/
│   ├── AAPL_2012-06-21_..._message_10.csv
│   ├── AAPL_2012-06-21_..._orderbook_10.csv
│   └── processed/              # pickled intermediate results used by the dashboard
├── notebooks/
│   └── 01_data_loading.ipynb   # full analysis: EDA → features → models → evaluation
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, Streamlit, Plotly, matplotlib.
