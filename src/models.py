"""
models.py

Training and evaluation helpers for the unsupervised (Isolation Forest) and
supervised (XGBoost) anomaly detection models.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, average_precision_score


def train_isolation_forest(feature_df, contamination=0.05, n_estimators=200, random_state=42):
    """
    Fit an Isolation Forest on the given feature table (numeric columns only).
    Returns the fitted model, the fitted scaler, and the scaled feature matrix.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X)

    return model, scaler, X


def score_isolation_forest(model, X):
    """Return (anomaly_score, is_anomaly) arrays for a fitted Isolation Forest."""
    anomaly_score = model.decision_function(X)
    is_anomaly = model.predict(X)  # -1 = anomaly, 1 = normal
    return anomaly_score, is_anomaly


def generate_synthetic_spoofing_episodes(df, n_episodes=200, orders_per_episode_range=(3, 8),
                                          price_offset_range=(0.01, 0.10),
                                          size_range=(100, 500),
                                          lifetime_range_sec=(0.0002, 0.003),
                                          random_state=42):
    """
    Generate synthetic spoofing episodes anchored to real moments in the trading day,
    used as ground-truth positive labels for supervised evaluation.
    """
    rng = np.random.default_rng(random_state)
    synthetic_orders = []

    anchor_times = rng.choice(df['time'].values, size=n_episodes, replace=False)

    for episode_id, anchor_time in enumerate(anchor_times):
        nearest_idx = (df['time'] - anchor_time).abs().idxmin()
        ref_row = df.loc[nearest_idx]
        best_bid = ref_row['bid_price_1']
        best_ask = ref_row['ask_price_1']

        n_orders = rng.integers(orders_per_episode_range[0], orders_per_episode_range[1] + 1)

        for i in range(n_orders):
            direction = rng.choice([1, -1])
            price_offset = rng.uniform(*price_offset_range)
            price = (best_ask + price_offset) if direction == -1 else (best_bid - price_offset)

            size = rng.integers(*size_range)
            submit_time = anchor_time + i * 0.001
            lifetime = rng.uniform(*lifetime_range_sec)

            synthetic_orders.append({
                'episode_id': episode_id,
                'time_submit': submit_time,
                'time_cancel': submit_time + lifetime,
                'lifetime_sec': lifetime,
                'size': size,
                'price': price,
                'direction': direction,
                'label': 1
            })

    return pd.DataFrame(synthetic_orders)


def train_xgboost_classifier(X, y, test_size=0.3, random_state=42, n_estimators=200, max_depth=5):
    """
    Train a class-weighted XGBoost classifier on labeled data (real + synthetic),
    using a stratified train/test split to preserve the rare positive-class ratio.

    Returns the fitted model, X_test, y_test, y_pred, y_proba, and the printed report string.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        scale_pos_weight=scale_pos_weight,
        eval_metric='aucpr',
        random_state=random_state
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=['Real', 'Synthetic Spoofing'])
    pr_auc = average_precision_score(y_test, y_proba)

    return model, X_test, y_test, y_pred, y_proba, report, pr_auc