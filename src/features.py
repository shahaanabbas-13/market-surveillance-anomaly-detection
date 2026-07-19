"""
features.py

Feature engineering functions for market surveillance anomaly detection.
Each function is a deterministic transformation, reusable across the
analysis notebook, the dashboard, and any future scripts.
"""

import numpy as np
import pandas as pd


def build_order_lifetimes(df, ultra_fast_threshold_sec=0.001):
    """
    Match each new_order event to its corresponding full_cancel event (by order_id)
    and compute how long each order survived before cancellation.

    Returns a DataFrame with one row per matched (submitted-and-cancelled) order.
    """
    new_orders = df[df['event_label'] == 'new_order'][
        ['order_id', 'time', 'size', 'price', 'direction']
    ]
    cancels = df[df['event_label'] == 'full_cancel'][['order_id', 'time']]

    lifetimes = new_orders.merge(cancels, on='order_id', suffixes=('_submit', '_cancel'))
    lifetimes['lifetime_sec'] = lifetimes['time_cancel'] - lifetimes['time_submit']
    lifetimes['log_lifetime'] = np.log10(lifetimes['lifetime_sec'] + 1e-6)
    lifetimes['is_ultra_fast_cancel'] = (
        lifetimes['lifetime_sec'] < ultra_fast_threshold_sec
    ).astype(int)

    return lifetimes


def build_bucket_stats(df, bucket_seconds=10):
    """
    Aggregate event counts into fixed time windows and compute the
    order-to-trade ratio (new orders per execution) for each window.
    """
    df = df.copy()
    df['time_bucket'] = (df['time'] // bucket_seconds).astype(int)

    bucket_counts = (
        df.groupby('time_bucket')['event_label']
        .value_counts()
        .unstack(fill_value=0)
    )

    if 'new_order' not in bucket_counts.columns:
        bucket_counts['new_order'] = 0
    if 'visible_execution' not in bucket_counts.columns:
        bucket_counts['visible_execution'] = 0

    bucket_counts['order_to_trade_ratio'] = (
        bucket_counts['new_order'] / bucket_counts['visible_execution'].replace(0, 1)
    )

    return bucket_counts


def flag_high_ratio_windows(bucket_counts, percentile=0.95):
    """Flag time windows whose order-to-trade ratio exceeds the given percentile."""
    threshold = bucket_counts['order_to_trade_ratio'].quantile(percentile)
    bucket_counts = bucket_counts.copy()
    bucket_counts['is_high_ratio_window'] = (
        bucket_counts['order_to_trade_ratio'] > threshold
    ).astype(int)
    return bucket_counts, threshold


def attach_window_features(lifetimes, bucket_counts, bucket_seconds=10):
    """Attach window-level features (order-to-trade ratio, high-ratio flag) to each order."""
    lifetimes = lifetimes.copy()
    lifetimes['time_bucket'] = (lifetimes['time_submit'] // bucket_seconds).astype(int)

    features = lifetimes.merge(
        bucket_counts[['order_to_trade_ratio', 'is_high_ratio_window']],
        on='time_bucket',
        how='left'
    )
    features['order_to_trade_ratio'] = features['order_to_trade_ratio'].fillna(
        features['order_to_trade_ratio'].median()
    )
    features['is_high_ratio_window'] = features['is_high_ratio_window'].fillna(0)

    features['dual_flag'] = (
        (features['is_ultra_fast_cancel'] == 1) & (features['is_high_ratio_window'] == 1)
    ).astype(int)

    return features


def attach_price_level_features(features, df):
    """
    Attach size-relative-to-average and repeated-fast-cancel-at-price features.
    `df` should be the full raw event DataFrame (for computing real average sizes).
    """
    features = features.copy()

    avg_size_by_price = df.groupby('price')['size'].mean()
    size_info = df[df['event_label'] == 'new_order'][['order_id', 'price', 'size']].copy()
    size_info['avg_size_at_price'] = size_info['price'].map(avg_size_by_price)
    size_info['size_vs_avg'] = size_info['size'] / size_info['avg_size_at_price']

    features = features.merge(
        size_info[['order_id', 'size_vs_avg']], on='order_id', how='left'
    )
    features['size_vs_avg'] = features['size_vs_avg'].fillna(features['size_vs_avg'].median())

    features['fast_cancels_at_this_price'] = (
        features.groupby('price')['is_ultra_fast_cancel'].transform('sum')
    )

    return features


FEATURE_COLUMNS = [
    'lifetime_sec', 'log_lifetime', 'is_ultra_fast_cancel',
    'order_to_trade_ratio', 'is_high_ratio_window', 'dual_flag',
    'size_vs_avg', 'fast_cancels_at_this_price'
]


def build_full_feature_table(df, ultra_fast_threshold_sec=0.001, bucket_seconds=10, high_ratio_percentile=0.95):
    """
    End-to-end pipeline: raw merged event DataFrame -> full engineered feature table.
    This is the single entry point the dashboard and any future script should call.
    """
    lifetimes = build_order_lifetimes(df, ultra_fast_threshold_sec=ultra_fast_threshold_sec)
    bucket_counts = build_bucket_stats(df, bucket_seconds=bucket_seconds)
    bucket_counts, _ = flag_high_ratio_windows(bucket_counts, percentile=high_ratio_percentile)
    features = attach_window_features(lifetimes, bucket_counts, bucket_seconds=bucket_seconds)
    features = attach_price_level_features(features, df)
    return features, bucket_counts