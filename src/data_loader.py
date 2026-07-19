"""
data_loader.py

Loads and merges LOBSTER message + orderbook CSV files into a single,
analysis-ready DataFrame with real dollar prices and human-readable event labels.
"""

import pandas as pd

MESSAGE_COLS = ['time', 'event_type', 'order_id', 'size', 'price', 'direction']

EVENT_TYPE_MAP = {
    1: 'new_order',
    2: 'partial_cancel',
    3: 'full_cancel',
    4: 'visible_execution',
    5: 'hidden_execution',
    7: 'trading_halt'
}


def load_message_file(path):
    """Load a LOBSTER message CSV, assign column names, and scale price to dollars."""
    messages = pd.read_csv(path, names=MESSAGE_COLS, header=None)
    messages['price'] = messages['price'] / 10000
    return messages


def load_orderbook_file(path, n_levels=10):
    """Load a LOBSTER orderbook CSV, assign level-based column names, and scale prices."""
    orderbook_cols = []
    for level in range(1, n_levels + 1):
        orderbook_cols += [
            f'ask_price_{level}', f'ask_size_{level}',
            f'bid_price_{level}', f'bid_size_{level}'
        ]

    orderbook = pd.read_csv(path, names=orderbook_cols, header=None)

    price_cols = [col for col in orderbook.columns if 'price' in col]
    orderbook[price_cols] = orderbook[price_cols] / 10000

    return orderbook


def load_and_merge(message_path, orderbook_path, n_levels=10):
    """
    Load both LOBSTER files, merge them side-by-side (they share row alignment
    by construction), and add human-readable event labels and timestamps.

    Returns a single DataFrame with 6 + (4 * n_levels) columns plus derived columns.
    """
    messages = load_message_file(message_path)
    orderbook = load_orderbook_file(orderbook_path, n_levels=n_levels)

    if len(messages) != len(orderbook):
        raise ValueError(
            f"Row count mismatch: message file has {len(messages)} rows, "
            f"orderbook file has {len(orderbook)} rows. These files must be aligned."
        )

    df = pd.concat([messages, orderbook], axis=1)
    df['event_label'] = df['event_type'].map(EVENT_TYPE_MAP)
    df['time_readable'] = pd.to_timedelta(df['time'], unit='s')

    return df