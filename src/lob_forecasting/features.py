"""Feature engineering utilities for FI-2010 limit order book data."""

import pandas as pd


def make_lob_column_names(n_levels=10):
    """
    Create readable column names for top-k limit order book levels.

    The FI-2010 LOB columns are assumed to follow:
        ask_price_i, ask_size_i, bid_price_i, bid_size_i
    for levels i = 1, ..., n_levels.
    """
    columns = []

    for level in range(1, n_levels + 1):
        columns.extend([
            f"ask_price_{level}",
            f"ask_size_{level}",
            f"bid_price_{level}",
            f"bid_size_{level}",
        ])

    return columns


def make_lob_dataframe(X, n_levels=10):
    """
    Convert the first 4 * n_levels columns of an FI-2010 feature matrix
    into a readable limit order book DataFrame.
    """
    columns = make_lob_column_names(n_levels=n_levels)
    n_cols = 4 * n_levels

    return pd.DataFrame(X[:, :n_cols], columns=columns)


def add_basic_lob_features(lob_df):
    """
    Construct top-of-book microstructure features.
    """
    features = pd.DataFrame(index=lob_df.index)

    best_ask = lob_df["ask_price_1"]
    ask_size = lob_df["ask_size_1"]
    best_bid = lob_df["bid_price_1"]
    bid_size = lob_df["bid_size_1"]

    features["mid_price"] = (best_ask + best_bid) / 2
    features["spread"] = best_ask - best_bid

    features["level_1_imbalance"] = (
        (bid_size - ask_size) /
        (bid_size + ask_size)
    )

    features["microprice"] = (
        (best_ask * bid_size + best_bid * ask_size) /
        (bid_size + ask_size)
    )

    features["microprice_deviation"] = (
        features["microprice"] - features["mid_price"]
    )

    return features


def add_depth_features(lob_df, features, levels=(5, 10)):
    """
    Add bid/ask depth and depth-imbalance features over multiple levels.
    """
    features = features.copy()

    for k in levels:
        bid_size_cols = [f"bid_size_{level}" for level in range(1, k + 1)]
        ask_size_cols = [f"ask_size_{level}" for level in range(1, k + 1)]

        bid_depth = lob_df[bid_size_cols].sum(axis=1)
        ask_depth = lob_df[ask_size_cols].sum(axis=1)

        features[f"bid_depth_{k}"] = bid_depth
        features[f"ask_depth_{k}"] = ask_depth

        features[f"depth_imbalance_{k}"] = (
            (bid_depth - ask_depth) /
            (bid_depth + ask_depth)
        )

    return features


def add_rolling_context_features(features):
    """
    Add recent-history features based on mid-price, spread, imbalance,
    and microprice deviation.
    """
    features = features.copy()

    mid_price = features["mid_price"]
    spread = features["spread"]
    imbalance = features["level_1_imbalance"]
    microprice_dev = features["microprice_deviation"]

    features["mid_return_1"] = mid_price.pct_change(1)
    features["mid_return_5"] = mid_price.pct_change(5)
    features["mid_return_10"] = mid_price.pct_change(10)

    features["rolling_vol_10"] = features["mid_return_1"].rolling(10).std()
    features["rolling_vol_50"] = features["mid_return_1"].rolling(50).std()

    features["spread_mean_10"] = spread.rolling(10).mean()
    features["spread_std_10"] = spread.rolling(10).std()

    features["imbalance_mean_10"] = imbalance.rolling(10).mean()
    features["imbalance_std_10"] = imbalance.rolling(10).std()

    features["microprice_dev_mean_10"] = microprice_dev.rolling(10).mean()
    features["microprice_dev_std_10"] = microprice_dev.rolling(10).std()

    return features


def build_microstructure_features(X, include_context=True, fillna=True):
    """
    Build a complete custom microstructure feature table from FI-2010 features.

    Parameters
    ----------
    X : np.ndarray
        FI-2010 feature matrix.
    include_context : bool
        Whether to add rolling/recent-history features.
    fillna : bool
        Whether to fill missing rolling values with 0.

    Returns
    -------
    features : pd.DataFrame
        Engineered microstructure feature table.
    """
    lob_df = make_lob_dataframe(X)

    features = add_basic_lob_features(lob_df)
    features = add_depth_features(lob_df, features)

    if include_context:
        features = add_rolling_context_features(features)

    if fillna:
        features = features.fillna(0)

    return features