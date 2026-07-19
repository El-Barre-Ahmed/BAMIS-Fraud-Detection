"""
BAMIS Fraud Detection — Feature Engineering Module
====================================================
Builds behavioral features from the preprocessed Mobile Money transaction
dataframe (output of src.data.preprocessing.preprocess).

Feature groups
--------------
1. Temporal          — hour-of-day, is_night, day-of-week
2. Rolling windows   — tx count and total amount over 1h / 24h / 7d per sender
3. Velocity          — time since last transaction per sender
4. Threshold         — proximity to common business limits, splitting flag
5. Fee ratio         — fee as fraction of amount (anomaly signal)
6. Counterparty      — unique counterparties per sender in last 7 days

All functions accept and return pd.DataFrame. They are pure transformations:
no data is loaded or saved here. Call build_features() for the full pipeline.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Business constants
# ---------------------------------------------------------------------------

# Common regulatory / operator thresholds in XOF (raw units, not normalised)
THRESHOLD_LOW = 100_000      # 100 000 XOF — small transaction limit
THRESHOLD_MID = 500_000      # 500 000 XOF — mid-tier alert
THRESHOLD_HIGH = 1_000_000   # 1 000 000 XOF — high-value threshold

# Night window: 22:00 – 05:59
NIGHT_START = 22
NIGHT_END = 6

# Rolling windows expressed in hours
WINDOWS_HOURS = [1, 24, 168]   # 1h, 24h, 7d

# Splitting detection: transactions within this % of a threshold
SPLIT_PROXIMITY_PCT = 0.10     # within 10 % below threshold


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe from ``src.data.preprocessing.preprocess``.
        Must contain at minimum:
        TRANSACTION_DATE (datetime64), TRANSACTION_AMOUNT (float),
        SOURCE_PHONE (str), TRANSACTION_FEES (float).

    Returns
    -------
    pd.DataFrame
        Original columns plus all engineered feature columns.
    """
    _check_required_columns(df)
    df = df.copy()
    df = add_temporal_features(df)
    df = add_rolling_features(df)
    df = add_velocity_features(df)
    df = add_threshold_features(df)
    df = add_fee_ratio_feature(df)
    df = add_counterparty_features(df)
    logger.info(
        "Feature engineering complete — %d rows, %d columns",
        len(df), len(df.columns),
    )
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features derived from TRANSACTION_DATE.

    New columns
    -----------
    tx_hour         : int  — hour of day (0–23)
    tx_day_of_week  : int  — day of week (0=Monday … 6=Sunday)
    tx_is_weekend   : bool — Saturday or Sunday
    tx_is_night     : bool — hour in [22, 23, 0, 1, 2, 3, 4, 5]
    """
    dt = df["TRANSACTION_DATE"]
    df["tx_hour"] = dt.dt.hour
    df["tx_day_of_week"] = dt.dt.dayofweek
    df["tx_is_weekend"] = df["tx_day_of_week"].isin([5, 6])
    df["tx_is_night"] = (df["tx_hour"] >= NIGHT_START) | (
        df["tx_hour"] < NIGHT_END
    )
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling count and amount features per SOURCE_PHONE.

    For each window W in [1h, 24h, 7d] adds:
        roll_count_Wh   : number of transactions by this sender in last W hours
        roll_amount_Wh  : total amount sent by this sender in last W hours

    Requires df to be sorted by TRANSACTION_DATE (sorting is done here).
    """
    df = df.sort_values("TRANSACTION_DATE").reset_index(drop=True)
    df = df.set_index("TRANSACTION_DATE")

    for hours in WINDOWS_HOURS:
        window = f"{hours}h"
        label = f"{hours}h" if hours < 168 else "7d"

        grouped = df.groupby("SOURCE_PHONE", group_keys=False)

        df[f"roll_count_{label}"] = grouped["TRANSACTION_AMOUNT"].transform(
            lambda s: s.rolling(window, min_periods=1).count()
        )
        df[f"roll_amount_{label}"] = grouped["TRANSACTION_AMOUNT"].transform(
            lambda s: s.rolling(window, min_periods=1).sum()
        )

    df = df.reset_index()   # restore TRANSACTION_DATE as column
    return df


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-since-last-transaction per SOURCE_PHONE (in seconds).

    New columns
    -----------
    seconds_since_last_tx : float — seconds elapsed since sender's previous tx.
                            NaN for the sender's first transaction.
    """
    if "TRANSACTION_DATE" not in df.columns:
        return df

    df = df.sort_values(["SOURCE_PHONE", "TRANSACTION_DATE"]).reset_index(drop=True)
    df["seconds_since_last_tx"] = (
        df.groupby("SOURCE_PHONE")["TRANSACTION_DATE"]
        .diff()
        .dt.total_seconds()
    )
    return df


def add_threshold_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add business-rule proximity and splitting signals.

    New columns
    -----------
    is_near_threshold_low  : bool — amount within SPLIT_PROXIMITY_PCT below THRESHOLD_LOW
    is_near_threshold_mid  : bool — amount within SPLIT_PROXIMITY_PCT below THRESHOLD_MID
    is_near_threshold_high : bool — amount within SPLIT_PROXIMITY_PCT below THRESHOLD_HIGH
    is_potential_split     : bool — any near-threshold flag is True
    exceeds_high_threshold : bool — amount > THRESHOLD_HIGH
    """
    amt = df["TRANSACTION_AMOUNT"]

    def _near(threshold: float) -> pd.Series:
        lower = threshold * (1 - SPLIT_PROXIMITY_PCT)
        return (amt >= lower) & (amt < threshold)

    df["is_near_threshold_low"] = _near(THRESHOLD_LOW)
    df["is_near_threshold_mid"] = _near(THRESHOLD_MID)
    df["is_near_threshold_high"] = _near(THRESHOLD_HIGH)
    df["is_potential_split"] = (
        df["is_near_threshold_low"]
        | df["is_near_threshold_mid"]
        | df["is_near_threshold_high"]
    )
    df["exceeds_high_threshold"] = amt > THRESHOLD_HIGH
    return df


def add_fee_ratio_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add fee-to-amount ratio as an anomaly signal.

    A zero fee on a large transaction, or an unusually high fee, can
    indicate a manipulated or fraudulent operation.

    New columns
    -----------
    fee_ratio : float — TRANSACTION_FEES / TRANSACTION_AMOUNT (NaN if amount == 0)
    """
    amount = df["TRANSACTION_AMOUNT"].replace(0, np.nan)
    df["fee_ratio"] = df["TRANSACTION_FEES"] / amount
    return df


def add_counterparty_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add unique destination count per SOURCE_PHONE in the last 7 days.

    A high fan-out (one sender → many receivers) is a classic mule indicator.

    New columns
    -----------
    unique_dest_7d : float — distinct DESTINATION_PHONE values for this sender
                    in a 7-day rolling window.
    """
    if "DESTINATION_PHONE" not in df.columns:
        return df

    df = df.sort_values("TRANSACTION_DATE").reset_index(drop=True)

    # Encode string destinations as numeric category codes so rolling.apply()
    # receives a float array (it cannot operate on object/str columns directly).
    dest_codes = (
        df["DESTINATION_PHONE"]
        .astype("category")
        .cat.codes
        .astype(float)
        .replace(-1.0, np.nan)   # category code -1 encodes NaN → restore NaN
    )

    # Build a helper DataFrame with a DatetimeIndex so time-based rolling works.
    tmp = pd.DataFrame(
        {"sender": df["SOURCE_PHONE"].values, "dest_code": dest_codes.values},
        index=df["TRANSACTION_DATE"],
    )

    def _rolling_nunique(s: pd.Series) -> pd.Series:
        return s.rolling("168h", min_periods=1).apply(
            lambda x: float(pd.Series(x).dropna().nunique()), raw=True
        )

    # transform keeps the original index length and order — no MultiIndex issue.
    counts = tmp.groupby("sender", group_keys=False)["dest_code"].transform(
        _rolling_nunique
    )

    df["unique_dest_7d"] = counts.values
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "TRANSACTION_DATE",
    "TRANSACTION_AMOUNT",
    "SOURCE_PHONE",
    "TRANSACTION_FEES",
}


def _check_required_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns for feature engineering: {missing}. "
            "Run src.data.preprocessing.preprocess() first."
        )
