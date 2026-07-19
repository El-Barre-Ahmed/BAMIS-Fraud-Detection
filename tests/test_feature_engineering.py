"""
Unit tests for src/features/feature_engineering.py
====================================================
Uses a small in-memory DataFrame (not the real 368 MB dataset).
"""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import (
    NIGHT_END,
    NIGHT_START,
    SPLIT_PROXIMITY_PCT,
    THRESHOLD_HIGH,
    THRESHOLD_LOW,
    THRESHOLD_MID,
    _check_required_columns,
    add_counterparty_features,
    add_fee_ratio_feature,
    add_rolling_features,
    add_temporal_features,
    add_threshold_features,
    add_velocity_features,
    build_features,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def base_df() -> pd.DataFrame:
    """Minimal preprocessed-style DataFrame with 5 transactions."""
    return pd.DataFrame(
        {
            "TRANSACTION_DATE": pd.to_datetime(
                [
                    "2022-07-19 10:00:00",  # day, normal hour
                    "2022-07-19 10:20:00",  # same sender, 20 min later
                    "2022-07-19 23:05:00",  # night
                    "2022-07-20 10:00:00",  # next day
                    "2022-07-23 10:00:00",  # 4 days later
                ]
            ),
            "SOURCE_PHONE": ["TEL001", "TEL001", "TEL002", "TEL001", "TEL001"],
            "DESTINATION_PHONE": ["TEL010", "TEL011", "TEL012", "TEL010", "TEL013"],
            "TRANSACTION_AMOUNT": [
                90_000.0,     # just below THRESHOLD_LOW (100k)
                500_000.0,    # exactly at THRESHOLD_MID
                1_500_000.0,  # above THRESHOLD_HIGH
                200_000.0,    # normal
                450_100.0,    # near THRESHOLD_MID (within 10%)
            ],
            "TRANSACTION_FEES": [100.0, 500.0, 0.0, 200.0, 450.0],
            "TRANSACTION_STATUS": [
                "VALIDATED", "VALIDATED", "VALIDATED", "VALIDATED", "VALIDATED"
            ],
        }
    )


# ---------------------------------------------------------------------------
# _check_required_columns
# ---------------------------------------------------------------------------


class TestCheckRequiredColumns:
    def test_passes_with_all_columns(self, base_df):
        _check_required_columns(base_df)  # should not raise

    def test_raises_when_column_missing(self, base_df):
        df = base_df.drop(columns=["TRANSACTION_AMOUNT"])
        with pytest.raises(ValueError, match="Missing required columns"):
            _check_required_columns(df)


# ---------------------------------------------------------------------------
# add_temporal_features
# ---------------------------------------------------------------------------


class TestTemporalFeatures:
    def test_adds_expected_columns(self, base_df):
        result = add_temporal_features(base_df.copy())
        for col in ["tx_hour", "tx_day_of_week", "tx_is_weekend", "tx_is_night"]:
            assert col in result.columns

    def test_hour_values_correct(self, base_df):
        result = add_temporal_features(base_df.copy())
        assert result["tx_hour"].iloc[0] == 10
        assert result["tx_hour"].iloc[2] == 23

    def test_night_flag_true_at_23h(self, base_df):
        result = add_temporal_features(base_df.copy())
        assert result["tx_is_night"].iloc[2] is True or result["tx_is_night"].iloc[2] == True

    def test_night_flag_false_at_10h(self, base_df):
        result = add_temporal_features(base_df.copy())
        assert result["tx_is_night"].iloc[0] is False or result["tx_is_night"].iloc[0] == False

    def test_weekend_flag(self, base_df):
        result = add_temporal_features(base_df.copy())
        # 2022-07-19 is a Tuesday (weekday=1) → not weekend
        assert not result["tx_is_weekend"].iloc[0]
        # 2022-07-23 is a Saturday (weekday=5) → weekend
        assert result["tx_is_weekend"].iloc[4]

    def test_no_rows_lost(self, base_df):
        result = add_temporal_features(base_df.copy())
        assert len(result) == len(base_df)


# ---------------------------------------------------------------------------
# add_rolling_features
# ---------------------------------------------------------------------------


class TestRollingFeatures:
    def test_adds_expected_columns(self, base_df):
        result = add_rolling_features(base_df.copy())
        for col in [
            "roll_count_1h", "roll_amount_1h",
            "roll_count_24h", "roll_amount_24h",
            "roll_count_7d", "roll_amount_7d",
        ]:
            assert col in result.columns, f"Missing: {col}"

    def test_1h_count_for_two_close_tx(self, base_df):
        # TEL001 has tx at 10:00 and 10:20 (same hour window)
        result = add_rolling_features(base_df.copy())
        tel001 = result[result["SOURCE_PHONE"] == "TEL001"].sort_values("TRANSACTION_DATE")
        # second tx should see count >= 2 within 1h
        assert tel001["roll_count_1h"].iloc[1] >= 2

    def test_roll_amount_is_non_negative(self, base_df):
        result = add_rolling_features(base_df.copy())
        assert (result["roll_amount_1h"] >= 0).all()
        assert (result["roll_amount_24h"] >= 0).all()
        assert (result["roll_amount_7d"] >= 0).all()

    def test_7d_accumulates_all_tel001(self, base_df):
        result = add_rolling_features(base_df.copy())
        tel001 = result[result["SOURCE_PHONE"] == "TEL001"].sort_values("TRANSACTION_DATE")
        # Last tx of TEL001 is 4 days after the first → should see 3 tx in 7d window
        assert tel001["roll_count_7d"].iloc[-1] >= 3

    def test_no_rows_lost(self, base_df):
        result = add_rolling_features(base_df.copy())
        assert len(result) == len(base_df)


# ---------------------------------------------------------------------------
# add_velocity_features
# ---------------------------------------------------------------------------


class TestVelocityFeatures:
    def test_adds_seconds_since_last_tx(self, base_df):
        result = add_velocity_features(base_df.copy())
        assert "seconds_since_last_tx" in result.columns

    def test_first_tx_per_sender_is_nan(self, base_df):
        result = add_velocity_features(base_df.copy())
        tel002_rows = result[result["SOURCE_PHONE"] == "TEL002"]
        assert tel002_rows["seconds_since_last_tx"].isna().all()

    def test_velocity_is_positive_for_subsequent_tx(self, base_df):
        result = add_velocity_features(base_df.copy())
        non_nan = result["seconds_since_last_tx"].dropna()
        assert (non_nan > 0).all()

    def test_20_min_gap_is_1200_seconds(self, base_df):
        result = add_velocity_features(base_df.copy())
        tel001 = result[result["SOURCE_PHONE"] == "TEL001"].sort_values(
            "TRANSACTION_DATE"
        )
        # 10:00 → 10:20 = 1200 seconds
        assert tel001["seconds_since_last_tx"].iloc[1] == pytest.approx(1200.0)

    def test_no_rows_lost(self, base_df):
        result = add_velocity_features(base_df.copy())
        assert len(result) == len(base_df)


# ---------------------------------------------------------------------------
# add_threshold_features
# ---------------------------------------------------------------------------


class TestThresholdFeatures:
    def test_adds_expected_columns(self, base_df):
        result = add_threshold_features(base_df.copy())
        for col in [
            "is_near_threshold_low",
            "is_near_threshold_mid",
            "is_near_threshold_high",
            "is_potential_split",
            "exceeds_high_threshold",
        ]:
            assert col in result.columns

    def test_90k_near_threshold_low(self, base_df):
        # 90 000 is within 10% below 100 000 (lower bound = 90 000)
        result = add_threshold_features(base_df.copy())
        assert result["is_near_threshold_low"].iloc[0]

    def test_1_5m_exceeds_high_threshold(self, base_df):
        result = add_threshold_features(base_df.copy())
        assert result["exceeds_high_threshold"].iloc[2]

    def test_normal_200k_not_near_any_threshold(self, base_df):
        result = add_threshold_features(base_df.copy())
        row = result.iloc[3]
        assert not row["is_near_threshold_low"]
        assert not row["is_near_threshold_mid"]
        assert not row["is_near_threshold_high"]

    def test_potential_split_is_union(self, base_df):
        result = add_threshold_features(base_df.copy())
        expected = (
            result["is_near_threshold_low"]
            | result["is_near_threshold_mid"]
            | result["is_near_threshold_high"]
        )
        pd.testing.assert_series_equal(
            result["is_potential_split"], expected, check_names=False
        )

    def test_no_rows_lost(self, base_df):
        result = add_threshold_features(base_df.copy())
        assert len(result) == len(base_df)


# ---------------------------------------------------------------------------
# add_fee_ratio_feature
# ---------------------------------------------------------------------------


class TestFeeRatioFeature:
    def test_adds_fee_ratio_column(self, base_df):
        result = add_fee_ratio_feature(base_df.copy())
        assert "fee_ratio" in result.columns

    def test_ratio_correct_value(self, base_df):
        result = add_fee_ratio_feature(base_df.copy())
        # row 0: 100 / 90_000
        assert result["fee_ratio"].iloc[0] == pytest.approx(100 / 90_000)

    def test_zero_amount_produces_nan(self):
        df = pd.DataFrame(
            {
                "TRANSACTION_AMOUNT": [0.0],
                "TRANSACTION_FEES": [50.0],
            }
        )
        result = add_fee_ratio_feature(df)
        assert pd.isna(result["fee_ratio"].iloc[0])

    def test_zero_fee_on_large_amount(self, base_df):
        # row 2: fees=0, amount=1_500_000 → ratio should be 0.0
        result = add_fee_ratio_feature(base_df.copy())
        assert result["fee_ratio"].iloc[2] == pytest.approx(0.0)

    def test_no_rows_lost(self, base_df):
        result = add_fee_ratio_feature(base_df.copy())
        assert len(result) == len(base_df)


# ---------------------------------------------------------------------------
# add_counterparty_features
# ---------------------------------------------------------------------------


class TestCounterpartyFeatures:
    def test_adds_unique_dest_7d(self, base_df):
        result = add_counterparty_features(base_df.copy())
        assert "unique_dest_7d" in result.columns

    def test_unique_count_is_positive(self, base_df):
        result = add_counterparty_features(base_df.copy())
        assert (result["unique_dest_7d"] >= 1).all()

    def test_no_rows_lost(self, base_df):
        result = add_counterparty_features(base_df.copy())
        assert len(result) == len(base_df)

    def test_returns_unchanged_without_destination_col(self, base_df):
        df = base_df.drop(columns=["DESTINATION_PHONE"])
        result = add_counterparty_features(df.copy())
        assert "unique_dest_7d" not in result.columns
        assert len(result) == len(base_df)


# ---------------------------------------------------------------------------
# build_features (integration)
# ---------------------------------------------------------------------------


class TestBuildFeatures:
    def test_returns_dataframe(self, base_df):
        result = build_features(base_df)
        assert isinstance(result, pd.DataFrame)

    def test_no_rows_lost(self, base_df):
        result = build_features(base_df)
        assert len(result) == len(base_df)

    def test_all_feature_columns_present(self, base_df):
        result = build_features(base_df)
        expected = [
            "tx_hour", "tx_day_of_week", "tx_is_weekend", "tx_is_night",
            "roll_count_1h", "roll_amount_1h",
            "roll_count_24h", "roll_amount_24h",
            "roll_count_7d", "roll_amount_7d",
            "seconds_since_last_tx",
            "is_near_threshold_low", "is_near_threshold_mid",
            "is_near_threshold_high", "is_potential_split",
            "exceeds_high_threshold",
            "fee_ratio",
            "unique_dest_7d",
        ]
        for col in expected:
            assert col in result.columns, f"Missing feature column: {col}"

    def test_original_columns_preserved(self, base_df):
        result = build_features(base_df)
        for col in base_df.columns:
            assert col in result.columns

    def test_raises_on_missing_required_column(self, base_df):
        df = base_df.drop(columns=["SOURCE_PHONE"])
        with pytest.raises(ValueError):
            build_features(df)

    def test_no_all_nan_feature_column(self, base_df):
        result = build_features(base_df)
        feature_cols = [c for c in result.columns if c not in base_df.columns]
        for col in feature_cols:
            if col == "seconds_since_last_tx":
                continue   # first tx per sender is legitimately NaN
            assert not result[col].isna().all(), f"All-NaN column: {col}"
