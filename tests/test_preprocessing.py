"""
Unit tests for src/data/preprocessing.py
=========================================
Tests use an in-memory synthetic CSV to stay fast and independent of the
real 368 MB dataset.
"""

import io
import math
from pathlib import Path

import pandas as pd
import pytest

from src.data.preprocessing import (
    COLUMNS_EXPECTED,
    CRITICAL_COLUMNS,
    DATE_COLUMNS,
    NUMERIC_COLUMNS,
    _clean_strings,
    _normalize_amounts,
    _parse_dates,
    _parse_numerics,
    _validate,
    data_quality_report,
    load_raw,
    preprocess,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Minimal valid CSV that mirrors the real dataset schema
MINIMAL_CSV = (
    '"TRANSACTION_CODE","SERVICE_CODE","TRANSACTION_STATUS","TRANSACTION_DATE",'
    '"TRANSACTION_AMOUNT","REQUEST_REFERENCE","REQUEST_DATE","RESPONSE_DATE",'
    '"SOURCE_PHONE","DESTINATION_PHONE","TRANSACTION_FEES","DESTINATION_TYPE",'
    '"PARTNER_REFERENCE","BATCH_ID","SOURCE_CUSTOMER","DESTINATION_CUSTOMER",'
    '"TRANSACTION_DIRECTION","QR_INDICATOR","ACCOUNTING_RESPONSE_DATE",'
    '"ACCOUNTING_REQUEST_DATE","SETTLEMENT_STATUS","CHANNEL_TYPE","LANGUAGE_CODE"\n'
    '"TX001","SERVICE_06","VALIDATED","19/07/22 10:00:00","500000","REF01","",'
    '"19/07/22 10:00:01","TEL001","TEL002","100","EXTERNAL","","","","","","","","","","MOBILE","FR"\n'
    '"TX002","SERVICE_08","REJECTED","20/07/22 22:30:00","1000000","REF02","",'
    '"20/07/22 22:30:01","TEL003","","50","INTERNAL","","","","","","","","","","WEB","FR"\n'
    '"TX003","SERVICE_06","VALIDATED","21/07/22 08:15:00","250000","REF03","",'
    '"21/07/22 08:15:01","TEL001","TEL004","25","EXTERNAL","","","","","","","","","","MOBILE","FR"\n'
)

MISSING_CRITICAL_CSV = (
    '"TRANSACTION_CODE","SERVICE_CODE","TRANSACTION_STATUS","TRANSACTION_DATE",'
    '"TRANSACTION_AMOUNT","REQUEST_REFERENCE","REQUEST_DATE","RESPONSE_DATE",'
    '"SOURCE_PHONE","DESTINATION_PHONE","TRANSACTION_FEES","DESTINATION_TYPE",'
    '"PARTNER_REFERENCE","BATCH_ID","SOURCE_CUSTOMER","DESTINATION_CUSTOMER",'
    '"TRANSACTION_DIRECTION","QR_INDICATOR","ACCOUNTING_RESPONSE_DATE",'
    '"ACCOUNTING_REQUEST_DATE","SETTLEMENT_STATUS","CHANNEL_TYPE","LANGUAGE_CODE"\n'
    # Missing TRANSACTION_DATE
    '"TX099","SERVICE_06","VALIDATED","","500000","REF01","","","TEL001","TEL002",'
    '"100","EXTERNAL","","","","","","","","","","MOBILE","FR"\n'
)

DIRTY_CSV = (
    '"TRANSACTION_CODE","SERVICE_CODE","TRANSACTION_STATUS","TRANSACTION_DATE",'
    '"TRANSACTION_AMOUNT","REQUEST_REFERENCE","REQUEST_DATE","RESPONSE_DATE",'
    '"SOURCE_PHONE","DESTINATION_PHONE","TRANSACTION_FEES","DESTINATION_TYPE",'
    '"PARTNER_REFERENCE","BATCH_ID","SOURCE_CUSTOMER","DESTINATION_CUSTOMER",'
    '"TRANSACTION_DIRECTION","QR_INDICATOR","ACCOUNTING_RESPONSE_DATE",'
    '"ACCOUNTING_REQUEST_DATE","SETTLEMENT_STATUS","CHANNEL_TYPE","LANGUAGE_CODE"\n'
    # Amount with locale comma artefact: "1,500,000" → should become 1500000
    '"TX010","SERVICE_06","VALIDATED","25/07/22 09:00:00","1500000","","","","TEL005",'
    '"TEL006","  ","EXTERNAL","","","","","","","","","","MOBILE","FR"\n'
)

# Datetime values use comma as decimal separator for fractional seconds.
# These fields are intentionally unquoted to reproduce the real ingestion issue.
FRACTIONAL_DATETIME_COMMA_CSV = (
    '"TRANSACTION_CODE","SERVICE_CODE","TRANSACTION_STATUS","TRANSACTION_DATE",'
    '"TRANSACTION_AMOUNT","REQUEST_REFERENCE","REQUEST_DATE","RESPONSE_DATE",'
    '"SOURCE_PHONE","DESTINATION_PHONE","TRANSACTION_FEES","DESTINATION_TYPE",'
    '"PARTNER_REFERENCE","BATCH_ID","SOURCE_CUSTOMER","DESTINATION_CUSTOMER",'
    '"TRANSACTION_DIRECTION","QR_INDICATOR","ACCOUNTING_RESPONSE_DATE",'
    '"ACCOUNTING_REQUEST_DATE","SETTLEMENT_STATUS","CHANNEL_TYPE","LANGUAGE_CODE"\n'
    '"TXF01","SERVICE_06","VALIDATED",12/06/22 15:05:44,421000000,"500000","REFX",'
    '"12/06/22 15:05:45","12/06/22 15:05:46","TEL001","TEL002","10","EXTERNAL",'
    '"","","","","","","","0","MOBILE","FR"\n'
    '"TXF02","SERVICE_08","REJECTED","13/06/22 16:00:00","600000","REFY",'
    '13/06/22 16:01:00,123000000,13/06/22 16:01:05,456000000,"TEL003","TEL004",'
    '"20","INTERNAL","","","","","","","","0","WEB","FR"\n'
)

# French decimal comma in TRANSACTION_AMOUNT (24 fields after datetime fix)
# Raw line: 3 datetime commas (add 3 fields) + 1 French decimal in AMOUNT = 27 raw fields
# After datetime fix: 27 - 3 = 24 fields → algorithm merges AMOUNT
FRENCH_DECIMAL_AMOUNT_CSV = (
    '"TRANSACTION_CODE","SERVICE_CODE","TRANSACTION_STATUS","TRANSACTION_DATE",'
    '"TRANSACTION_AMOUNT","REQUEST_REFERENCE","REQUEST_DATE","RESPONSE_DATE",'
    '"SOURCE_PHONE","DESTINATION_PHONE","TRANSACTION_FEES","DESTINATION_TYPE",'
    '"PARTNER_REFERENCE","BATCH_ID","SOURCE_CUSTOMER","DESTINATION_CUSTOMER",'
    '"TRANSACTION_DIRECTION","QR_INDICATOR","ACCOUNTING_RESPONSE_DATE",'
    '"ACCOUNTING_REQUEST_DATE","SETTLEMENT_STATUS","CHANNEL_TYPE","LANGUAGE_CODE"\n'
    '"TXA01","SERVICE_09","VALIDATED",12/06/22 15:05:44,421000000,"500000","REFA",'
    '12/06/22 15:05:45,123000000,12/06/22 15:05:46,456000000,"TEL001","TEL002","10","EXTERNAL",'
    '"","","","","","","","","0","MOBILE","FR"\n'
    '"TXA02","SERVICE_09","VALIDATED",12/06/22 16:00:00,123000000,5,5,"",'
    '12/06/22 16:00:01,789000000,12/06/22 16:00:02,111000000,"TEL003","TEL004","20","INTERNAL",'
    '"","","","","","","","","0","WEB","FR"\n'
)

# French decimal comma in TRANSACTION_FEES (24 fields after datetime fix)
# Raw line: 3 datetime commas + 1 French decimal in FEES = 27 raw fields
FRENCH_DECIMAL_FEES_CSV = (
    '"TRANSACTION_CODE","SERVICE_CODE","TRANSACTION_STATUS","TRANSACTION_DATE",'
    '"TRANSACTION_AMOUNT","REQUEST_REFERENCE","REQUEST_DATE","RESPONSE_DATE",'
    '"SOURCE_PHONE","DESTINATION_PHONE","TRANSACTION_FEES","DESTINATION_TYPE",'
    '"PARTNER_REFERENCE","BATCH_ID","SOURCE_CUSTOMER","DESTINATION_CUSTOMER",'
    '"TRANSACTION_DIRECTION","QR_INDICATOR","ACCOUNTING_RESPONSE_DATE",'
    '"ACCOUNTING_REQUEST_DATE","SETTLEMENT_STATUS","CHANNEL_TYPE","LANGUAGE_CODE"\n'
    '"TXF01","SERVICE_08","VALIDATED",13/06/22 16:00:00,123000000,"3960","",'
    '13/06/22 16:00:01,456000000,13/06/22 16:00:05,789000000,"TEL001","TEL002",39,6,"EXTERNAL",'
    '"","","","","","","","","0","MOBILE","FR"\n'
)

# French decimal comma in BOTH AMOUNT and FEES (25 fields after datetime fix)
# Raw line: 3 datetime commas + 2 French decimals = 28 raw fields
FRENCH_DECIMAL_BOTH_CSV = (
    '"TRANSACTION_CODE","SERVICE_CODE","TRANSACTION_STATUS","TRANSACTION_DATE",'
    '"TRANSACTION_AMOUNT","REQUEST_REFERENCE","REQUEST_DATE","RESPONSE_DATE",'
    '"SOURCE_PHONE","DESTINATION_PHONE","TRANSACTION_FEES","DESTINATION_TYPE",'
    '"PARTNER_REFERENCE","BATCH_ID","SOURCE_CUSTOMER","DESTINATION_CUSTOMER",'
    '"TRANSACTION_DIRECTION","QR_INDICATOR","ACCOUNTING_RESPONSE_DATE",'
    '"ACCOUNTING_REQUEST_DATE","SETTLEMENT_STATUS","CHANNEL_TYPE","LANGUAGE_CODE"\n'
    '"TXB01","SERVICE_08","VALIDATED",21/07/22 12:21:51,613000000,2,2,"",'
    '21/07/22 12:21:52,123000000,21/07/22 12:21:53,456000000,"TEL001","TEL002",0,01,"EXTERNAL",'
    '"","","","","","","","","0","MOBILE","FR"\n'
)


@pytest.fixture
def minimal_df(tmp_path: Path) -> pd.DataFrame:
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(MINIMAL_CSV)
    return load_raw(csv_file)


@pytest.fixture
def preprocessed_df(tmp_path: Path) -> pd.DataFrame:
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(MINIMAL_CSV)
    return preprocess(csv_file)


# ---------------------------------------------------------------------------
# load_raw
# ---------------------------------------------------------------------------


class TestLoadRaw:
    def test_returns_dataframe(self, minimal_df):
        assert isinstance(minimal_df, pd.DataFrame)

    def test_row_count(self, minimal_df):
        assert len(minimal_df) == 3

    def test_column_names_uppercase(self, minimal_df):
        assert all(c == c.upper() for c in minimal_df.columns)

    def test_expected_columns_present(self, minimal_df):
        from src.data.preprocessing import COLUMNS_EXPECTED
        for col in COLUMNS_EXPECTED:
            assert col in minimal_df.columns, f"Missing column: {col}"

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_raw(tmp_path / "nonexistent.csv")

    def test_all_columns_are_str_type(self, minimal_df):
        # Before type casting, every column should still be object/str
        object_cols = minimal_df.select_dtypes(include="object").columns
        # At minimum the core string columns must be str
        assert "SERVICE_CODE" in object_cols
        assert "SOURCE_PHONE" in object_cols

    def test_repairs_fractional_datetime_comma_without_schema_shift(self, tmp_path):
        csv_file = tmp_path / "fractional.csv"
        csv_file.write_text(FRACTIONAL_DATETIME_COMMA_CSV)

        df = load_raw(csv_file)

        from src.data.preprocessing import COLUMNS_EXPECTED

        assert list(df.columns) == COLUMNS_EXPECTED
        assert df.loc[0, "TRANSACTION_DATE"] == "12/06/22 15:05:44.421000000"
        assert df.loc[0, "TRANSACTION_AMOUNT"] == "500000"
        assert df.loc[1, "REQUEST_DATE"] == "13/06/22 16:01:00.123000000"
        assert df.loc[1, "RESPONSE_DATE"] == "13/06/22 16:01:05.456000000"


# ---------------------------------------------------------------------------
# _parse_dates
# ---------------------------------------------------------------------------


class TestParseDates:
    def test_transaction_date_is_datetime(self, minimal_df):
        result = _parse_dates(minimal_df.copy())
        assert pd.api.types.is_datetime64_any_dtype(result["TRANSACTION_DATE"])

    def test_date_values_are_correct(self, minimal_df):
        result = _parse_dates(minimal_df.copy())
        assert result["TRANSACTION_DATE"].iloc[0].year == 2022
        assert result["TRANSACTION_DATE"].iloc[0].month == 7
        assert result["TRANSACTION_DATE"].iloc[0].day == 19

    def test_invalid_date_becomes_nat(self, tmp_path):
        bad_csv = MINIMAL_CSV.replace(
            '"19/07/22 10:00:00"', '"NOT_A_DATE"'
        )
        csv_file = tmp_path / "bad_date.csv"
        csv_file.write_text(bad_csv)
        df = load_raw(csv_file)
        result = _parse_dates(df)
        assert pd.isna(result["TRANSACTION_DATE"].iloc[0])

    def test_empty_date_becomes_nat(self, tmp_path):
        csv_file = tmp_path / "empty_date.csv"
        csv_file.write_text(MISSING_CRITICAL_CSV)
        df = load_raw(csv_file)
        result = _parse_dates(df)
        assert pd.isna(result["TRANSACTION_DATE"].iloc[0])

    def test_fractional_seconds_are_preserved(self, tmp_path):
        csv_file = tmp_path / "fractional_date.csv"
        csv_file.write_text(FRACTIONAL_DATETIME_COMMA_CSV)

        df = load_raw(csv_file)
        result = _parse_dates(df)

        assert result["TRANSACTION_DATE"].iloc[0] == pd.Timestamp(
            "2022-06-12 15:05:44.421000000"
        )
        assert result["REQUEST_DATE"].iloc[1] == pd.Timestamp(
            "2022-06-13 16:01:00.123000000"
        )
        assert result["RESPONSE_DATE"].iloc[1] == pd.Timestamp(
            "2022-06-13 16:01:05.456000000"
        )


# ---------------------------------------------------------------------------
# _parse_numerics
# ---------------------------------------------------------------------------


class TestParseNumerics:
    def test_amount_is_float(self, minimal_df):
        result = _parse_numerics(minimal_df.copy())
        assert pd.api.types.is_float_dtype(result["TRANSACTION_AMOUNT"])

    def test_amount_values_correct(self, minimal_df):
        result = _parse_numerics(minimal_df.copy())
        assert result["TRANSACTION_AMOUNT"].iloc[0] == pytest.approx(500000.0)
        assert result["TRANSACTION_AMOUNT"].iloc[1] == pytest.approx(1000000.0)

    def test_fees_is_float(self, minimal_df):
        result = _parse_numerics(minimal_df.copy())
        assert pd.api.types.is_float_dtype(result["TRANSACTION_FEES"])

    def test_non_numeric_becomes_nan(self, tmp_path):
        bad_csv = MINIMAL_CSV.replace('"500000"', '"N/A"', 1)
        csv_file = tmp_path / "bad_num.csv"
        csv_file.write_text(bad_csv)
        df = load_raw(csv_file)
        result = _parse_numerics(df)
        assert pd.isna(result["TRANSACTION_AMOUNT"].iloc[0])


# ---------------------------------------------------------------------------
# _clean_strings
# ---------------------------------------------------------------------------


class TestCleanStrings:
    def test_whitespace_only_becomes_na(self, tmp_path):
        csv_file = tmp_path / "dirty.csv"
        csv_file.write_text(DIRTY_CSV)
        df = load_raw(csv_file)
        df = _parse_numerics(df)
        result = _clean_strings(df)
        # TRANSACTION_FEES was "  " (whitespace) — should become NA
        assert pd.isna(result["TRANSACTION_FEES"].iloc[0])

    def test_empty_string_becomes_na(self, minimal_df):
        result = _clean_strings(minimal_df.copy())
        # REQUEST_DATE is "" in the CSV
        assert pd.isna(result["REQUEST_DATE"].iloc[0])

    def test_non_empty_values_preserved(self, minimal_df):
        result = _clean_strings(minimal_df.copy())
        assert result["SOURCE_PHONE"].iloc[0] == "TEL001"


# ---------------------------------------------------------------------------
# _validate
# ---------------------------------------------------------------------------


class TestValidate:
    def test_valid_rows_kept(self, preprocessed_df):
        assert len(preprocessed_df) == 3

    def test_row_missing_critical_column_dropped(self, tmp_path):
        csv_file = tmp_path / "missing.csv"
        csv_file.write_text(MISSING_CRITICAL_CSV)
        result = preprocess(csv_file)
        assert len(result) == 0

    def test_index_reset_after_drop(self, tmp_path):
        # Build a CSV with one valid and one invalid row
        mixed = MINIMAL_CSV + (
            '"TXBAD","SERVICE_06","VALIDATED","","999","","","","","","0","","","","","","","","","","","MOBILE","FR"\n'
        )
        csv_file = tmp_path / "mixed.csv"
        csv_file.write_text(mixed)
        result = preprocess(csv_file)
        # Only the original 3 valid rows should remain with clean index
        assert list(result.index) == list(range(len(result)))


# ---------------------------------------------------------------------------
# _normalize_amounts
# ---------------------------------------------------------------------------


class TestNormalizeAmounts:
    def test_norm_columns_exist(self, preprocessed_df):
        assert "TRANSACTION_AMOUNT_NORM" in preprocessed_df.columns
        assert "TRANSACTION_FEES_NORM" in preprocessed_df.columns

    def test_norm_values_correct(self, preprocessed_df):
        # 500000 / 1_000_000 = 0.5
        assert preprocessed_df["TRANSACTION_AMOUNT_NORM"].iloc[0] == pytest.approx(0.5)

    def test_no_inf_values(self, preprocessed_df):
        assert not preprocessed_df["TRANSACTION_AMOUNT_NORM"].isin(
            [float("inf"), float("-inf")]
        ).any()


# ---------------------------------------------------------------------------
# preprocess (integration)
# ---------------------------------------------------------------------------


class TestPreprocess:
    def test_full_pipeline_returns_dataframe(self, preprocessed_df):
        assert isinstance(preprocessed_df, pd.DataFrame)

    def test_date_columns_are_datetime(self, preprocessed_df):
        for col in DATE_COLUMNS:
            if col in preprocessed_df.columns:
                assert pd.api.types.is_datetime64_any_dtype(
                    preprocessed_df[col]
                ), f"{col} is not datetime"

    def test_numeric_columns_are_float(self, preprocessed_df):
        for col in NUMERIC_COLUMNS:
            assert pd.api.types.is_float_dtype(
                preprocessed_df[col]
            ), f"{col} is not float"

    def test_no_blank_strings_remain(self, preprocessed_df):
        str_cols = preprocessed_df.select_dtypes(include="object").columns
        for col in str_cols:
            blanks = preprocessed_df[col].str.strip().eq("").sum()
            assert blanks == 0, f"Blank strings found in column {col}"

    def test_norm_column_ratio(self, preprocessed_df):
        import numpy as np
        ratio = (
            preprocessed_df["TRANSACTION_AMOUNT_NORM"]
            / preprocessed_df["TRANSACTION_AMOUNT"]
        )
        assert np.allclose(ratio, 1e-6)

    def test_fractional_datetime_csv_end_to_end(self, tmp_path):
        csv_file = tmp_path / "fractional_pipeline.csv"
        csv_file.write_text(FRACTIONAL_DATETIME_COMMA_CSV)

        result = preprocess(csv_file)

        assert len(result) == 2
        assert result["TRANSACTION_AMOUNT"].iloc[0] == pytest.approx(500000.0)
        assert result["TRANSACTION_AMOUNT"].iloc[1] == pytest.approx(600000.0)
        assert result["TRANSACTION_DATE"].iloc[0] == pd.Timestamp(
            "2022-06-12 15:05:44.421000000"
        )


# ---------------------------------------------------------------------------
# data_quality_report
# ---------------------------------------------------------------------------


class TestDataQualityReport:
    def test_returns_dict(self, preprocessed_df):
        report = data_quality_report(preprocessed_df)
        assert isinstance(report, dict)

    def test_shape_key_present(self, preprocessed_df):
        report = data_quality_report(preprocessed_df)
        assert "shape" in report
        assert report["shape"] == preprocessed_df.shape

    def test_duplicate_tx_codes_is_int(self, preprocessed_df):
        report = data_quality_report(preprocessed_df)
        assert isinstance(report["duplicate_tx_codes"], int)

    def test_date_range_keys(self, preprocessed_df):
        report = data_quality_report(preprocessed_df)
        assert "date_range" in report
        assert "min" in report["date_range"]
        assert "max" in report["date_range"]


# ---------------------------------------------------------------------------
# French decimal comma handling
# ---------------------------------------------------------------------------


class TestFrenchDecimalComma:
    def test_amount_decimal_comma(self, tmp_path):
        csv_file = tmp_path / "amount_decimal.csv"
        csv_file.write_text(FRENCH_DECIMAL_AMOUNT_CSV)
        df = load_raw(csv_file)
        assert list(df.columns) == COLUMNS_EXPECTED
        assert len(df) == 2
        assert df.loc[1, "TRANSACTION_DATE"] == "12/06/22 16:00:00.123000000"
        assert df.loc[1, "TRANSACTION_AMOUNT"] == "5.5"
        # REQUEST_REFERENCE is "" in CSV which becomes NaN after _clean_strings
        assert pd.isna(df.loc[1, "REQUEST_REFERENCE"])

    def test_fees_decimal_comma(self, tmp_path):
        csv_file = tmp_path / "fees_decimal.csv"
        csv_file.write_text(FRENCH_DECIMAL_FEES_CSV)
        df = load_raw(csv_file)
        assert list(df.columns) == COLUMNS_EXPECTED
        assert len(df) == 1
        assert df.loc[0, "TRANSACTION_AMOUNT"] == "3960"
        assert df.loc[0, "TRANSACTION_FEES"] == "39.6"

    def test_both_decimal_comma(self, tmp_path):
        csv_file = tmp_path / "both_decimal.csv"
        csv_file.write_text(FRENCH_DECIMAL_BOTH_CSV)
        df = load_raw(csv_file)
        assert list(df.columns) == COLUMNS_EXPECTED
        assert len(df) == 1
        assert df.loc[0, "TRANSACTION_DATE"] == "21/07/22 12:21:51.613000000"
        assert df.loc[0, "TRANSACTION_AMOUNT"] == "2.2"
        assert df.loc[0, "TRANSACTION_FEES"] == "0.01"

    def test_end_to_end_preprocess_amount_decimal(self, tmp_path):
        csv_file = tmp_path / "amount_pipeline.csv"
        csv_file.write_text(FRENCH_DECIMAL_AMOUNT_CSV)
        result = preprocess(csv_file)
        assert len(result) == 2
        assert result["TRANSACTION_AMOUNT"].iloc[1] == pytest.approx(5.5)
        assert result["TRANSACTION_DATE"].iloc[1] == pd.Timestamp(
            "2022-06-12 16:00:00.123000000"
        )

    def test_end_to_end_preprocess_fees_decimal(self, tmp_path):
        csv_file = tmp_path / "fees_pipeline.csv"
        csv_file.write_text(FRENCH_DECIMAL_FEES_CSV)
        result = preprocess(csv_file)
        assert len(result) == 1
        assert result["TRANSACTION_FEES"].iloc[0] == pytest.approx(39.6)

    def test_end_to_end_preprocess_both_decimal(self, tmp_path):
        csv_file = tmp_path / "both_pipeline.csv"
        csv_file.write_text(FRENCH_DECIMAL_BOTH_CSV)
        result = preprocess(csv_file)
        assert len(result) == 1
        assert result["TRANSACTION_AMOUNT"].iloc[0] == pytest.approx(2.2)
        assert result["TRANSACTION_FEES"].iloc[0] == pytest.approx(0.01)
