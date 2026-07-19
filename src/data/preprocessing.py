"""
BAMIS Fraud Detection — Data Preprocessing Module
==================================================
Handles loading, cleaning, and validating the Mobile Money
transaction dataset (DATASET_ESP-2026.csv).

Pipeline steps
--------------
1. load_raw      — Read CSV, skip malformed lines, load all as str
2. _parse_dates  — Cast date columns to datetime (dayfirst, coerce errors)
3. _parse_numerics — Cast amount/fee columns to float (strip locale artefacts)
4. _clean_strings — Replace blank/whitespace strings with pd.NA
5. _validate     — Drop rows missing critical identifiers
6. _normalize_amounts — Add scaled *_NORM columns (÷ 1 000 000 for XOF)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & schema constants
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).parent / "DATASET_ESP-2026.csv"

COLUMNS_EXPECTED = [
    "TRANSACTION_CODE",
    "SERVICE_CODE",
    "TRANSACTION_STATUS",
    "TRANSACTION_DATE",
    "TRANSACTION_AMOUNT",
    "REQUEST_REFERENCE",
    "REQUEST_DATE",
    "RESPONSE_DATE",
    "SOURCE_PHONE",
    "DESTINATION_PHONE",
    "TRANSACTION_FEES",
    "DESTINATION_TYPE",
    "PARTNER_REFERENCE",
    "BATCH_ID",
    "SOURCE_CUSTOMER",
    "DESTINATION_CUSTOMER",
    "TRANSACTION_DIRECTION",
    "QR_INDICATOR",
    "ACCOUNTING_RESPONSE_DATE",
    "ACCOUNTING_REQUEST_DATE",
    "SETTLEMENT_STATUS",
    "CHANNEL_TYPE",
    "LANGUAGE_CODE",
]

DATE_COLUMNS = [
    "TRANSACTION_DATE",
    "REQUEST_DATE",
    "RESPONSE_DATE",
    "ACCOUNTING_RESPONSE_DATE",
    "ACCOUNTING_REQUEST_DATE",
]

NUMERIC_COLUMNS = [
    "TRANSACTION_AMOUNT",
    "TRANSACTION_FEES",
]

# Rows missing any of these are considered unprocessable and dropped.
CRITICAL_COLUMNS = [
    "TRANSACTION_CODE",
    "TRANSACTION_DATE",
    "TRANSACTION_AMOUNT",
    "SOURCE_PHONE",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_raw(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw CSV without any type casting.

    All columns are loaded as ``str`` so that downstream steps can apply
    precise, explicit conversions.  Malformed lines (field-count mismatches
    caused by unquoted commas inside values) are skipped with a warning.

    Parameters
    ----------
    path : Path, optional
        Override the default dataset path for testing or experimentation.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with all values as strings.
    """
    path = Path(path) if path else DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(
        path,
        sep=",",
        quotechar='"',
        dtype=str,
        on_bad_lines="warn",
        low_memory=False,
        encoding="utf-8",
    )
    # Strip leading/trailing whitespace from column names
    df.columns = df.columns.str.strip().str.upper()

    missing = set(COLUMNS_EXPECTED) - set(df.columns)
    if missing:
        logger.warning("Expected columns not found: %s", missing)

    logger.info("Raw load — %d rows, %d columns", len(df), len(df.columns))
    return df


def preprocess(path: Optional[Path] = None) -> pd.DataFrame:
    """Run the full preprocessing pipeline.

    Parameters
    ----------
    path : Path, optional
        Path to the raw CSV file.  Defaults to the bundled dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for feature engineering.
    """
    df = load_raw(path)
    df = _parse_dates(df)
    df = _parse_numerics(df)
    df = _clean_strings(df)
    df = _validate(df)
    df = _normalize_amounts(df)
    logger.info("Preprocessing complete — final shape: %s", str(df.shape))
    return df


def data_quality_report(df: pd.DataFrame) -> dict:
    """Return a lightweight quality summary dictionary.

    Useful for notebook inspection or automated quality gates.

    Parameters
    ----------
    df : pd.DataFrame
        A preprocessed dataframe (output of :func:`preprocess`).

    Returns
    -------
    dict
        Keys: shape, null_counts, duplicate_tx_codes, date_range,
        transaction_statuses, amount_stats.
    """
    report = {
        "shape": df.shape,
        "null_counts": df.isnull().sum().to_dict(),
        "duplicate_tx_codes": int(
            df["TRANSACTION_CODE"].duplicated(keep=False).sum()
        ),
    }

    if "TRANSACTION_DATE" in df.columns and pd.api.types.is_datetime64_any_dtype(
        df["TRANSACTION_DATE"]
    ):
        report["date_range"] = {
            "min": str(df["TRANSACTION_DATE"].min()),
            "max": str(df["TRANSACTION_DATE"].max()),
        }

    if "TRANSACTION_STATUS" in df.columns:
        report["transaction_statuses"] = (
            df["TRANSACTION_STATUS"].value_counts().to_dict()
        )

    if "TRANSACTION_AMOUNT" in df.columns:
        report["amount_stats"] = df["TRANSACTION_AMOUNT"].describe().to_dict()

    return report


# ---------------------------------------------------------------------------
# Private pipeline steps
# ---------------------------------------------------------------------------


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Cast date columns to ``datetime64``, coercing unparseable values to NaT."""
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col].str.strip(),
                format="mixed",
                dayfirst=True,
                errors="coerce",
            )
    return df


def _parse_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Cast numeric columns to ``float64``.

    Handles locale artefacts:
    - Removes non-numeric characters except digits, dot, minus.
    - Coerces remaining unparseable values to NaN.
    """
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            cleaned = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(r"[^\d.\-]", "", regex=True)
            )
            cleaned = cleaned.mask(cleaned.str.strip() == "")
            df[col] = pd.to_numeric(cleaned, errors="coerce").astype("float64")
    return df


def _clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Replace blank strings (empty or whitespace-only) with ``pd.NA``."""
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(
        lambda s: s.str.strip().replace("", pd.NA)
    )
    return df


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that are missing any critical column value.

    Critical columns are defined in :data:`CRITICAL_COLUMNS`.
    Removal count is logged as a warning when non-zero.
    """
    before = len(df)
    df = df.dropna(subset=[c for c in CRITICAL_COLUMNS if c in df.columns])
    removed = before - len(df)
    if removed:
        logger.warning(
            "Validation removed %d rows missing critical columns (%s)",
            removed,
            CRITICAL_COLUMNS,
        )
    return df.reset_index(drop=True)


def _normalize_amounts(df: pd.DataFrame) -> pd.DataFrame:
    """Add *_NORM columns with amounts divided by 1 000 000 (XOF scale).

    The raw dataset stores amounts as integers in a sub-unit scale.
    Dividing by 1 000 000 converts to standard XOF francs.
    """
    if "TRANSACTION_AMOUNT" in df.columns:
        df["TRANSACTION_AMOUNT_NORM"] = df["TRANSACTION_AMOUNT"] / 1_000_000
    if "TRANSACTION_FEES" in df.columns:
        df["TRANSACTION_FEES_NORM"] = df["TRANSACTION_FEES"] / 1_000_000
    return df
