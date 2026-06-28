"""
data_cleaning.py
================
Data-cleaning utilities for the Customer Personality dataset.

Steps performed:
    1. Drop constant / non-informative columns.
    2. Convert column dtypes (Dt_Customer -> datetime).
    3. Remove duplicate rows.
    4. Handle missing values (median impute for numeric, mode for categorical).
    5. Consolidate noisy categorical levels (Marital_Status, Education).
    6. Detect & cap outliers using the IQR method (winsorization).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that have a single unique value (no information)."""
    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if const_cols:
        print(f"[clean] Dropping constant columns: {const_cols}")
    return df.drop(columns=const_cols)


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to their semantically correct dtypes."""
    df = df.copy()
    if "Dt_Customer" in df.columns:
        df["Dt_Customer"] = pd.to_datetime(
            df["Dt_Customer"], errors="coerce", dayfirst=False
        )
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove fully duplicated rows (keeping the first occurrence)."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[clean] Removed {before - len(df)} duplicate rows.")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute numeric NaNs with median, categorical with mode."""
    df = df.copy()
    n_missing = int(df.isna().sum().sum())
    if n_missing == 0:
        print("[clean] No missing values found.")
        return df

    print(f"[clean] Imputing {n_missing} missing values.")
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df


def consolidate_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce noisy / overlapping categories to a clean canonical set."""
    df = df.copy()

    if "Marital_Status" in df.columns:
        mapping = {
            "Married": "Partner", "Together": "Partner",
            "Single": "Single", "Alone": "Single",
            "Divorced": "Single", "Widow": "Single",
            "Absurd": "Single", "YOLO": "Single",
        }
        df["Marital_Status"] = df["Marital_Status"].replace(mapping)

    if "Education" in df.columns:
        mapping = {
            "Basic": "Undergraduate",
            "2n Cycle": "Undergraduate",
            "Graduation": "Graduate",
            "Master": "Postgraduate",
            "PhD": "Postgraduate",
        }
        df["Education"] = df["Education"].replace(mapping)

    return df


def cap_outliers_iqr(df: pd.DataFrame, cols: list[str], k: float = 1.5) -> pd.DataFrame:
    """Cap extreme values using the IQR rule (winsorization).

    Capping (not removal) preserves sample size — important for clustering.
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
        if n_out:
            print(f"[clean] Capping {n_out} outliers in '{col}'.")
        df[col] = df[col].clip(lower=lo, upper=hi)
    return df


# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline in the recommended order."""
    print("\n=== DATA CLEANING ===")
    df = drop_constant_columns(df)
    df = fix_dtypes(df)
    df = remove_duplicates(df)
    df = handle_missing(df)
    df = consolidate_categoricals(df)
    df = cap_outliers_iqr(df, cols=["Income"])
    print(f"[clean] Final shape: {df.shape}")
    return df
