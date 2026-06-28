"""
feature_engineering.py
======================
Derive analytics-ready features from the cleaned customer dataset.

Engineered features:
    - Age                  : Current year - Year_Birth
    - Age_Group            : Binned age categories
    - Customer_Tenure_Days : Days since Dt_Customer (enrollment)
    - Total_Spend          : Sum across all Mnt* product columns
    - Avg_Spend            : Total_Spend / Total_Purchases
    - Total_Purchases      : Sum of Web + Catalog + Store + Deals purchases
    - Total_Accepted_Cmp   : Sum of campaign acceptances
    - Children             : Kidhome + Teenhome
    - Family_Size          : Adults (1 or 2) + Children
    - Is_Parent            : Binary
    - Income_Group         : Binned income categories
    - Loyalty_Score        : Composite score (tenure * frequency / recency)
    - Recency              : (already present — days since last purchase)
    - Frequency            : Total_Purchases (RFM)
    - Monetary             : Total_Spend (RFM)
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

SPEND_COLS = [
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds",
]
PURCHASE_COLS = [
    "NumWebPurchases", "NumCatalogPurchases",
    "NumStorePurchases", "NumDealsPurchases",
]
CAMPAIGN_COLS = [f"AcceptedCmp{i}" for i in range(1, 6)]


# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame, reference_year: int = 2015) -> pd.DataFrame:
    """Add all engineered features to the cleaned DataFrame.

    Args:
        df: cleaned customer DataFrame.
        reference_year: anchor year for Age (the dataset is 2014–2015 era,
            so we use 2015 instead of "today" to keep ages realistic).
    """
    print("\n=== FEATURE ENGINEERING ===")
    df = df.copy()

    # ---------- Age ----------
    df["Age"] = reference_year - df["Year_Birth"]
    df["Age"] = df["Age"].clip(lower=18, upper=90)  # safety
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[17, 30, 45, 60, 100],
        labels=["18-30", "31-45", "46-60", "60+"],
    )

    # ---------- Tenure ----------
    if "Dt_Customer" in df.columns:
        ref_date = pd.Timestamp(f"{reference_year}-01-01")
        df["Customer_Tenure_Days"] = (ref_date - df["Dt_Customer"]).dt.days.clip(lower=0)
    else:
        df["Customer_Tenure_Days"] = 0

    # ---------- Family ----------
    df["Children"] = df["Kidhome"] + df["Teenhome"]
    df["Is_Parent"] = (df["Children"] > 0).astype(int)
    df["Family_Size"] = df["Marital_Status"].map({"Partner": 2, "Single": 1}).fillna(1) + df["Children"]

    # ---------- Income groups ----------
    df["Income_Group"] = pd.cut(
        df["Income"],
        bins=[0, 30_000, 60_000, 90_000, np.inf],
        labels=["Low", "Mid", "High", "Premium"],
    )

    # ---------- Spending ----------
    df["Total_Spend"] = df[SPEND_COLS].sum(axis=1)

    # ---------- Purchases ----------
    df["Total_Purchases"] = df[PURCHASE_COLS].sum(axis=1)
    df["Avg_Spend"] = np.where(
        df["Total_Purchases"] > 0,
        df["Total_Spend"] / df["Total_Purchases"],
        0,
    )

    # ---------- Campaigns ----------
    df["Total_Accepted_Cmp"] = df[CAMPAIGN_COLS].sum(axis=1)

    # ---------- RFM ----------
    df["Frequency"] = df["Total_Purchases"]
    df["Monetary"] = df["Total_Spend"]

    # ---------- Loyalty Score ----------
    # Higher tenure & frequency, lower recency -> higher loyalty.
    tenure_norm = df["Customer_Tenure_Days"] / df["Customer_Tenure_Days"].max()
    freq_norm = df["Frequency"] / df["Frequency"].max()
    rec_norm = 1 - (df["Recency"] / df["Recency"].max())
    df["Loyalty_Score"] = (0.4 * freq_norm + 0.3 * tenure_norm + 0.3 * rec_norm) * 100
    df["Loyalty_Score"] = df["Loyalty_Score"].round(2)

    print(f"[feat] Engineered features. New shape: {df.shape}")
    return df
