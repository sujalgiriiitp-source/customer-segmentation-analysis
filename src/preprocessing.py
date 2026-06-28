"""
preprocessing.py
================
Transform the engineered DataFrame into a numeric, ML-ready matrix.

Steps:
    1. Select clustering features (drop ID, dates, target columns).
    2. Label-encode binary categoricals (Education ordinal, Is_Parent already 0/1).
    3. One-hot encode nominal categoricals (Marital_Status).
    4. Standard-scale all numeric features so distance-based algorithms
       (K-Means) treat each feature fairly.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Columns used for clustering (numeric & engineered).
CLUSTERING_NUMERIC = [
    "Income", "Recency", "Age",
    "Total_Spend", "Total_Purchases", "Avg_Spend",
    "Customer_Tenure_Days", "Children", "Family_Size",
    "Total_Accepted_Cmp", "Loyalty_Score",
    "NumWebVisitsMonth",
]

# Ordinal mapping for Education (preserves order).
EDUCATION_ORDER = {"Undergraduate": 0, "Graduate": 1, "Postgraduate": 2}


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Return (scaled numeric matrix, fitted StandardScaler).

    The returned DataFrame keeps column names so cluster profiling is easy.
    """
    print("\n=== PREPROCESSING ===")

    X = df.copy()

    # ---------- Encoding ----------
    if "Education" in X:
        X["Education_Ord"] = X["Education"].map(EDUCATION_ORDER).fillna(1).astype(int)
    if "Marital_Status" in X:
        X = pd.concat(
            [X, pd.get_dummies(X["Marital_Status"], prefix="Marital", dtype=int)],
            axis=1,
        )

    # Final clustering feature set
    feat_cols = [c for c in CLUSTERING_NUMERIC if c in X.columns]
    feat_cols += ["Education_Ord"] if "Education_Ord" in X else []
    feat_cols += [c for c in X.columns if c.startswith("Marital_") and c != "Marital_Status"]
    feat_cols += ["Is_Parent"] if "Is_Parent" in X else []

    feat_cols = list(dict.fromkeys(feat_cols))  # de-dupe, preserve order

    print(f"[preproc] Using {len(feat_cols)} features for clustering.")
    X_feat = X[feat_cols].astype(float)

    # ---------- Scaling ----------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feat_cols, index=X.index)

    print(f"[preproc] Scaled matrix shape: {X_scaled_df.shape}")
    return X_scaled_df, scaler
