"""
Customer churn — feature engineering.

Derived features that add business signal beyond raw columns. All
transformations are pure functions of the input row, with no leakage from
target or holdout (safe to apply before the train/holdout split).

Derived features
----------------
tenure_bucket
    Ordinal bucket ("0-12", "12-24", "24-48", "48+" months). Drives the
    segment-performance view in the dashboard.

charge_ratio
    ``MonthlyCharges / max(TotalCharges, 1)``. High ratio = new customer
    paying their first months; low ratio = long-tenured customer whose
    cumulative spend dwarfs current monthly.

avg_charges_per_month
    ``TotalCharges / max(tenure, 1)``. Useful proxy for average plan price
    when tenure is non-trivial.

services_count
    Number of subscribed add-on services across the six service columns.
    A naive "engagement" signal: customers with more services tend to
    churn less (loyalty), but those with high spend AND high churn risk
    are the highest-value targets for retention.

has_streaming
    Combined flag for streaming TV / movies. Useful for cross-sell
    segmentation.

is_new_customer
    Binary flag for ``tenure <= 6``. Early-tenure customers are a
    well-known churn cohort.

contract_months
    Numeric encoding of the ``Contract`` field (1 / 12 / 24 months).
    Captures the same information as the categorical encoding but in a
    monotonic form that gradient-boosting models exploit efficiently.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


ADDON_SERVICES: list[str] = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

TENURE_BUCKETS: list[str] = ["0-12", "12-24", "24-48", "48+"]

CONTRACT_TO_MONTHS: dict[str, int] = {
    "Month-to-month": 1,
    "One year": 12,
    "Two year": 24,
}


def tenure_bucket(tenure: pd.Series) -> pd.Series:
    """Bucket tenure (months) into the four ordinal cohorts."""
    bins = [-0.1, 12, 24, 48, np.inf]
    return pd.cut(tenure, bins=bins, labels=TENURE_BUCKETS).astype(str)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append derived features to a Telco-shaped DataFrame.

    Returns a copy. Idempotent in the sense that re-running on a frame that
    already has the derived columns will simply overwrite them with the
    same values.
    """
    df = df.copy()

    df["tenure_bucket"] = tenure_bucket(df["tenure"])
    df["charge_ratio"] = df["MonthlyCharges"] / df["TotalCharges"].clip(lower=1.0)
    df["avg_charges_per_month"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
    df["services_count"] = sum(
        (df[col] == "Yes").astype(int) for col in ADDON_SERVICES
    )
    df["has_streaming"] = (
        (df["StreamingTV"] == "Yes") | (df["StreamingMovies"] == "Yes")
    ).astype(int)
    df["contract_months"] = (
        df["Contract"].map(CONTRACT_TO_MONTHS).fillna(1).astype(int)
    )

    return df


# ---------------------------------------------------------------------------
# Feature lists consumed by preprocess.py
# ---------------------------------------------------------------------------
NUMERIC_FEATURES: list[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "charge_ratio",
    "avg_charges_per_month",
    "services_count",
    "has_streaming",
    "is_new_customer",
    "contract_months",
]

CATEGORICAL_FEATURES: list[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_bucket",
]
