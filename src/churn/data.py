"""
Customer churn — data layer.

Responsibilities:
    * Load the raw IBM Telco Customer Churn CSV.
    * Clean known gotchas (`TotalCharges` has blank-space strings for
      tenure-zero customers; `SeniorCitizen` is encoded as 0/1 int but is
      conceptually categorical).
    * Resolve the binary churn target (`Yes`/`No` -> 1/0).
    * Provide a deterministic train / holdout split (`random_state=42`,
      `stratify=y`, `test_size=0.20`) that is used identically by the
      training pipeline (`train.py`) AND by the downstream business-metrics
      module so both score on the same holdout slice without sharing state.

This module deliberately has no ML dependencies. It is safe to import from
the dashboard at runtime.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_PATH: Path = Path("data/raw/churn.csv")
PROCESSED_DIR: Path = Path("data/processed")

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20

TARGET_COL: str = "Churn"
TARGET_BINARY: str = "churn_target"
ID_COL: str = "customerID"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_raw(path: Path | str = RAW_PATH) -> pd.DataFrame:
    """Load the raw Telco Customer Churn CSV and apply known cleanups.

    Known gotchas handled:
      * ``TotalCharges`` contains blank-space strings (" ") for customers
        with ``tenure == 0``. Coerced to numeric and filled with 0.0.
      * ``SeniorCitizen`` is encoded as ``0``/``1`` int but is conceptually
        a Yes/No categorical. Recoded so it joins the rest of the
        categorical pipeline cleanly.
    """
    df = pd.read_csv(path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    df["SeniorCitizen"] = (
        df["SeniorCitizen"].astype(str).map({"0": "No", "1": "Yes"}).fillna("No")
    )

    return df


def resolve_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map ``Churn`` (Yes/No) to a 0/1 integer target column.

    Idempotent: if ``churn_target`` already exists, the frame is returned
    unchanged.
    """
    if TARGET_BINARY in df.columns:
        return df
    df = df.copy()
    df[TARGET_BINARY] = (df[TARGET_COL] == "Yes").astype(int)
    return df


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------
def split_train_holdout(
    df: pd.DataFrame,
    target: str = TARGET_BINARY,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic stratified train / holdout split.

    The same call signature is used by ``train.py`` and
    ``analytics/business_metrics.py`` so that both reconstruct the identical
    holdout slice without sharing in-memory state. Critical for the
    cloud-safe pattern where the dashboard precomputes all artifacts and
    never re-trains.
    """
    train_df, holdout_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target],
    )
    return (
        train_df.reset_index(drop=True),
        holdout_df.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_raw()
    df = resolve_target(df)
    train, holdout = split_train_holdout(df)

    print(f"Raw:     {df.shape[0]:>5} rows   churn rate {df[TARGET_BINARY].mean():.3f}")
    print(f"Train:   {train.shape[0]:>5} rows   churn rate {train[TARGET_BINARY].mean():.3f}")
    print(f"Holdout: {holdout.shape[0]:>5} rows   churn rate {holdout[TARGET_BINARY].mean():.3f}")
