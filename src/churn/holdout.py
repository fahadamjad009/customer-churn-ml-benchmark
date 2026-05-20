"""
Customer churn — holdout reconstruction utility.

Provides a single function that any downstream module (business-metrics,
dashboard, evaluation notebooks) can call to deterministically reconstruct
the train / holdout split *without* re-running the training pipeline.

This decoupling is what makes the cloud-safe pattern work: the dashboard
imports ``get_holdout`` to retrieve the same holdout that ``train.py`` saw,
scores it against a precomputed predicted-probability frame, and renders
business metrics on demand without any sklearn/xgboost calls at runtime.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn.data import (
    TARGET_BINARY,
    load_raw,
    resolve_target,
    split_train_holdout,
)
from churn.features import add_engineered_features

ID_COL = "customerID"
RAW_TARGET_COL = "Churn"


def get_holdout(
    raw_path: str | Path = "data/raw/churn.csv",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Reproduce the exact holdout split used in training.

    Returns
    -------
    X_holdout : DataFrame of features (engineered features included).
    y_holdout : Series of binary churn target.
    raw_holdout : DataFrame including identifier and any non-feature columns
        used by the dashboard for segment views (Contract, tenure_bucket,
        PaymentMethod, MonthlyCharges, etc.).
    """
    df = load_raw(raw_path)
    df = resolve_target(df)
    df = add_engineered_features(df)
    _, holdout = split_train_holdout(df)

    y_holdout = holdout[TARGET_BINARY]
    drop_cols = [TARGET_BINARY, RAW_TARGET_COL, ID_COL]
    X_holdout = holdout.drop(columns=drop_cols)

    return X_holdout, y_holdout, holdout


def get_train_and_holdout(
    raw_path: str | Path = "data/raw/churn.csv",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Full split utility for training paths.

    Returns
    -------
    X_train, y_train, X_holdout, y_holdout : the four arrays needed to fit
    and evaluate a model end-to-end.
    """
    df = load_raw(raw_path)
    df = resolve_target(df)
    df = add_engineered_features(df)
    train, holdout = split_train_holdout(df)

    y_train = train[TARGET_BINARY]
    y_holdout = holdout[TARGET_BINARY]
    drop_cols = [TARGET_BINARY, RAW_TARGET_COL, ID_COL]
    X_train = train.drop(columns=drop_cols)
    X_holdout = holdout.drop(columns=drop_cols)

    return X_train, y_train, X_holdout, y_holdout
