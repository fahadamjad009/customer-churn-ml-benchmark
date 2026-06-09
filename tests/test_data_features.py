"""
Tests for src/churn/data.py and src/churn/features.py.

These modules are pure-function data utilities with no ML dependencies.
Tests run against the real CSV at data/raw/churn.csv.
"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from churn.data import load_raw, resolve_target, split_train_holdout, TARGET_BINARY
from churn.features import (
    add_engineered_features,
    tenure_bucket,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ADDON_SERVICES,
)

@pytest.fixture(scope="module")
def raw_df():
    return load_raw("data/raw/churn.csv")

@pytest.fixture(scope="module")
def clean_df(raw_df):
    return resolve_target(raw_df)

@pytest.fixture(scope="module")
def featured_df(clean_df):
    return add_engineered_features(clean_df)

def test_load_raw_shape(raw_df):
    assert raw_df.shape[0] == 7043
    assert raw_df.shape[1] >= 20

def test_total_charges_no_nulls(raw_df):
    assert raw_df["TotalCharges"].isna().sum() == 0

def test_senior_citizen_is_string(raw_df):
    assert set(raw_df["SeniorCitizen"].unique()).issubset({"Yes", "No"})

def test_resolve_target_column_exists(clean_df):
    assert TARGET_BINARY in clean_df.columns

def test_resolve_target_binary_values(clean_df):
    assert set(clean_df[TARGET_BINARY].unique()).issubset({0, 1})

def test_resolve_target_churn_rate(clean_df):
    rate = clean_df[TARGET_BINARY].mean()
    assert 0.25 <= rate <= 0.28

def test_resolve_target_idempotent(clean_df):
    df2 = resolve_target(clean_df)
    assert df2.columns.tolist().count(TARGET_BINARY) == 1

def test_split_sizes(clean_df):
    train, holdout = split_train_holdout(clean_df)
    total = len(train) + len(holdout)
    assert total == len(clean_df)
    assert abs(len(holdout) / total - 0.20) < 0.01

def test_split_stratified(clean_df):
    train, holdout = split_train_holdout(clean_df)
    full_rate = clean_df[TARGET_BINARY].mean()
    holdout_rate = holdout[TARGET_BINARY].mean()
    assert abs(holdout_rate - full_rate) < 0.01

def test_split_deterministic(clean_df):
    _, h1 = split_train_holdout(clean_df)
    _, h2 = split_train_holdout(clean_df)
    assert h1.index.tolist() == h2.index.tolist()

def test_split_no_overlap(clean_df):
    train, holdout = split_train_holdout(clean_df)
    overlap = set(train["customerID"]) & set(holdout["customerID"])
    assert len(overlap) == 0

def test_tenure_bucket_values():
    s = pd.Series([0, 6, 13, 25, 49, 72])
    buckets = tenure_bucket(s)
    assert buckets.iloc[0] == "0-12"
    assert buckets.iloc[2] == "12-24"
    assert buckets.iloc[3] == "24-48"
    assert buckets.iloc[4] == "48+"

def test_tenure_bucket_no_nulls(clean_df):
    result = tenure_bucket(clean_df["tenure"])
    assert result.isna().sum() == 0

def test_engineered_columns_present(featured_df):
    expected = ["tenure_bucket", "charge_ratio", "avg_charges_per_month",
                "is_new_customer", "services_count", "has_streaming", "contract_months"]
    for col in expected:
        assert col in featured_df.columns

def test_services_count_range(featured_df):
    assert featured_df["services_count"].min() >= 0
    assert featured_df["services_count"].max() <= len(ADDON_SERVICES)

def test_is_new_customer_binary(featured_df):
    assert set(featured_df["is_new_customer"].unique()).issubset({0, 1})

def test_has_streaming_binary(featured_df):
    assert set(featured_df["has_streaming"].unique()).issubset({0, 1})

def test_contract_months_values(featured_df):
    assert set(featured_df["contract_months"].unique()).issubset({1, 12, 24})

def test_charge_ratio_no_nulls(featured_df):
    assert featured_df["charge_ratio"].isna().sum() == 0

def test_add_features_idempotent(featured_df):
    df2 = add_engineered_features(featured_df)
    assert df2["services_count"].equals(featured_df["services_count"])

def test_feature_lists_non_empty():
    assert len(NUMERIC_FEATURES) > 0
    assert len(CATEGORICAL_FEATURES) > 0
