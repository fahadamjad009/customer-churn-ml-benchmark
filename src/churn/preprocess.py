from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


def make_preprocess_pipeline(df: pd.DataFrame, target_col: str) -> ColumnTransformer:
    X = df.drop(columns=[target_col])
    cat = [c for c in X.columns if X[c].dtype == "object"]
    num = [c for c in X.columns if c not in cat]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), num),
            ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ],
        remainder="drop",
    )


def split_xy(df: pd.DataFrame, target_col: str, test_size: float, random_state: int):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
