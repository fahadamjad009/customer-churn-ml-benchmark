"""
Customer churn — preprocessing pipeline.

Constructs the leak-safe ``ColumnTransformer`` that handles:

* ``StandardScaler`` for numeric features.
* ``OneHotEncoder`` for categorical features (with ``handle_unknown="ignore"``
  so a category seen only in the holdout cannot crash inference).

This transformer is embedded inside an ``sklearn.pipeline.Pipeline`` together
with a ``CalibratedClassifierCV`` so that the scaler and encoder are fitted
on the training fold only during cross-validated calibration. No leakage.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Construct the leak-safe ColumnTransformer used by all models.

    Returns a fresh instance per call so each model in ``train.py`` gets its
    own transformer (otherwise the CalibratedClassifierCV cross-validation
    folds would silently share state).
    """
    numeric_pipe = Pipeline(
        steps=[
            ("scale", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
