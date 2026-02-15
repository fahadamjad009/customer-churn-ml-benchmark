from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .preprocess import make_preprocess_pipeline, split_xy


def _as_binary(y: pd.Series) -> pd.Series:
    """Convert Yes/No or other labels to 0/1."""
    if y.dtype.kind in {"i", "u", "b", "f"}:
        return y.astype(int)

    lowered = y.astype(str).str.lower().str.strip()
    mapping = {"yes": 1, "y": 1, "true": 1, "1": 1, "churn": 1,
               "no": 0, "n": 0, "false": 0, "0": 0, "no churn": 0}

    if lowered.isin(mapping.keys()).all():
        return lowered.map(mapping).astype(int)

    codes, _ = pd.factorize(y)
    if len(set(codes)) != 2:
        raise ValueError("Target must be binary for churn benchmark.")
    return pd.Series(codes, index=y.index).astype(int)


def model_registry(random_state: int):
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "SVM": SVC(probability=True),
        "DecisionTree": DecisionTreeClassifier(random_state=random_state),
    }


def train_all_models(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
    save_dir: Path,
):
    """Train multiple ML models and return evaluation results."""
    save_dir.mkdir(parents=True, exist_ok=True)

    y = _as_binary(df[target_col])
    df2 = df.copy()
    df2[target_col] = y

    X_train, X_test, y_train, y_test = split_xy(df2, target_col, test_size, random_state)

    pre = make_preprocess_pipeline(df2, target_col)
    models = model_registry(random_state)

    results = {}

    for name, clf in models.items():
        pipe = Pipeline(steps=[("preprocess", pre), ("model", clf)])
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
        }

        if proba is not None and len(np.unique(y_test)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
            fpr, tpr, _ = roc_curve(y_test, proba)
        else:
            metrics["roc_auc"] = None
            fpr, tpr = None, None

        cm = confusion_matrix(y_test, preds).tolist()

        out_path = save_dir / f"{name}.joblib"
        joblib.dump(pipe, out_path)

        results[name] = {
            "model_path": str(out_path),
            "metrics": metrics,
            "confusion_matrix": cm,
            "roc": {
                "fpr": fpr.tolist() if fpr is not None else None,
                "tpr": tpr.tolist() if tpr is not None else None,
            },
        }

    return results
