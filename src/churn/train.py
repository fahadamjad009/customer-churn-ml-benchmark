"""
Customer churn — training pipeline.

Trains three calibrated classifiers on the IBM Telco Customer Churn dataset:

1. **XGBoost** (champion candidate; gradient boosting + isotonic
   calibration).
2. **LightGBM** (champion candidate; faster training, often comparable
   PR-AUC on tabular benchmarks).
3. **Logistic Regression** (interpretable baseline; the model a hiring
   manager can reason about with a single coefficient vector).

All three are wrapped in ``CalibratedClassifierCV(method="isotonic", cv=5)``
so that the predicted probabilities are well-calibrated. This matters because
every downstream business metric (decile lift, cost-based threshold,
campaign ROI) is computed against these probabilities directly. A model with
strong PR-AUC but poor calibration is unsuitable for thresholding against
business KPIs.

Class imbalance is handled by ``scale_pos_weight`` (boosting models) and
``class_weight="balanced"`` (LR). The isotonic calibration corrects the
probability scale that these adjustments distort.

Champion selection
------------------
Champion = highest PR-AUC on the holdout. PR-AUC is preferred over ROC-AUC
for class-imbalanced churn (~26%) because PR-AUC focuses on positive-class
precision/recall rather than ranking pure-negative customers.

Outputs
-------
``models/champion.joblib``
    The winning model (full sklearn Pipeline).

``models/{xgboost,lightgbm,lr}.joblib``
    All three trained pipelines, for the dashboard "model comparison" view.

``models/leaderboard.json``
    Model-vs-model metrics summary + champion declaration + split metadata.

``reports/holdout_scores.parquet``
    Predicted probability per holdout customer for every model. Consumed by
    ``analytics/business_metrics.py`` to compute decile lift, cost-based
    threshold sweep, segment performance, etc. without re-loading models.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from churn.holdout import get_train_and_holdout
from churn.preprocess import build_preprocessor

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

# Telco churn rate is ~26.5%, so neg/pos ratio ~= 2.77.
# Used as scale_pos_weight for boosting models.
TELCO_POS_WEIGHT = 2.77


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_models() -> dict[str, Pipeline]:
    """Construct three calibrated sklearn Pipelines keyed by short name."""
    base_xgb = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        scale_pos_weight=TELCO_POS_WEIGHT,
        random_state=42,
        n_jobs=-1,
    )
    base_lgb = lgb.LGBMClassifier(
        n_estimators=400,
        max_depth=-1,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        is_unbalance=True,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    base_lr = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    def wrap(base) -> Pipeline:
        """Wrap an estimator in (preprocess -> calibrated classifier)."""
        return Pipeline(
            steps=[
                ("pre", build_preprocessor()),
                (
                    "clf",
                    CalibratedClassifierCV(
                        estimator=base,
                        method="isotonic",
                        cv=5,
                    ),
                ),
            ]
        )

    return {
        "xgboost": wrap(base_xgb),
        "lightgbm": wrap(base_lgb),
        "lr": wrap(base_lr),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(
    name: str,
    pipe: Pipeline,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
) -> dict[str, Any]:
    """Score a fitted pipeline on the holdout and return a metrics dict."""
    proba = pipe.predict_proba(X_holdout)[:, 1]
    return {
        "model": name,
        "pr_auc": float(average_precision_score(y_holdout, proba)),
        "roc_auc": float(roc_auc_score(y_holdout, proba)),
        "brier": float(brier_score_loss(y_holdout, proba)),
        "holdout_n": int(len(y_holdout)),
        "holdout_positive_rate": float(y_holdout.mean()),
    }


# ---------------------------------------------------------------------------
# Cost-based threshold optimiser
# ---------------------------------------------------------------------------
def cost_based_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    contact_cost: float = 80.00,
    retention_success_rate: float = 0.20,
    avg_clv: float = 1_200.00,
) -> dict[str, float]:
    """Pick the threshold that maximises net retention benefit.

    For each candidate threshold t in [0.00, 0.01, ..., 1.00]:

        flagged   = (y_score >= t)
        n_flagged = flagged.sum()
        true_pos  = (flagged & (y_true == 1)).sum()

        outreach_cost   = n_flagged * contact_cost
        retained_value  = true_pos * retention_success_rate * avg_clv
        net_benefit     = retained_value - outreach_cost

    Returns the threshold that maximises ``net_benefit`` along with the
    full vector for the dashboard slider.
    """
    thresholds = np.round(np.linspace(0.0, 1.0, 101), 2)
    best = {"threshold": 0.5, "net_benefit": -np.inf}
    for t in thresholds:
        flagged = y_score >= t
        n_flagged = int(flagged.sum())
        true_pos = int((flagged & (y_true == 1)).sum())
        outreach_cost = n_flagged * contact_cost
        retained_value = true_pos * retention_success_rate * avg_clv
        net_benefit = retained_value - outreach_cost
        if net_benefit > best["net_benefit"]:
            best = {
                "threshold": float(t),
                "n_flagged": n_flagged,
                "true_pos": true_pos,
                "outreach_cost": float(outreach_cost),
                "retained_value": float(retained_value),
                "net_benefit": float(net_benefit),
            }
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data and reconstructing splits ...")
    X_train, y_train, X_holdout, y_holdout = get_train_and_holdout()
    print(
        f"  Train:   {len(X_train):>5} rows  ({y_train.mean() * 100:.1f}% churn)"
    )
    print(
        f"  Holdout: {len(X_holdout):>5} rows  ({y_holdout.mean() * 100:.1f}% churn)"
    )

    models = build_models()
    results: list[dict[str, Any]] = []
    holdout_scores = pd.DataFrame({"y_true": y_holdout.values})

    for name, pipe in models.items():
        print(f"\n[ {name} ] training ...")
        pipe.fit(X_train, y_train)
        metrics = evaluate(name, pipe, X_holdout, y_holdout)
        results.append(metrics)
        print(
            f"  PR-AUC = {metrics['pr_auc']:.4f}   "
            f"ROC-AUC = {metrics['roc_auc']:.4f}   "
            f"Brier = {metrics['brier']:.4f}"
        )
        joblib.dump(pipe, MODELS_DIR / f"{name}.joblib")
        holdout_scores[f"score_{name}"] = pipe.predict_proba(X_holdout)[:, 1]

    leaderboard = sorted(results, key=lambda r: r["pr_auc"], reverse=True)
    champion_name = leaderboard[0]["model"]
    champion = models[champion_name]
    joblib.dump(champion, MODELS_DIR / "champion.joblib")

    # Quick cost-based threshold on champion scores for leaderboard summary.
    champion_scores = holdout_scores[f"score_{champion_name}"].to_numpy()
    cb = cost_based_threshold(
        y_holdout.to_numpy(),
        champion_scores,
        contact_cost=80.00,
        retention_success_rate=0.20,
        avg_clv=1200.00,
    )

    print("\n" + "=" * 64)
    print(" LEADERBOARD (sorted by PR-AUC)")
    print("=" * 64)
    for r in leaderboard:
        crown = "  <- CHAMPION" if r["model"] == champion_name else ""
        print(
            f"  {r['model']:<10}  "
            f"PR-AUC={r['pr_auc']:.4f}   "
            f"ROC-AUC={r['roc_auc']:.4f}   "
            f"Brier={r['brier']:.4f}{crown}"
        )
    print("=" * 64)
    print(
        f"\n Champion threshold @ defaults ($80 contact, 20% success, $1200 CLV):"
    )
    print(
        f"   threshold      = {cb['threshold']:.2f}"
    )
    print(
        f"   flagged        = {cb['n_flagged']} / {len(y_holdout)}"
        f" ({cb['n_flagged'] / len(y_holdout) * 100:.1f}%)"
    )
    print(
        f"   true positives = {cb['true_pos']}"
    )
    print(
        f"   net benefit    = ${cb['net_benefit']:,.0f}"
    )

    # Persist leaderboard + cost-based decision + holdout scores
    leaderboard_payload = {
        "champion": champion_name,
        "results": leaderboard,
        "champion_cost_based_threshold": cb,
        "random_state": 42,
        "test_size": 0.20,
        "calibration": "isotonic, cv=5",
    }
    with open(MODELS_DIR / "leaderboard.json", "w", encoding="utf-8") as f:
        json.dump(leaderboard_payload, f, indent=2)

    holdout_scores.to_parquet(REPORTS_DIR / "holdout_scores.parquet")

    print(f"\nWrote models/champion.joblib  (= {champion_name})")
    print("Wrote models/xgboost.joblib, lightgbm.joblib, lr.joblib")
    print("Wrote models/leaderboard.json")
    print(
        f"Wrote reports/holdout_scores.parquet  ({len(holdout_scores)} rows)"
    )


if __name__ == "__main__":
    main()
