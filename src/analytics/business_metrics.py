"""
Customer churn — consultant-tier business metrics.

Six-tier analytics framework that consumes training artifacts
(``reports/holdout_scores.parquet`` + ``models/champion.joblib``) and
produces the full set of outputs the Streamlit dashboard reads at runtime.

Tiers
-----
**Tier 1 — Business Impact (dollars).**
    Per-customer 24-month CLV from ``MonthlyCharges * 24``. CLV-weighted
    catch rate, retained CLV $, contact cost, net retention benefit,
    campaign ROI at the cost-based optimal threshold.

**Tier 2 — Operations.**
    Full decile lift table via ``pd.qcut`` on the score (descending).
    Cumulative gains in both count-space and CLV-space. Retention-team
    capacity math (top-K customers reachable per agent).

**Tier 3 — Model Quality.**
    Brier score, score Kolmogorov-Smirnov statistic (separation), and a
    calibration / reliability curve (predicted vs observed probability in
    10 quantile bins).

**Tier 4 — Segments.**
    Performance breakdown by Contract * tenure_bucket * PaymentMethod *
    InternetService. Each segment's churn rate, captured churners at the
    optimal threshold, and lift over the segment base rate.

**Tier 5 — Drivers (SHAP).**
    SHAP global feature importance plus per-customer SHAP values. Computed
    on a separately-trained uncalibrated XGBoost (single model, fit on the
    full training set) — not the calibrated champion. This avoids
    extracting SHAP values from the inner classifiers of
    ``CalibratedClassifierCV`` and gives stable, interpretable
    attributions. Trade-off documented in the README.

**Tier 6 — Threshold Sweep (cloud-safe).**
    101-point sweep across thresholds 0.00 -> 1.00 in 0.01 steps with
    ``n_flagged``, ``true_positives``, ``false_positives``, ``precision``,
    ``recall``, and ``clv_captured`` precomputed at every threshold. The
    dashboard's Campaign Simulator does an indexed lookup at the slider
    value and live-recomputes net benefit from user-set contact cost,
    success rate, and CLV — zero ML library calls at runtime.

Outputs
-------
``reports/business_metrics.json``
    Master JSON containing every numeric output. Includes the embedded
    101-point sweep (~80 KB).

``reports/decile_lift_table.csv``
    The decile lift table, one row per decile.

``reports/segment_metrics.json``
    Per-segment performance (Contract, tenure_bucket, PaymentMethod,
    InternetService).

``reports/calibration_curve.json``
    Reliability-curve data points for the dashboard's calibration plot.

``reports/shap_global.json``
    Mean-absolute-SHAP per feature, ranked.

``reports/shap_per_customer.parquet``
    Per-customer SHAP value matrix for the dashboard drill-down view.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import brier_score_loss

from churn.features import add_engineered_features
from churn.holdout import get_holdout, get_train_and_holdout
from churn.preprocess import build_preprocessor

REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")

# Tier-1 economic defaults — kept in sync with train.py
DEFAULT_CONTACT_COST: float = 80.00
DEFAULT_SUCCESS_RATE: float = 0.20
CLV_HORIZON_MONTHS: int = 24


# ===========================================================================
# Tier 1 — Business Impact (CLV + retention math)
# ===========================================================================
def compute_clv(
    df: pd.DataFrame,
    horizon_months: int = CLV_HORIZON_MONTHS,
) -> pd.Series:
    """Per-customer 24-month forward CLV.

    Simple heuristic: ``MonthlyCharges * horizon_months``. Stronger
    survival-based estimates (e.g. KaplanMeier fit on tenure x Churn) are
    deferred — documented as a future extension in the README.
    """
    return (df["MonthlyCharges"].astype(float) * horizon_months).rename("clv")


def compute_tier1_business_impact(
    y_true: np.ndarray,
    y_score: np.ndarray,
    clv: np.ndarray,
    *,
    threshold: float,
    contact_cost: float = DEFAULT_CONTACT_COST,
    success_rate: float = DEFAULT_SUCCESS_RATE,
) -> dict[str, float]:
    """CLV-weighted retention math at a given threshold."""
    flagged = y_score >= threshold
    n_flagged = int(flagged.sum())
    true_pos = int(((y_true == 1) & flagged).sum())
    false_pos = int(((y_true == 0) & flagged).sum())

    clv_total = float(clv.sum())
    clv_at_risk = float(clv[y_true == 1].sum())
    clv_captured = float(clv[(y_true == 1) & flagged].sum())
    clv_missed = float(clv[(y_true == 1) & ~flagged].sum())

    outreach_cost = n_flagged * contact_cost
    retained_value = clv_captured * success_rate
    net_benefit = retained_value - outreach_cost
    campaign_roi = (net_benefit / outreach_cost) if outreach_cost > 0 else 0.0

    catch_rate_count = (true_pos / max(int((y_true == 1).sum()), 1))
    catch_rate_clv = (clv_captured / max(clv_at_risk, 1.0))

    return {
        "threshold": float(threshold),
        "n_flagged": n_flagged,
        "true_positives": true_pos,
        "false_positives": false_pos,
        "alert_rate": n_flagged / len(y_true),
        "precision": (true_pos / n_flagged) if n_flagged else 0.0,
        "recall": catch_rate_count,
        "catch_rate_clv": catch_rate_clv,
        "clv_total": clv_total,
        "clv_at_risk": clv_at_risk,
        "clv_captured": clv_captured,
        "clv_missed": clv_missed,
        "outreach_cost": float(outreach_cost),
        "retained_value": float(retained_value),
        "net_benefit": float(net_benefit),
        "campaign_roi": float(campaign_roi),
        "contact_cost": float(contact_cost),
        "success_rate": float(success_rate),
    }


# ===========================================================================
# Tier 2 — Operations (decile lift, cumulative gains)
# ===========================================================================
def compute_decile_lift(
    y_true: np.ndarray,
    y_score: np.ndarray,
    clv: np.ndarray,
) -> pd.DataFrame:
    """Full decile table sorted by score descending.

    Each row: decile rank (1=top), customer count, captured churners,
    precision, lift over base, cumulative recall, cumulative CLV captured.
    """
    df = pd.DataFrame(
        {"y_true": y_true, "score": y_score, "clv": clv}
    ).sort_values("score", ascending=False).reset_index(drop=True)

    df["decile"] = pd.qcut(
        df.index.values, q=10, labels=range(1, 11)
    ).astype(int)

    base_rate = float(y_true.mean())
    total_pos = int(y_true.sum())
    total_clv_at_risk = float(clv[y_true == 1].sum())

    out_rows = []
    cum_pos = 0
    cum_clv = 0.0
    for d in range(1, 11):
        chunk = df[df["decile"] == d]
        n = len(chunk)
        pos = int(chunk["y_true"].sum())
        clv_at_risk = float(chunk.loc[chunk["y_true"] == 1, "clv"].sum())
        cum_pos += pos
        cum_clv += clv_at_risk
        out_rows.append(
            {
                "decile": d,
                "n_customers": n,
                "churners_captured": pos,
                "precision": pos / n if n else 0.0,
                "lift": (pos / n) / base_rate if base_rate > 0 else 0.0,
                "cumulative_recall": cum_pos / max(total_pos, 1),
                "cumulative_clv_share": cum_clv / max(total_clv_at_risk, 1.0),
            }
        )
    return pd.DataFrame(out_rows)


# ===========================================================================
# Tier 3 — Model Quality (Brier, KS, reliability curve)
# ===========================================================================
def compute_score_ks(
    y_true: np.ndarray, y_score: np.ndarray
) -> float:
    """KS statistic: max separation between score distributions of
    churners vs non-churners. Closer to 1.0 = better separation.
    """
    scores_pos = np.sort(y_score[y_true == 1])
    scores_neg = np.sort(y_score[y_true == 0])
    if len(scores_pos) == 0 or len(scores_neg) == 0:
        return 0.0
    grid = np.linspace(0.0, 1.0, 200)
    cdf_pos = np.searchsorted(scores_pos, grid) / len(scores_pos)
    cdf_neg = np.searchsorted(scores_neg, grid) / len(scores_neg)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def compute_reliability_curve(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10
) -> list[dict[str, float]]:
    """Quantile-binned predicted vs observed probability."""
    df = pd.DataFrame({"y_true": y_true, "score": y_score})
    # Use rank-based quantiles for stable bin sizes
    df["bin"] = pd.qcut(df["score"].rank(method="first"), q=n_bins, labels=False)
    rows: list[dict[str, float]] = []
    for b in range(n_bins):
        chunk = df[df["bin"] == b]
        if len(chunk) == 0:
            continue
        rows.append(
            {
                "bin": int(b),
                "n": int(len(chunk)),
                "predicted_mean": float(chunk["score"].mean()),
                "observed_rate": float(chunk["y_true"].mean()),
                "score_lower": float(chunk["score"].min()),
                "score_upper": float(chunk["score"].max()),
            }
        )
    return rows


# ===========================================================================
# Tier 4 — Segments
# ===========================================================================
def compute_segments(
    holdout_df: pd.DataFrame,
    y_score: np.ndarray,
    clv: np.ndarray,
    *,
    threshold: float,
    segment_columns: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Per-segment performance at the optimal threshold."""
    if segment_columns is None:
        segment_columns = [
            "Contract",
            "tenure_bucket",
            "PaymentMethod",
            "InternetService",
        ]

    y_true = holdout_df["churn_target"].to_numpy()
    flagged = y_score >= threshold
    base_rate = float(y_true.mean())

    result: dict[str, list[dict[str, Any]]] = {}
    for col in segment_columns:
        rows: list[dict[str, Any]] = []
        for value in holdout_df[col].dropna().unique():
            mask = holdout_df[col].values == value
            n = int(mask.sum())
            if n == 0:
                continue
            seg_y = y_true[mask]
            seg_score = y_score[mask]
            seg_flagged = flagged[mask]
            seg_clv = clv[mask]
            seg_pos = int(seg_y.sum())
            seg_tp = int(((seg_y == 1) & seg_flagged).sum())
            seg_churn_rate = float(seg_y.mean())
            rows.append(
                {
                    "value": str(value),
                    "n_customers": n,
                    "churn_rate": seg_churn_rate,
                    "lift_vs_overall": (
                        seg_churn_rate / base_rate if base_rate > 0 else 0.0
                    ),
                    "mean_score": float(seg_score.mean()),
                    "flagged": int(seg_flagged.sum()),
                    "true_positives": seg_tp,
                    "segment_recall": seg_tp / max(seg_pos, 1),
                    "clv_total": float(seg_clv.sum()),
                    "clv_captured": float(seg_clv[(seg_y == 1) & seg_flagged].sum()),
                }
            )
        rows.sort(key=lambda r: r["churn_rate"], reverse=True)
        result[col] = rows
    return result


# ===========================================================================
# Tier 5 — Drivers (SHAP)
# ===========================================================================
def fit_shap_reference_model(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[xgb.XGBClassifier, np.ndarray, list[str]]:
    """Train an uncalibrated XGBoost on the full training set for SHAP.

    SHAP on ``CalibratedClassifierCV`` is ambiguous (5 inner classifiers,
    one per CV fold). A single XGBoost fitted on the full train set gives
    stable per-customer attributions that match the spirit of the champion
    model (similar feature importance) without the calibration wrapper.

    Returns the fitted model, the transformed train matrix, and the list
    of feature names emitted by the ColumnTransformer.
    """
    pre = build_preprocessor()
    X_train_arr = pre.fit_transform(X_train)
    feature_names = list(pre.get_feature_names_out())

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        scale_pos_weight=2.77,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_arr, y_train)
    return model, X_train_arr, feature_names


def compute_shap(
    shap_model: xgb.XGBClassifier,
    pre,
    X_holdout: pd.DataFrame,
    feature_names: list[str],
    top_k: int = 15,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run TreeExplainer on the holdout and return global + per-customer
    SHAP values.

    Global: list of {feature, mean_abs_shap} sorted descending, top_k entries.
    Per-customer: DataFrame of shape (n_holdout, n_features) with column
    names from the preprocessor's feature_names_out.
    """
    X_holdout_arr = pre.transform(X_holdout)
    explainer = shap.TreeExplainer(shap_model)
    shap_values = explainer.shap_values(X_holdout_arr)

    # Newer SHAP returns ndarray directly for XGBClassifier; older returns list.
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]  # positive class
    if shap_values.ndim == 3:  # (n, n_features, n_classes) in some versions
        shap_values = shap_values[..., -1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = sorted(
        zip(feature_names, mean_abs.tolist()),
        key=lambda p: p[1],
        reverse=True,
    )
    top = ranked[:top_k]

    global_summary = {
        "top_features": [
            {"feature": name, "mean_abs_shap": float(val)} for name, val in top
        ],
        "n_features_total": len(feature_names),
        "shap_reference_model": "uncalibrated XGBoost, single fit on full train set",
    }
    per_customer = pd.DataFrame(shap_values, columns=feature_names)
    return global_summary, per_customer


# ===========================================================================
# Tier 6 — Threshold Sweep (cloud-safe)
# ===========================================================================
def compute_threshold_sweep(
    y_true: np.ndarray,
    y_score: np.ndarray,
    clv: np.ndarray,
) -> dict[str, list[float]]:
    """101-point threshold sweep with all components needed for live
    dashboard recomputation.

    Returns arrays of length 101 indexed by threshold t = 0.00 ... 1.00 in
    0.01 steps. The dashboard slider does ``int(round(threshold * 100))``
    and looks up every component at that index, then computes
    ``net_benefit = clv_captured * success_rate - n_flagged * contact_cost``
    on the fly.
    """
    thresholds = np.round(np.linspace(0.0, 1.0, 101), 2)
    n_flagged: list[int] = []
    true_positives: list[int] = []
    false_positives: list[int] = []
    precisions: list[float] = []
    recalls: list[float] = []
    clv_captured: list[float] = []

    total_pos = int((y_true == 1).sum())
    for t in thresholds:
        flagged = y_score >= t
        nf = int(flagged.sum())
        tp = int(((y_true == 1) & flagged).sum())
        fp = int(((y_true == 0) & flagged).sum())
        n_flagged.append(nf)
        true_positives.append(tp)
        false_positives.append(fp)
        precisions.append(tp / nf if nf else 0.0)
        recalls.append(tp / max(total_pos, 1))
        clv_captured.append(float(clv[(y_true == 1) & flagged].sum()))

    return {
        "thresholds": thresholds.tolist(),
        "n_flagged": n_flagged,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "precision": precisions,
        "recall": recalls,
        "clv_captured": clv_captured,
    }


def pick_optimal_threshold(
    sweep: dict[str, list[float]],
    *,
    avg_clv: float,
    success_rate: float = DEFAULT_SUCCESS_RATE,
    contact_cost: float = DEFAULT_CONTACT_COST,
) -> dict[str, Any]:
    """Walk the precomputed sweep and pick the net-benefit-maximising
    threshold under given economic params. Returns the operating point.
    """
    best_idx = 0
    best_net = -np.inf
    for i, _ in enumerate(sweep["thresholds"]):
        nf = sweep["n_flagged"][i]
        clv_cap = sweep["clv_captured"][i]
        net = clv_cap * success_rate - nf * contact_cost
        if net > best_net:
            best_net = net
            best_idx = i
    return {
        "threshold": float(sweep["thresholds"][best_idx]),
        "index": int(best_idx),
        "n_flagged": int(sweep["n_flagged"][best_idx]),
        "true_positives": int(sweep["true_positives"][best_idx]),
        "false_positives": int(sweep["false_positives"][best_idx]),
        "precision": float(sweep["precision"][best_idx]),
        "recall": float(sweep["recall"][best_idx]),
        "clv_captured": float(sweep["clv_captured"][best_idx]),
        "net_benefit": float(best_net),
        "contact_cost": float(contact_cost),
        "success_rate": float(success_rate),
        "avg_clv": float(avg_clv),
    }


# ===========================================================================
# Main orchestration
# ===========================================================================
def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading holdout artifacts ...")
    X_holdout, y_holdout, raw_holdout = get_holdout()
    scores_df = pd.read_parquet(REPORTS_DIR / "holdout_scores.parquet")
    with open(MODELS_DIR / "leaderboard.json", "r", encoding="utf-8") as f:
        leaderboard = json.load(f)

    champion_name = leaderboard["champion"]
    champion_scores = scores_df[f"score_{champion_name}"].to_numpy()
    y_true = y_holdout.to_numpy()

    clv_series = compute_clv(raw_holdout)
    clv = clv_series.to_numpy()
    avg_clv = float(clv.mean())
    print(
        f"  Champion: {champion_name}    "
        f"Holdout: {len(y_true)} rows    "
        f"Avg CLV: ${avg_clv:,.0f}"
    )

    # --- Tier 6 first so we can derive the optimal threshold -----------------
    print("\n[Tier 6] computing 101-point threshold sweep ...")
    sweep = compute_threshold_sweep(y_true, champion_scores, clv)
    optimal = pick_optimal_threshold(
        sweep,
        avg_clv=avg_clv,
        success_rate=DEFAULT_SUCCESS_RATE,
        contact_cost=DEFAULT_CONTACT_COST,
    )
    print(
        f"  Optimal threshold = {optimal['threshold']:.2f}    "
        f"flag {optimal['n_flagged']}/{len(y_true)} "
        f"({optimal['n_flagged'] / len(y_true) * 100:.1f}%)    "
        f"net = ${optimal['net_benefit']:,.0f}"
    )

    # --- Tier 1 at the optimal threshold ------------------------------------
    print("\n[Tier 1] business impact at the optimal threshold ...")
    tier1 = compute_tier1_business_impact(
        y_true,
        champion_scores,
        clv,
        threshold=optimal["threshold"],
        contact_cost=DEFAULT_CONTACT_COST,
        success_rate=DEFAULT_SUCCESS_RATE,
    )
    print(
        f"  CLV captured = ${tier1['clv_captured']:,.0f} / "
        f"${tier1['clv_at_risk']:,.0f} at-risk "
        f"({tier1['catch_rate_clv'] * 100:.1f}%)"
    )
    print(
        f"  Net benefit  = ${tier1['net_benefit']:,.0f}    "
        f"Campaign ROI = {tier1['campaign_roi'] * 100:.0f}%"
    )

    # --- Tier 2 -------------------------------------------------------------
    print("\n[Tier 2] decile lift table ...")
    decile_df = compute_decile_lift(y_true, champion_scores, clv)
    decile_df.to_csv(REPORTS_DIR / "decile_lift_table.csv", index=False)
    top1 = decile_df.iloc[0]
    print(
        f"  Decile 1: {int(top1['churners_captured'])} churners captured, "
        f"precision {top1['precision'] * 100:.0f}%, "
        f"lift {top1['lift']:.1f}x"
    )

    # --- Tier 3 -------------------------------------------------------------
    print("\n[Tier 3] model quality (Brier, KS, reliability) ...")
    brier = float(brier_score_loss(y_true, champion_scores))
    ks = compute_score_ks(y_true, champion_scores)
    reliability = compute_reliability_curve(y_true, champion_scores, n_bins=10)
    with open(REPORTS_DIR / "calibration_curve.json", "w", encoding="utf-8") as f:
        json.dump(
            {"reliability_curve": reliability, "brier": brier, "ks": ks},
            f,
            indent=2,
        )
    print(f"  Brier = {brier:.4f}    KS = {ks:.3f}")

    # --- Tier 4 -------------------------------------------------------------
    print("\n[Tier 4] segment performance ...")
    raw_holdout_with_target = raw_holdout.copy()
    if "churn_target" not in raw_holdout_with_target.columns:
        raw_holdout_with_target["churn_target"] = y_true
    segments = compute_segments(
        raw_holdout_with_target,
        champion_scores,
        clv,
        threshold=optimal["threshold"],
    )
    with open(REPORTS_DIR / "segment_metrics.json", "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2)
    print(
        f"  Computed {len(segments)} segment axes: "
        f"{', '.join(segments.keys())}"
    )

    # --- Tier 5 -------------------------------------------------------------
    print("\n[Tier 5] SHAP drivers (training reference XGB on full train) ...")
    X_train, y_train, _, _ = get_train_and_holdout()
    shap_model, _, feature_names = fit_shap_reference_model(X_train, y_train)
    shap_pre = build_preprocessor()
    shap_pre.fit(X_train)
    shap_global, shap_per_customer = compute_shap(
        shap_model, shap_pre, X_holdout, feature_names
    )
    with open(REPORTS_DIR / "shap_global.json", "w", encoding="utf-8") as f:
        json.dump(shap_global, f, indent=2)
    shap_per_customer.to_parquet(REPORTS_DIR / "shap_per_customer.parquet")
    top3 = shap_global["top_features"][:3]
    print(
        "  Top 3 drivers: "
        + ", ".join(f"{t['feature']} ({t['mean_abs_shap']:.3f})" for t in top3)
    )

    # --- Master JSON --------------------------------------------------------
    print("\nWriting master business_metrics.json ...")
    master = {
        "champion_model": champion_name,
        "holdout_n": int(len(y_true)),
        "holdout_churn_rate": float(y_true.mean()),
        "avg_clv": avg_clv,
        "defaults": {
            "contact_cost": DEFAULT_CONTACT_COST,
            "success_rate": DEFAULT_SUCCESS_RATE,
            "clv_horizon_months": CLV_HORIZON_MONTHS,
        },
        "optimal_threshold": optimal,
        "tier1_business_impact": tier1,
        "tier3_model_quality": {"brier": brier, "ks": ks},
        "tier6_threshold_sweep": sweep,
    }
    with open(REPORTS_DIR / "business_metrics.json", "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2)

    print("\nDone. Wrote:")
    print("  reports/business_metrics.json        (master, includes 101-pt sweep)")
    print("  reports/decile_lift_table.csv        (10 deciles)")
    print("  reports/segment_metrics.json         (Contract / tenure / Payment / Internet)")
    print("  reports/calibration_curve.json       (reliability + Brier + KS)")
    print("  reports/shap_global.json             (top 15 features)")
    print("  reports/shap_per_customer.parquet    (1409 x feature SHAP matrix)")


if __name__ == "__main__":
    main()
