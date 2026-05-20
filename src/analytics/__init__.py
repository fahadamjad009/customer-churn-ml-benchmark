"""Customer churn — consultant-tier business analytics layer.

Pure analytics modules that consume training artifacts
(``reports/holdout_scores.parquet`` + per-model joblibs) and produce
business-oriented outputs (decile lift, CLV-weighted retention math,
segment performance, SHAP drivers, threshold sweep).

Outputs are written to ``reports/`` and consumed by the Streamlit dashboard
at runtime with no ML library calls — the cloud-safe pattern.
"""
