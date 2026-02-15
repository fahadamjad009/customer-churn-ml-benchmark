from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on PYTHONPATH so `import src...` works under Streamlit
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from src.churn.data import load_csv, resolve_target_col
from src.churn.train import train_all_models
from src.churn.eval import summarize_results, plot_confusion_matrix, plot_roc_curve

st.set_page_config(page_title="Customer Churn ML Benchmark", layout="wide")

st.title("Customer Churn ML Benchmark")
st.caption("Benchmark classic ML models for churn prediction with an interactive dashboard.")

with st.sidebar:
    st.header("Data")
    csv_path = st.text_input("CSV path", value="data/raw/churn.csv")
    test_size = st.slider("Test size", 0.05, 0.5, 0.2)
    random_state = st.number_input("Random state", min_value=0, value=42, step=1)

df = None
try:
    df = load_csv(Path(csv_path))
    st.success("Dataset loaded")
except Exception as e:
    st.error(f"Could not load dataset: {e}")

if df is not None:
    auto_target = resolve_target_col(df, user_value=None, default="Churn")

    with st.sidebar:
        target_col = st.text_input("Target column", value=auto_target)

    try:
        target_col = resolve_target_col(df, user_value=target_col, default="Churn")
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.subheader("Dataset preview")
    st.dataframe(df.head(10), use_container_width=True)

    with st.sidebar:
        st.header("Training")
        go = st.button("Train / Re-train models", type="primary")

    if go:
        with st.spinner("Training models..."):
            results = train_all_models(
                df,
                target_col=target_col,
                test_size=float(test_size),
                random_state=int(random_state),
                save_dir=Path("models"),
            )
        st.session_state["results"] = results

    results = st.session_state.get("results")
    if results:
        st.subheader("Model leaderboard")
        st.dataframe(summarize_results(results), use_container_width=True)

        st.subheader("Inspect model")
        chosen = st.selectbox("Model", list(results.keys()), index=0)

        st.subheader("Confusion matrix")
        st.pyplot(plot_confusion_matrix(results[chosen]), clear_figure=True)

        st.subheader("ROC curve")
        st.pyplot(plot_roc_curve(results[chosen]), clear_figure=True)
