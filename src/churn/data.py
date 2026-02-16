from __future__ import annotations

from pathlib import Path
import pandas as pd
import urllib.request

TELCO_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

def ensure_telco_csv(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(TELCO_URL, path)
    return path

def load_csv(path: Path) -> pd.DataFrame:
    # If user uses default location, auto-fetch it on first run (Streamlit Cloud friendly)
    if str(path).replace("\\", "/") == "data/raw/churn.csv":
        ensure_telco_csv(path)

    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def resolve_target_col(df: pd.DataFrame, user_value: str | None, default: str = "Churn") -> str:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    if user_value:
        uv = user_value.strip()
        if uv in cols:
            return uv
        if uv.lower() in lower_map:
            return lower_map[uv.lower()]

    if default in cols:
        return default
    if default.lower() in lower_map:
        return lower_map[default.lower()]

    for cand in ["churn", "Churn", "Exited", "exit", "target", "label"]:
        if cand in cols:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    raise KeyError(f"Target column not found. Available columns: {cols}")
