from __future__ import annotations
from pathlib import Path
import pandas as pd

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)

    # Normalize column names: trim spaces and keep original case, but store a lowercase map
    df.columns = [str(c).strip() for c in df.columns]
    return df

def resolve_target_col(df: pd.DataFrame, user_value: str | None, default: str = "Churn") -> str:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    # if user provided something, resolve case-insensitively
    if user_value:
        uv = user_value.strip()
        if uv in cols:
            return uv
        if uv.lower() in lower_map:
            return lower_map[uv.lower()]

    # fallback to default (also case-insensitive)
    if default in cols:
        return default
    if default.lower() in lower_map:
        return lower_map[default.lower()]

    # last resort: common variants
    for cand in ["churn", "Churn", "Exited", "exit", "target", "label"]:
        if cand in cols:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    raise KeyError(f"Target column not found. Available columns: {cols}")
