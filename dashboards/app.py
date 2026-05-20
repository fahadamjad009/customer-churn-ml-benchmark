"""
Customer Churn Retention Platform — Streamlit dashboard.

Consultant-tier interface in the Institutional Dark aesthetic
(BlackRock Aladdin / Citi Velocity / Bloomberg Terminal lineage).

Architecture
------------
Pure read-only at runtime. No sklearn / xgboost / lightgbm / shap imports
during user interaction. Every panel reads precomputed artifacts from
``reports/``:

  * ``business_metrics.json``     — master, includes the 101-point sweep
  * ``decile_lift_table.csv``     — top-decile + cumulative gains
  * ``segment_metrics.json``      — Contract / tenure / Payment / Internet
  * ``calibration_curve.json``    — reliability points + Brier + KS
  * ``shap_global.json``          — ranked feature importance
  * ``models/leaderboard.json``   — model comparison

The Campaign Simulator slider drives an indexed lookup into the sweep
arrays (``int(round(t * 100))``) and recomputes net benefit in pure NumPy
from user-set economic params — no model inference at any user input.

Run locally
-----------
    streamlit run dashboards/app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Retention Platform",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"About": "Customer Churn ML Benchmark — Fahad Amjad"},
)

# ---------------------------------------------------------------------------
# Institutional Dark — design tokens
# ---------------------------------------------------------------------------
BG_BASE = "#0B1220"
BG_SURFACE = "#131C2E"
BG_ELEVATED = "#1A2436"
BORDER_SUBTLE = "#1F2A3F"
BORDER_DEFAULT = "#2B3854"
TEXT_PRIMARY = "#E8ECF4"
TEXT_SECONDARY = "#9BA8BF"
TEXT_TERTIARY = "#6B7791"
ACCENT_GOLD = "#C9A961"
ACCENT_BLUE = "#4A90D9"
ACCENT_PURPLE = "#8B7AB8"
COLOR_SUCCESS = "#5BA77D"
COLOR_WARNING = "#D49B5B"
COLOR_DANGER = "#C45A5A"

# ---------------------------------------------------------------------------
# Plotly Institutional template
# ---------------------------------------------------------------------------
pio.templates["institutional"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=BG_SURFACE,
        font=dict(
            color=TEXT_PRIMARY,
            family='"Inter", "IBM Plex Sans", system-ui, sans-serif',
            size=12,
        ),
        title=dict(
            font=dict(color=TEXT_PRIMARY, size=15, family='"Inter", sans-serif'),
            x=0.0,
            xanchor="left",
            pad=dict(l=4, b=8),
        ),
        xaxis=dict(
            gridcolor=BORDER_SUBTLE,
            zerolinecolor=BORDER_DEFAULT,
            tickcolor=TEXT_TERTIARY,
            tickfont=dict(color=TEXT_SECONDARY, size=11),
            title=dict(font=dict(color=TEXT_SECONDARY, size=11)),
            linecolor=BORDER_DEFAULT,
        ),
        yaxis=dict(
            gridcolor=BORDER_SUBTLE,
            zerolinecolor=BORDER_DEFAULT,
            tickcolor=TEXT_TERTIARY,
            tickfont=dict(color=TEXT_SECONDARY, size=11),
            title=dict(font=dict(color=TEXT_SECONDARY, size=11)),
            linecolor=BORDER_DEFAULT,
        ),
        colorway=[
            ACCENT_GOLD,
            ACCENT_BLUE,
            ACCENT_PURPLE,
            COLOR_SUCCESS,
            COLOR_WARNING,
            COLOR_DANGER,
        ],
        legend=dict(
            font=dict(color=TEXT_SECONDARY, size=11),
            bgcolor="rgba(19,28,46,0.85)",
            bordercolor=BORDER_SUBTLE,
            borderwidth=1,
        ),
        margin=dict(t=44, r=20, b=44, l=56),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor=BG_ELEVATED,
            bordercolor=BORDER_DEFAULT,
            font=dict(color=TEXT_PRIMARY, size=12),
        ),
    )
)
pio.templates.default = "institutional"

# ---------------------------------------------------------------------------
# Global CSS — refines the .streamlit/config.toml dark base
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    /* Root tokens (referenced by custom classes below) */
    :root {{
        --bg-base: {BG_BASE};
        --bg-surface: {BG_SURFACE};
        --bg-elevated: {BG_ELEVATED};
        --border-subtle: {BORDER_SUBTLE};
        --border-default: {BORDER_DEFAULT};
        --text-primary: {TEXT_PRIMARY};
        --text-secondary: {TEXT_SECONDARY};
        --text-tertiary: {TEXT_TERTIARY};
        --accent: {ACCENT_GOLD};
        --accent-blue: {ACCENT_BLUE};
    }}

    /* App background */
    .stApp {{ background-color: var(--bg-base); }}

    /* Hide Streamlit branding */
    #MainMenu, header, footer {{ visibility: hidden; }}

    /* Headings */
    h1, h2, h3, h4 {{
        color: var(--text-primary) !important;
        font-family: 'Inter', 'IBM Plex Sans', system-ui, sans-serif !important;
        letter-spacing: -0.01em;
        font-weight: 600;
    }}
    h1 {{ font-size: 28px; margin-bottom: 4px; }}
    h2 {{ font-size: 20px; }}
    h3 {{ font-size: 15px; color: var(--text-secondary) !important; font-weight: 500; }}

    /* Eyebrow / section label */
    .eyebrow {{
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--text-tertiary);
        margin: 24px 0 8px 0;
    }}

    /* Subtitle under H1 */
    .subtitle {{
        color: var(--text-secondary);
        font-size: 14px;
        margin-bottom: 28px;
        max-width: 720px;
    }}

    /* st.metric — institutional card style */
    [data-testid="stMetric"] {{
        background-color: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 4px;
        padding: 18px 20px;
    }}
    [data-testid="stMetricLabel"] p {{
        color: var(--text-tertiary) !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase;
    }}
    [data-testid="stMetricValue"] {{
        color: var(--accent) !important;
        font-size: 26px !important;
        font-weight: 600 !important;
        font-family: 'IBM Plex Mono', 'Consolas', monospace !important;
        font-feature-settings: "tnum" !important;
    }}
    [data-testid="stMetricDelta"] {{
        color: var(--text-secondary) !important;
        font-size: 12px !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        background: transparent;
        border-bottom: 1px solid var(--border-default);
        padding: 0;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        color: var(--text-secondary) !important;
        padding: 12px 22px !important;
        border-bottom: 2px solid transparent !important;
        font-weight: 500 !important;
        font-size: 13px !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: var(--accent) !important;
        border-bottom-color: var(--accent) !important;
        background: transparent !important;
    }}

    /* Sliders */
    .stSlider [role="slider"] {{
        background-color: var(--accent) !important;
        border-color: var(--accent) !important;
    }}

    /* Dataframes / tables */
    [data-testid="stDataFrame"] {{
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 4px;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: var(--bg-surface);
        border-right: 1px solid var(--border-subtle);
    }}

    /* Buttons */
    .stButton button {{
        background: var(--bg-elevated) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-default) !important;
        border-radius: 4px !important;
        font-weight: 500 !important;
    }}
    .stButton button:hover {{
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }}

    /* Markdown paragraphs in the body */
    .stMarkdown p {{ color: var(--text-secondary); font-size: 14px; }}

    /* Caption styling */
    .caption {{
        color: var(--text-tertiary);
        font-size: 12px;
        margin-top: 4px;
    }}

    /* "Headline number" — for hero stats above the fold */
    .headline {{
        font-family: 'IBM Plex Mono', 'Consolas', monospace;
        font-size: 36px;
        font-weight: 600;
        color: var(--accent);
        font-feature-settings: "tnum";
        line-height: 1.0;
    }}
    .headline-label {{
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--text-tertiary);
        margin-top: 8px;
    }}

    /* Tag pill */
    .pill {{
        display: inline-block;
        padding: 3px 10px;
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        border-radius: 12px;
        font-size: 11px;
        color: var(--text-secondary);
        margin-right: 6px;
        font-weight: 500;
    }}
    .pill-accent {{
        background: rgba(201,169,97,0.08);
        border-color: rgba(201,169,97,0.4);
        color: var(--accent);
    }}

    /* Divider — hairline */
    hr {{
        border: none;
        border-top: 1px solid var(--border-subtle);
        margin: 20px 0;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data loaders — cached, read-only
# ---------------------------------------------------------------------------
REPORTS = Path("reports")
MODELS = Path("models")


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    with open(REPORTS / "business_metrics.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_decile() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "decile_lift_table.csv")


@st.cache_data(show_spinner=False)
def load_segments() -> dict:
    with open(REPORTS / "segment_metrics.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_calibration() -> dict:
    with open(REPORTS / "calibration_curve.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_shap_global() -> dict:
    with open(REPORTS / "shap_global.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_leaderboard() -> dict:
    with open(MODELS / "leaderboard.json", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def fmt_money(value: float, *, compact: bool = False) -> str:
    if compact:
        if abs(value) >= 1_000_000:
            return f"${value/1_000_000:,.1f}M"
        if abs(value) >= 1_000:
            return f"${value/1_000:,.1f}K"
    return f"${value:,.0f}"


def fmt_pct(value: float, decimals: int = 1) -> str:
    return f"{value*100:.{decimals}f}%"


def clean_feature_name(raw: str) -> str:
    """Turn ColumnTransformer feature names into human-readable labels.

    ``num__tenure``                    -> ``tenure``
    ``cat__Contract_Month-to-month``   -> ``Contract: Month-to-month``
    """
    if raw.startswith("num__"):
        return raw[5:]
    if raw.startswith("cat__"):
        rest = raw[5:]
        if "_" in rest:
            col, val = rest.split("_", 1)
            return f"{col}: {val}"
        return rest
    return raw


def recompute_sweep_net(
    sweep: dict,
    *,
    contact_cost: float,
    success_rate: float,
    avg_clv_override: float | None = None,
) -> np.ndarray:
    """Recompute net-benefit array from the precomputed sweep + user inputs.

    If ``avg_clv_override`` is None, uses the actual per-customer CLV stored
    in the sweep (``clv_captured``). Otherwise applies a uniform avg CLV
    against true_positives — the "what if every customer had this CLV?" view.
    """
    n_flagged = np.array(sweep["n_flagged"])
    true_positives = np.array(sweep["true_positives"])
    clv_captured = np.array(sweep["clv_captured"])

    outreach_cost = n_flagged * contact_cost
    if avg_clv_override is None:
        retained_value = clv_captured * success_rate
    else:
        retained_value = true_positives * success_rate * avg_clv_override
    return retained_value - outreach_cost


# ---------------------------------------------------------------------------
# Load everything once
# ---------------------------------------------------------------------------
try:
    metrics = load_metrics()
    decile_df = load_decile()
    segments = load_segments()
    calibration = load_calibration()
    shap_global = load_shap_global()
    leaderboard = load_leaderboard()
except FileNotFoundError as e:
    st.error(
        f"Missing artifact: `{e.filename}`. Run the analytics pipeline first:\n\n"
        f"```bash\npython -m churn.train\npython -m analytics.business_metrics\n```"
    )
    st.stop()


sweep = metrics["tier6_threshold_sweep"]
thresholds = np.array(sweep["thresholds"])
default_contact_cost = float(metrics["defaults"]["contact_cost"])
default_success_rate = float(metrics["defaults"]["success_rate"])
avg_clv = float(metrics["avg_clv"])
holdout_n = int(metrics["holdout_n"])
holdout_churn_rate = float(metrics["holdout_churn_rate"])


# ===========================================================================
# Header
# ===========================================================================
st.markdown(
    f"""
    <div style="display:flex; align-items:baseline; gap:12px; margin-bottom:0;">
      <h1 style="margin:0; color:{TEXT_PRIMARY};">Customer Churn Retention Platform</h1>
      <span class="pill pill-accent">{leaderboard["champion"].upper()} CHAMPION</span>
      <span class="pill">CALIBRATED · ISOTONIC · CV-5</span>
    </div>
    <div class="subtitle">
      End-to-end churn ML platform with CLV-weighted business metrics, cost-based
      decisioning, segment analysis, and SHAP attribution. Built on the IBM Telco
      Customer Churn benchmark ({holdout_n:,} customers in holdout, {holdout_churn_rate*100:.1f}% churn rate).
    </div>
    """,
    unsafe_allow_html=True,
)

tab_exec, tab_sim, tab_decile, tab_seg, tab_model, tab_drivers = st.tabs(
    [
        "Executive Summary",
        "Campaign Simulator",
        "Decile Lift",
        "Segments",
        "Model Performance",
        "Calibration & Drivers",
    ]
)


# ===========================================================================
# TAB 1 — Executive Summary
# ===========================================================================
with tab_exec:
    st.markdown('<div class="eyebrow">Headline Performance</div>', unsafe_allow_html=True)

    tier1 = metrics["tier1_business_impact"]
    optimal = metrics["optimal_threshold"]
    decile_1 = decile_df.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top-decile precision", f"{decile_1['precision']*100:.0f}%",
              f"{decile_1['lift']:.1f}× lift over base")
    c2.metric("CLV catch rate", fmt_pct(tier1["catch_rate_clv"], 1),
              f"{fmt_money(tier1['clv_captured'], compact=True)} captured")
    c3.metric("Net benefit", fmt_money(tier1["net_benefit"], compact=True),
              f"on {holdout_n:,}-row holdout")
    c4.metric("Campaign ROI", f"{tier1['campaign_roi']*100:.0f}%",
              f"@ threshold {optimal['threshold']:.2f}")

    st.markdown('<div class="eyebrow" style="margin-top:32px;">Operating Point @ defaults</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="caption">contact cost = {fmt_money(default_contact_cost)} · '
        f'success rate = {fmt_pct(default_success_rate, 0)} · '
        f'avg CLV = {fmt_money(avg_clv)} (24-month horizon).</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Threshold", f"{optimal['threshold']:.2f}")
    c2.metric("Customers flagged",
              f"{optimal['n_flagged']:,}",
              f"{optimal['n_flagged']/holdout_n*100:.1f}% of holdout")
    c3.metric("True positives", f"{optimal['true_positives']:,}",
              f"{optimal['precision']*100:.0f}% precision")
    c4.metric("Recall (count)", fmt_pct(optimal["recall"], 1))

    st.markdown('<div class="eyebrow" style="margin-top:32px;">Cumulative Gains</div>',
                unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=decile_df["decile"], y=decile_df["cumulative_recall"] * 100,
        mode="lines+markers", name="Captured churners",
        line=dict(color=ACCENT_GOLD, width=2.5),
        marker=dict(size=7),
        hovertemplate="Decile %{x}<br>Captured: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=decile_df["decile"], y=decile_df["cumulative_clv_share"] * 100,
        mode="lines+markers", name="Captured at-risk CLV",
        line=dict(color=ACCENT_BLUE, width=2.5, dash="dot"),
        marker=dict(size=7),
        hovertemplate="Decile %{x}<br>CLV: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[1, 10], y=[10, 100],
        mode="lines", name="Random",
        line=dict(color=TEXT_TERTIARY, width=1, dash="dash"),
        hoverinfo="skip",
    ))
    fig.update_layout(
        height=380,
        xaxis=dict(title="Decile (1 = highest score)", dtick=1),
        yaxis=dict(title="Cumulative %", range=[0, 105]),
        legend=dict(orientation="h", y=-0.2, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# TAB 2 — Campaign Simulator (THE differentiator)
# ===========================================================================
with tab_sim:
    st.markdown('<div class="eyebrow">Campaign Simulator</div>', unsafe_allow_html=True)
    st.markdown(
        "Set your retention-team economics; the simulator picks the threshold "
        "that maximises net benefit and shows the operating point live. "
        "Recomputed in pure NumPy against the precomputed 101-point threshold "
        "sweep — no model inference."
    )

    sim_left, sim_right = st.columns([1, 1.4])

    with sim_left:
        st.markdown('<div class="eyebrow" style="margin-top:0;">Economic Inputs</div>',
                    unsafe_allow_html=True)
        contact_cost = st.slider(
            "Contact cost per customer ($)",
            min_value=10.0, max_value=200.0,
            value=float(default_contact_cost), step=5.0,
        )
        success_rate = st.slider(
            "Retention success rate",
            min_value=0.05, max_value=0.50,
            value=float(default_success_rate), step=0.01,
            format="%.2f",
        )
        clv_mode = st.radio(
            "CLV basis",
            options=["Actual per-customer CLV", "Uniform CLV override"],
            index=0,
            horizontal=False,
        )
        if clv_mode == "Uniform CLV override":
            avg_clv_override = st.slider(
                "Uniform CLV per customer ($)",
                min_value=500.0, max_value=3500.0,
                value=float(avg_clv), step=50.0,
            )
        else:
            avg_clv_override = None

        # Recompute
        nets = recompute_sweep_net(
            sweep,
            contact_cost=contact_cost,
            success_rate=success_rate,
            avg_clv_override=avg_clv_override,
        )
        optimal_idx = int(np.argmax(nets))
        optimal_t = float(thresholds[optimal_idx])
        optimal_net = float(nets[optimal_idx])
        optimal_n_flagged = int(sweep["n_flagged"][optimal_idx])
        optimal_tp = int(sweep["true_positives"][optimal_idx])
        optimal_clv_captured = float(sweep["clv_captured"][optimal_idx])
        outreach_cost = optimal_n_flagged * contact_cost
        roi = (optimal_net / outreach_cost) if outreach_cost > 0 else 0.0

        st.markdown('<div class="eyebrow" style="margin-top:24px;">Operating Point</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Optimal threshold", f"{optimal_t:.2f}")
        c2.metric("Net benefit", fmt_money(optimal_net, compact=True))
        c1, c2 = st.columns(2)
        c1.metric("Flagged", f"{optimal_n_flagged:,}",
                  f"{optimal_n_flagged/holdout_n*100:.1f}% of {holdout_n:,}")
        c2.metric("Campaign ROI", f"{roi*100:.0f}%")
        c1, c2 = st.columns(2)
        c1.metric("Churners caught", f"{optimal_tp:,}")
        c2.metric("Outreach cost", fmt_money(outreach_cost, compact=True))

    with sim_right:
        st.markdown('<div class="eyebrow" style="margin-top:0;">Net Benefit vs Threshold</div>',
                    unsafe_allow_html=True)

        fig = go.Figure()
        # Shaded zero crossing
        fig.add_hline(y=0, line=dict(color=BORDER_DEFAULT, width=1, dash="dot"))
        fig.add_trace(go.Scatter(
            x=thresholds, y=nets,
            mode="lines",
            line=dict(color=ACCENT_GOLD, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(201,169,97,0.08)",
            name="Net benefit",
            hovertemplate="Threshold %{x:.2f}<br>Net $%{y:,.0f}<extra></extra>",
        ))
        # Optimal marker
        fig.add_trace(go.Scatter(
            x=[optimal_t], y=[optimal_net],
            mode="markers",
            marker=dict(color=ACCENT_BLUE, size=14, symbol="diamond",
                        line=dict(color=TEXT_PRIMARY, width=1.5)),
            name=f"Optimal (t={optimal_t:.2f})",
            hovertemplate=f"<b>Optimal</b><br>t = {optimal_t:.2f}<br>"
                         f"Net = ${optimal_net:,.0f}<extra></extra>",
        ))
        fig.update_layout(
            height=420,
            xaxis=dict(title="Threshold", tickformat=".2f", range=[0, 1]),
            yaxis=dict(title="Net benefit ($)", tickprefix="$", tickformat=",.0f"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f'<div class="caption">At the optimum: '
            f'flag {optimal_n_flagged:,} customers '
            f'({optimal_n_flagged/holdout_n*100:.1f}% of holdout), '
            f'catch {optimal_tp:,} churners '
            f'({optimal_tp/(optimal_n_flagged or 1)*100:.0f}% precision), '
            f'retain {fmt_money(optimal_clv_captured * success_rate, compact=True)} of CLV.</div>',
            unsafe_allow_html=True,
        )


# ===========================================================================
# TAB 3 — Decile Lift
# ===========================================================================
with tab_decile:
    st.markdown('<div class="eyebrow">Decile Lift Table</div>', unsafe_allow_html=True)
    st.markdown(
        "Customers sorted by churn score descending, split into 10 equal-size "
        "buckets. Decile 1 = top 10% scored. Lift is precision divided by the "
        "overall base rate ({:.1f}%).".format(holdout_churn_rate * 100)
    )

    c1, c2 = st.columns([1.2, 1])
    with c1:
        # Bar chart of lift per decile
        colors_bar = [
            ACCENT_GOLD if d <= 3 else ACCENT_BLUE if d <= 6 else BORDER_DEFAULT
            for d in decile_df["decile"]
        ]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=decile_df["decile"],
            y=decile_df["lift"],
            marker_color=colors_bar,
            text=[f"{v:.2f}×" for v in decile_df["lift"]],
            textposition="outside",
            textfont=dict(color=TEXT_PRIMARY, size=11),
            hovertemplate="Decile %{x}<br>Lift %{y:.2f}×<extra></extra>",
        ))
        fig.add_hline(
            y=1.0, line=dict(color=TEXT_TERTIARY, width=1, dash="dash"),
            annotation_text="Base rate", annotation_position="top right",
            annotation_font=dict(color=TEXT_TERTIARY, size=11),
        )
        fig.update_layout(
            height=380,
            title="Lift over base rate by decile",
            xaxis=dict(title="Decile", dtick=1),
            yaxis=dict(title="Lift (multiplier)", rangemode="tozero"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Cumulative recall area chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=decile_df["decile"],
            y=decile_df["cumulative_recall"] * 100,
            mode="lines+markers",
            line=dict(color=ACCENT_GOLD, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(201,169,97,0.08)",
            hovertemplate="Top %{x} deciles<br>Captured %{y:.1f}%<extra></extra>",
            showlegend=False,
        ))
        fig.update_layout(
            height=380,
            title="Cumulative churners captured",
            xaxis=dict(title="Top-K deciles", dtick=1),
            yaxis=dict(title="Cumulative recall (%)", range=[0, 105]),
        )
        st.plotly_chart(fig, use_container_width=True)

    # The table itself
    st.markdown('<div class="eyebrow" style="margin-top:24px;">Per-decile detail</div>',
                unsafe_allow_html=True)
    display_df = decile_df.copy()
    display_df["precision"] = display_df["precision"].map(lambda v: f"{v*100:.1f}%")
    display_df["lift"] = display_df["lift"].map(lambda v: f"{v:.2f}×")
    display_df["cumulative_recall"] = display_df["cumulative_recall"].map(lambda v: f"{v*100:.1f}%")
    display_df["cumulative_clv_share"] = display_df["cumulative_clv_share"].map(lambda v: f"{v*100:.1f}%")
    display_df.columns = [
        "Decile", "Customers", "Churners captured",
        "Precision", "Lift", "Cum. recall", "Cum. CLV captured",
    ]
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ===========================================================================
# TAB 4 — Segments
# ===========================================================================
with tab_seg:
    st.markdown('<div class="eyebrow">Segment Performance</div>', unsafe_allow_html=True)
    st.markdown(
        "Churn rate, mean score, and model recall by customer segment. "
        "Sorted by churn rate within each axis."
    )

    seg_axes = list(segments.keys())
    seg_choice = st.selectbox(
        "Segment axis",
        seg_axes,
        format_func=lambda x: x.replace("_", " ").title(),
        index=0,
    )
    seg_rows = pd.DataFrame(segments[seg_choice])

    c1, c2 = st.columns([1.2, 1])

    with c1:
        # Bar chart of churn rate per segment value, with overall line
        bars = seg_rows.sort_values("churn_rate", ascending=True)
        bar_colors = []
        for cr in bars["churn_rate"]:
            if cr > holdout_churn_rate * 1.3:
                bar_colors.append(COLOR_DANGER)
            elif cr > holdout_churn_rate:
                bar_colors.append(ACCENT_GOLD)
            else:
                bar_colors.append(ACCENT_BLUE)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=bars["value"],
            x=bars["churn_rate"] * 100,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v*100:.1f}%" for v in bars["churn_rate"]],
            textposition="outside",
            textfont=dict(color=TEXT_PRIMARY, size=11),
            hovertemplate="%{y}<br>Churn rate %{x:.1f}%<extra></extra>",
        ))
        fig.add_vline(
            x=holdout_churn_rate * 100,
            line=dict(color=TEXT_TERTIARY, width=1, dash="dash"),
            annotation_text=f"Overall {holdout_churn_rate*100:.1f}%",
            annotation_position="top right",
            annotation_font=dict(color=TEXT_TERTIARY, size=11),
        )
        fig.update_layout(
            height=max(360, 60 * len(bars) + 80),
            title=f"Churn rate by {seg_choice.replace('_', ' ').title()}",
            xaxis=dict(title="Churn rate (%)", ticksuffix="%"),
            yaxis=dict(title=""),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Segment recall vs churn rate scatter
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=seg_rows["churn_rate"] * 100,
            y=seg_rows["segment_recall"] * 100,
            mode="markers+text",
            marker=dict(
                size=np.clip(np.sqrt(seg_rows["n_customers"].astype(float)) * 1.5, 8, 40),
                color=ACCENT_GOLD,
                line=dict(color=TEXT_PRIMARY, width=1),
                opacity=0.85,
            ),
            text=seg_rows["value"],
            textposition="top center",
            textfont=dict(color=TEXT_SECONDARY, size=10),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Churn rate %{x:.1f}%<br>"
                "Recall %{y:.1f}%<br>"
                "n = %{marker.size:.0f}<extra></extra>"
            ),
        ))
        fig.update_layout(
            height=max(360, 60 * len(seg_rows) + 80),
            title="Segment recall vs churn rate",
            xaxis=dict(title="Segment churn rate (%)", ticksuffix="%"),
            yaxis=dict(title="Model recall on segment (%)", ticksuffix="%"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Detail table
    st.markdown('<div class="eyebrow" style="margin-top:24px;">Detail</div>', unsafe_allow_html=True)
    det = seg_rows[[
        "value", "n_customers", "churn_rate", "lift_vs_overall",
        "mean_score", "flagged", "true_positives", "segment_recall",
        "clv_total", "clv_captured",
    ]].copy()
    det["churn_rate"] = det["churn_rate"].map(lambda v: f"{v*100:.1f}%")
    det["lift_vs_overall"] = det["lift_vs_overall"].map(lambda v: f"{v:.2f}×")
    det["mean_score"] = det["mean_score"].map(lambda v: f"{v:.3f}")
    det["segment_recall"] = det["segment_recall"].map(lambda v: f"{v*100:.1f}%")
    det["clv_total"] = det["clv_total"].map(lambda v: f"${v:,.0f}")
    det["clv_captured"] = det["clv_captured"].map(lambda v: f"${v:,.0f}")
    det.columns = [
        "Value", "Customers", "Churn rate", "Lift vs overall",
        "Mean score", "Flagged", "True positives", "Recall",
        "CLV total", "CLV captured",
    ]
    st.dataframe(det, use_container_width=True, hide_index=True)


# ===========================================================================
# TAB 5 — Model Performance
# ===========================================================================
with tab_model:
    st.markdown('<div class="eyebrow">Model Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        "Three calibrated classifiers (`CalibratedClassifierCV(method='isotonic', cv=5)`). "
        f"Champion = **{leaderboard['champion']}** by PR-AUC. "
        "All three within 0.006 PR-AUC of each other — within calibration noise on a 7k-row tabular dataset. "
        "LR is kept as champion for interpretability; gradient boosting models remain as alternates."
    )

    rows = pd.DataFrame(leaderboard["results"])
    rows["champion"] = rows["model"] == leaderboard["champion"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=rows["model"],
        y=rows["pr_auc"],
        marker_color=[ACCENT_GOLD if c else BORDER_DEFAULT for c in rows["champion"]],
        text=[f"{v:.4f}" for v in rows["pr_auc"]],
        textposition="outside",
        textfont=dict(color=TEXT_PRIMARY, size=11),
        name="PR-AUC",
        hovertemplate="%{x}<br>PR-AUC %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        height=320,
        title="PR-AUC by model (champion highlighted)",
        xaxis=dict(title=""),
        yaxis=dict(title="PR-AUC", range=[0, max(rows["pr_auc"]) * 1.15]),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.markdown('<div class="eyebrow" style="margin-top:16px;">Leaderboard</div>',
                unsafe_allow_html=True)
    tbl = rows[["model", "pr_auc", "roc_auc", "brier", "holdout_n", "holdout_positive_rate"]].copy()
    tbl["pr_auc"] = tbl["pr_auc"].map(lambda v: f"{v:.4f}")
    tbl["roc_auc"] = tbl["roc_auc"].map(lambda v: f"{v:.4f}")
    tbl["brier"] = tbl["brier"].map(lambda v: f"{v:.4f}")
    tbl["holdout_positive_rate"] = tbl["holdout_positive_rate"].map(lambda v: f"{v*100:.1f}%")
    tbl["model"] = tbl.apply(
        lambda r: f"★ {r['model']}" if r["model"] == leaderboard["champion"] else r["model"],
        axis=1,
    )
    tbl.columns = ["Model", "PR-AUC", "ROC-AUC", "Brier", "Holdout N", "Holdout +rate"]
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    st.markdown(
        '<div class="caption">Random_state=42, test_size=0.20, stratified by target. '
        'Class imbalance: xgboost <code>scale_pos_weight=2.77</code>, '
        'lightgbm <code>is_unbalance=True</code>, lr <code>class_weight="balanced"</code>. '
        'Calibration corrects the probability scale these adjustments distort.</div>',
        unsafe_allow_html=True,
    )


# ===========================================================================
# TAB 6 — Calibration & Drivers
# ===========================================================================
with tab_drivers:
    st.markdown('<div class="eyebrow">Probability Calibration</div>', unsafe_allow_html=True)
    st.markdown(
        "Reliability curve — predicted vs observed churn rate in 10 score-quantile bins. "
        "A perfectly calibrated model lies on the dashed diagonal."
    )

    cal_rows = pd.DataFrame(calibration["reliability_curve"])
    c1, c2 = st.columns([1.2, 1])

    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines", name="Perfect calibration",
            line=dict(color=TEXT_TERTIARY, width=1, dash="dash"),
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=cal_rows["predicted_mean"], y=cal_rows["observed_rate"],
            mode="lines+markers", name="Champion",
            line=dict(color=ACCENT_GOLD, width=2.5),
            marker=dict(size=10, color=ACCENT_GOLD,
                        line=dict(color=TEXT_PRIMARY, width=1)),
            hovertemplate=(
                "Bin %{customdata}<br>"
                "Predicted %{x:.3f}<br>"
                "Observed %{y:.3f}<extra></extra>"
            ),
            customdata=cal_rows["bin"],
        ))
        fig.update_layout(
            height=380,
            title="Reliability curve",
            xaxis=dict(title="Mean predicted probability", range=[0, 1], tickformat=".2f"),
            yaxis=dict(title="Observed churn rate", range=[0, 1], tickformat=".2f"),
            legend=dict(orientation="h", y=-0.2, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="eyebrow" style="margin-top:0;">Quality Metrics</div>',
                    unsafe_allow_html=True)
        st.metric("Brier score", f"{calibration['brier']:.4f}",
                  "lower = better calibrated")
        st.metric("KS statistic", f"{calibration['ks']:.3f}",
                  "score separation; > 0.5 is strong")
        st.markdown(
            '<div class="caption" style="margin-top:12px;">'
            'Brier measures squared error between predicted probability and outcome — '
            'low Brier (<0.16 on Telco) signals well-calibrated probabilities suitable '
            'for the cost-based threshold optimiser. KS measures the maximum vertical '
            'separation between churner / non-churner score CDFs.</div>',
            unsafe_allow_html=True,
        )

    # SHAP global importance
    st.markdown('<div class="eyebrow" style="margin-top:32px;">Top Churn Drivers (SHAP)</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Mean absolute SHAP value per feature, computed on an uncalibrated XGBoost "
        "trained on the full training set (avoids `CalibratedClassifierCV` "
        "inner-classifier ambiguity). Higher = more influence on the prediction."
    )

    shap_rows = pd.DataFrame(shap_global["top_features"])
    shap_rows["feature_clean"] = shap_rows["feature"].map(clean_feature_name)
    shap_rows = shap_rows.sort_values("mean_abs_shap", ascending=True)  # for horizontal bar

    fig = go.Figure()
    n = len(shap_rows)
    bar_colors = [
        ACCENT_GOLD if i >= n - 3 else
        ACCENT_BLUE if i >= n - 8 else
        BORDER_DEFAULT
        for i in range(n)
    ]
    fig.add_trace(go.Bar(
        y=shap_rows["feature_clean"],
        x=shap_rows["mean_abs_shap"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.3f}" for v in shap_rows["mean_abs_shap"]],
        textposition="outside",
        textfont=dict(color=TEXT_PRIMARY, size=11),
        hovertemplate="<b>%{y}</b><br>mean |SHAP| %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        height=max(360, 26 * n + 80),
        title=f"Top {n} drivers (gold = top 3)",
        xaxis=dict(title="Mean |SHAP| value"),
        yaxis=dict(title=""),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f'<div class="caption">Reference model: {shap_global["shap_reference_model"]}. '
        f'Computed across {shap_global["n_features_total"]} total features '
        f'(numeric + one-hot-encoded categoricals). The top three drivers are '
        f'engineered features (`contract_months`, `charge_ratio`, '
        f'`avg_charges_per_month`) rather than raw columns — feature engineering '
        f'pays off.</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div style="margin-top:48px; padding-top:16px; border-top:1px solid {BORDER_SUBTLE};
                color:{TEXT_TERTIARY}; font-size:11px; display:flex;
                justify-content:space-between; align-items:center;">
      <div>Customer Churn Retention Platform · IBM Telco benchmark
        ({holdout_n:,} holdout / {holdout_churn_rate*100:.1f}% churn) · Champion {leaderboard['champion']}</div>
      <div>Built by Fahad Amjad · MIT 2026</div>
    </div>
    """,
    unsafe_allow_html=True,
)
