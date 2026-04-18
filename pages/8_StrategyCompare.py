"""
Strategy Comparison — calls the FastAPI backend.

Lets you run several strategies (lump sum + DCA at different frequencies)
on the same ticker/period and overlay their results.

This page deliberately talks to the API rather than importing Python directly.
That validates the API and mirrors exactly what a React frontend would do.
"""

from __future__ import annotations

import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import date

st.set_page_config(page_title="Strategy Comparison", layout="wide")
st.title("Strategy Comparison")
st.caption("Calls the FastAPI backend at localhost:8000 — make sure uvicorn is running.")

API = "http://localhost:8000/api"

# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR — shared parameters
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("Universe")
    ticker = st.text_input("Ticker", value="SPY").upper().strip()
    start  = st.date_input("Start date", value=date(2015, 1, 1))
    end    = st.date_input("End date",   value=date(2024, 12, 31))
    total  = st.number_input("Total capital ($)", value=10_000, step=1_000)
    cost   = st.number_input("Transaction cost (%)", value=0.0, step=0.05, format="%.2f") / 100

    st.divider()
    st.header("Strategies to compare")
    add_lump   = st.checkbox("Lump Sum",                    value=True)
    add_weekly = st.checkbox("DCA — Weekly",                value=False)
    add_bi     = st.checkbox("DCA — Bi-weekly",             value=False)
    add_month  = st.checkbox("DCA — Monthly",               value=True)
    add_qtr    = st.checkbox("DCA — Quarterly",             value=True)

    st.divider()
    st.subheader("All Weather")
    st.caption("SPY 30% · TLT 40% · IEF 15% · GLD 7.5% · GSG 7.5%")
    add_aw_monthly   = st.checkbox("All Weather — Monthly rebal",   value=False)
    add_aw_quarterly = st.checkbox("All Weather — Quarterly rebal", value=True)
    add_aw_annually  = st.checkbox("All Weather — Annual rebal",    value=False)

    run_btn = st.button("▶  Run comparison", use_container_width=True, type="primary")

# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

COLORS = ["#00b4d8", "#2a9d8f", "#f4a261", "#e63946", "#8338ec", "#fb5607"]


def _rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert '#rrggbb' to 'rgba(r,g,b,alpha)' — Plotly doesn't support 8-digit hex."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def _call(endpoint: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API}{endpoint}", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the API. Is `uvicorn api.main:app --reload` running?")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
        return None


def _build_payload(frequency: str | None = None) -> dict:
    base = {
        "ticker": ticker,
        "start":  start.isoformat(),
        "end":    end.isoformat(),
        "transaction_cost_pct": cost,
    }
    if frequency is None:  # lump sum
        return {**base, "amount": total}
    else:
        from research.strategies.equity import _FREQ_DAYS
        n_periods = max(1, (end - start).days // _FREQ_DAYS[frequency])
        return {**base, "periodic_amount": total / n_periods, "frequency": frequency}


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_lump_sum(ticker, start, end, total, cost):
    return _call("/research/backtest/lump-sum", _build_payload(None))

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_dca(ticker, start, end, total, cost, frequency):
    return _call("/research/backtest/dca", _build_payload(frequency))

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_all_weather(start, end, total, cost, rebalance_frequency):
    return _call("/research/backtest/all-weather", {
        "start":                start.isoformat(),
        "end":                  end.isoformat(),
        "initial_amount":       total,
        "rebalance_frequency":  rebalance_frequency,
        "transaction_cost_pct": cost,
    })


# ═══════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════

if not run_btn:
    st.info("Configure parameters in the sidebar and click **Run comparison**.")
    st.stop()

if not any([add_lump, add_weekly, add_bi, add_month, add_qtr]):
    st.warning("Select at least one strategy.")
    st.stop()

strategies: list[tuple[str, dict]] = []  # (label, result_dict)

todo_equity = []
if add_lump:   todo_equity.append(("Lump Sum",        None))
if add_weekly: todo_equity.append(("DCA — Weekly",    "weekly"))
if add_bi:     todo_equity.append(("DCA — Bi-weekly", "biweekly"))
if add_month:  todo_equity.append(("DCA — Monthly",   "monthly"))
if add_qtr:    todo_equity.append(("DCA — Quarterly", "quarterly"))

todo_aw = []
if add_aw_monthly:   todo_aw.append(("All Weather — Monthly",   "monthly"))
if add_aw_quarterly: todo_aw.append(("All Weather — Quarterly", "quarterly"))
if add_aw_annually:  todo_aw.append(("All Weather — Annual",    "annually"))

n_total = len(todo_equity) + len(todo_aw)
with st.spinner(f"Running {n_total} strategies…"):
    for label, freq in todo_equity:
        if freq is None:
            result = _fetch_lump_sum(ticker, start, end, total, cost)
        else:
            result = _fetch_dca(ticker, start, end, total, cost, freq)
        if result:
            strategies.append((label, result))

    for label, rebal_freq in todo_aw:
        result = _fetch_all_weather(start, end, total, cost, rebal_freq)
        if result:
            strategies.append((label, result))

if not strategies:
    st.stop()

# ═══════════════════════════════════════════════════════════════════════
# METRICS TABLE
# ═══════════════════════════════════════════════════════════════════════

st.subheader(f"{ticker}  ·  {start} → {end}")

cols = st.columns(len(strategies))
for col, (label, res), color in zip(cols, strategies, COLORS):
    p = res["performance"]
    col.markdown(f"**:{color.lstrip('#')} {label}**" if False else f"**{label}**")
    col.metric("Final Value",    f"${p['final_value']:,.0f}")
    col.metric("Total Invested", f"${p['total_invested']:,.0f}")
    col.metric("Total Return",   f"{p['total_return_pct']:+.1f}%")
    col.metric("CAGR",           f"{p['cagr_pct']:+.1f}%")
    col.metric("Sharpe",         f"{p['sharpe']:.2f}")
    col.metric("Max Drawdown",   f"{p['max_drawdown_pct']:.1f}%")
    col.metric("Volatility",     f"{p['volatility_ann_pct']:.1f}%")
    if p.get("avg_cost_basis"):
        col.metric("Avg Cost Basis", f"${p['avg_cost_basis']:,.2f}")

st.divider()

# ═══════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════

tab_value, tab_return, tab_dd, tab_invested = st.tabs([
    "Portfolio Value", "Return %", "Drawdown", "Invested vs Value"
])

# ── Portfolio value ───────────────────────────────────────────────────
with tab_value:
    fig = go.Figure()
    for (label, res), color in zip(strategies, COLORS):
        df = pd.DataFrame(res["daily"])
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["portfolio_value"],
            name=label, line=dict(color=color, width=2),
            hovertemplate=f"{label}<br>%{{x}}<br>$%{{y:,.0f}}<extra></extra>",
        ))
    fig.update_layout(
        yaxis_title="Portfolio Value ($)", height=420,
        hovermode="x unified", legend=dict(orientation="h", y=1.06),
        margin=dict(t=20, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Return % ─────────────────────────────────────────────────────────
with tab_return:
    fig2 = go.Figure()
    for (label, res), color in zip(strategies, COLORS):
        df = pd.DataFrame(res["daily"])
        fig2.add_trace(go.Scatter(
            x=df["date"], y=df["return_pct"],
            name=label, line=dict(color=color, width=2),
            hovertemplate=f"{label}<br>%{{x}}<br>%{{y:+.2f}}%<extra></extra>",
        ))
    fig2.add_hline(y=0, line_dash="dash", line_color="gray")
    fig2.update_layout(
        yaxis_title="Return on Invested Capital (%)", height=380,
        hovermode="x unified", legend=dict(orientation="h", y=1.06),
        margin=dict(t=20, b=30),
    )
    st.plotly_chart(fig2, use_container_width=True)


# ── Drawdown ─────────────────────────────────────────────────────────
with tab_dd:
    st.caption(
        "Drawdown = how far below the strategy's own peak it is at each point. "
        "Measures pain during the holding period."
    )
    fig3 = go.Figure()
    for (label, res), color in zip(strategies, COLORS):
        df = pd.DataFrame(res["daily"])
        values = pd.to_numeric(df["portfolio_value"])
        peak   = values.cummax()
        dd_pct = (values - peak) / peak * 100
        fig3.add_trace(go.Scatter(
            x=df["date"], y=dd_pct,
            name=label, line=dict(color=color, width=1.5),
            fill="tozeroy",
            fillcolor=_rgba(color, 0.15),
            hovertemplate=f"{label}<br>%{{x}}<br>%{{y:.2f}}%<extra></extra>",
        ))
    fig3.add_hline(y=0, line_dash="dash", line_color="gray")
    fig3.update_layout(
        yaxis_title="Drawdown (%)", height=360,
        hovermode="x unified", legend=dict(orientation="h", y=1.06),
        margin=dict(t=20, b=30),
    )
    st.plotly_chart(fig3, use_container_width=True)


# ── Invested vs Value ─────────────────────────────────────────────────
with tab_invested:
    st.caption(
        "For DCA strategies, capital is deployed gradually — "
        "the gap between 'invested' and 'value' shows the unrealized P&L at each point."
    )
    fig4 = make_subplots(
        rows=len(strategies), cols=1,
        shared_xaxes=True,
        subplot_titles=[label for label, _ in strategies],
        vertical_spacing=0.08,
    )
    for i, ((label, res), color) in enumerate(zip(strategies, COLORS), start=1):
        df = pd.DataFrame(res["daily"])
        fig4.add_trace(go.Scatter(
            x=df["date"], y=df["portfolio_value"],
            name="Value", line=dict(color=color, width=2),
            showlegend=(i == 1),
        ), row=i, col=1)
        fig4.add_trace(go.Scatter(
            x=df["date"], y=df["invested"],
            name="Invested", line=dict(color="gray", width=1.5, dash="dot"),
            showlegend=(i == 1),
        ), row=i, col=1)

    fig4.update_layout(
        height=280 * len(strategies),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02),
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# PERFORMANCE TABLE
# ═══════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("Full metrics comparison")

metric_labels = {
    "final_value":          "Final Value ($)",
    "total_invested":       "Total Invested ($)",
    "total_return_pct":     "Total Return (%)",
    "cagr_pct":             "CAGR (%)",
    "volatility_ann_pct":   "Volatility ann. (%)",
    "sharpe":               "Sharpe",
    "sortino":              "Sortino",
    "calmar":               "Calmar",
    "max_drawdown_pct":     "Max Drawdown (%)",
    "max_dd_duration_days": "Max DD Duration (days)",
    "ulcer_index":          "Ulcer Index",
    "pct_in_drawdown":      "% Time in Drawdown",
    "positive_months_pct":  "Positive Months (%)",
    "best_year_pct":        "Best Year (%)",
    "worst_year_pct":       "Worst Year (%)",
    "n_trades":             "# of Trades",
    "avg_cost_basis":       "Avg Cost Basis ($)",
    "total_rebal_cost":     "Total Rebal Cost ($)",
    "cost_drag_pct":        "Cost Drag (%)",
}

table = {}
for label, res in strategies:
    p = res["performance"]
    table[label] = {v: p.get(k) for k, v in metric_labels.items()}

df_table = pd.DataFrame(table)

def _highlight_best(row):
    """Green = best value in each row (highest for returns, least negative for drawdown)."""
    vals = pd.to_numeric(row, errors="coerce")
    if vals.isna().all():
        return [""] * len(row)
    if "Drawdown" in row.name or "Worst" in row.name:
        best_idx = vals.idxmax()   # least negative = best
    else:
        best_idx = vals.idxmax()
    return ["background-color: #d4edda" if col == best_idx else "" for col in row.index]

st.dataframe(
    df_table.style.apply(_highlight_best, axis=1),
    use_container_width=True,
)
