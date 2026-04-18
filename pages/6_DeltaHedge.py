# pages/6_DeltaHedge.py
"""
Delta-Hedged Short Straddle Backtest.

Concept
───────
Sell an ATM straddle every month. Re-delta-hedge daily using spot.
The strategy harvests the variance risk premium:
  Edge = Theta collected − Gamma paid
       = 0.5 Γ S² (σ_implied² − σ_realized²) × T

You profit whenever σ_realized < σ_implied (implied vol > realized vol).
Losses occur when spot makes large moves faster than theta accrues.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from market_data.providers.deribit_history import (
    fetch_spot_history,
    fetch_dvol_history,
    fetch_realized_vol,
)
from pricer.backtest.dh_straddle import run as run_backtest

st.set_page_config(page_title="Delta Hedge Backtest", layout="wide")
st.title("Delta-Hedged Short Straddle — Backtest")

st.caption(
    "**Strategy:** sell an ATM straddle every month, delta-hedge daily with spot/perp.  "
    "**Edge:** implied vol (DVOL) consistently exceeds 30D realized vol → "
    "theta collected > gamma paid.  "
    "**Risk:** short gamma — large spot moves eat the premium."
)

# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("Instrument")
    currency       = st.selectbox("Underlying", ["BTC", "ETH"])
    history_days   = st.selectbox("History", [180, 365, 400, 730], index=2,
                                  format_func=lambda d: f"{d} days")

    st.divider()
    st.header("Strategy Parameters")
    T_days        = st.selectbox("Straddle maturity", [7, 14, 21, 30, 45], index=3,
                                  format_func=lambda d: f"{d}D")
    rebal_freq    = st.selectbox("Rebalance frequency",
                                  [1, 2, 5, 7],
                                  format_func=lambda d: f"Every {d}D",
                                  index=0)
    notional      = st.number_input("Notional per trade ($)", value=100_000, step=10_000)

    st.divider()
    refresh = st.button("↺  Refresh data", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# DATA + BACKTEST
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def _load(currency, history_days):
    spot = fetch_spot_history(currency, history_days + 40)["close"]
    dvol = fetch_dvol_history(currency, history_days + 40)
    rv30 = fetch_realized_vol(currency, history_days + 40, 30)
    return spot, dvol, rv30

if refresh:
    st.cache_data.clear()

with st.spinner(f"Loading {currency} data…"):
    try:
        spot, dvol, rv30 = _load(currency, history_days)
    except Exception as exc:
        st.error(f"Data error: {exc}")
        st.stop()

with st.spinner("Running backtest…"):
    daily, trades = run_backtest(
        spot, dvol,
        T_days=T_days,
        rebalance_freq=rebal_freq,
        notional_usd=notional,
    )

if daily.empty or trades.empty:
    st.warning("Not enough data for the selected parameters.")
    st.stop()

# ─── Summary stats ─────────────────────────────────────────────────
total_pnl   = daily["total_pnl"].sum()
theta_total = daily["theta_pnl"].sum()
gamma_total = daily["gamma_pnl"].sum()
vega_total  = daily["vega_pnl"].sum()
cum_pnl     = daily["total_pnl"].cumsum()
drawdown    = cum_pnl - cum_pnl.cummax()
max_dd      = drawdown.min()
sharpe      = daily["total_pnl"].mean() / daily["total_pnl"].std() * np.sqrt(252)
win_rate    = (trades["P&L ($)"] > 0).mean() * 100
n_trades    = len(trades)
avg_pnl     = trades["P&L ($)"].mean()

# ── Top metrics ────────────────────────────────────────────────────
c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Total P&L",         f"${total_pnl:,.0f}")
c2.metric("Sharpe (ann.)",     f"{sharpe:.2f}")
c3.metric("Max Drawdown",      f"${max_dd:,.0f}")
c4.metric("Win Rate",          f"{win_rate:.0f}%", delta=f"{n_trades} trades")
c5.metric("Avg P&L / trade",   f"${avg_pnl:,.0f}")
c6.metric("Theta − Gamma",     f"${theta_total+gamma_total:,.0f}",
          delta=f"θ ${theta_total:,.0f}  γ ${gamma_total:,.0f}")

st.divider()

# ═══════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════

tab_pnl, tab_decomp, tab_greek, tab_trades = st.tabs([
    "P&L", "Theta vs Gamma", "Greeks", "Trade Log"
])


# ── Tab 1: Cumulative P&L ─────────────────────────────────────────────
with tab_pnl:
    st.subheader("Cumulative P&L & Drawdown")

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.04,
        subplot_titles=("Cumulative P&L ($)", "Daily P&L ($)", "Drawdown ($)"),
    )

    # shade individual trades
    for _, tr in trades.iterrows():
        for row in [1, 2, 3]:
            fig.add_vrect(
                x0=str(tr["Entry Date"]), x1=str(tr["Exit Date"]),
                fillcolor="#2a9d8f", opacity=0.06, line_width=0,
                row=row, col=1,
            )

    fig.add_trace(go.Scatter(
        x=cum_pnl.index, y=cum_pnl,
        name="Cum P&L", line=dict(color="#2a9d8f", width=2.5),
        fill="tozeroy", fillcolor="rgba(42,157,143,0.1)"),
        row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

    colors_daily = ["#2a9d8f" if v >= 0 else "#e63946" for v in daily["total_pnl"]]
    fig.add_trace(go.Bar(
        x=daily.index, y=daily["total_pnl"],
        name="Daily P&L", marker_color=colors_daily, showlegend=False),
        row=2, col=1)

    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown,
        name="Drawdown", line=dict(color="#e63946", width=1.5),
        fill="tozeroy", fillcolor="rgba(230,57,70,0.1)"),
        row=3, col=1)

    fig.update_layout(height=620, margin=dict(t=40, b=20),
                      hovermode="x unified", legend=dict(orientation="h", y=1.04))
    st.plotly_chart(fig, use_container_width=True)

    # Implied vs realized vol overlay
    st.subheader("Implied vs Realized Vol — the Edge")
    st.caption(
        "Green area = positive variance risk premium (IV > RV) = you profit.  "
        "Red area = negative premium (RV > IV) = spot moved too fast, gamma hurts."
    )

    iv_aligned = dvol.reindex(daily.index)
    rv_aligned = rv30.reindex(daily.index)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=daily.index, y=iv_aligned,
                              name="DVOL (Implied)", line=dict(color="#f4a261", width=2)))
    fig2.add_trace(go.Scatter(x=daily.index, y=rv_aligned,
                              name="30D Realized Vol", line=dict(color="#00b4d8", width=2)))
    # fill between
    fig2.add_trace(go.Scatter(
        x=list(daily.index) + list(daily.index[::-1]),
        y=list(iv_aligned.fillna(method="ffill")) + list(rv_aligned.fillna(method="ffill")[::-1]),
        fill="toself",
        fillcolor="rgba(42,157,143,0.15)",
        line=dict(width=0), showlegend=False, name="Premium",
    ))
    fig2.update_layout(height=280, margin=dict(t=20, b=30),
                       yaxis_title="Vol (%)", hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True)


# ── Tab 2: Theta vs Gamma decomposition ──────────────────────────────
with tab_decomp:
    st.subheader("P&L Decomposition: Theta vs Gamma vs Vega")
    st.caption(
        "**Theta** (green): time decay you collect each day — always positive for short vol.  \n"
        "**Gamma** (red): cost of spot moves — always negative, proportional to (ΔS)².  \n"
        "**Vega** (orange): P&L from IV changes — profit if vol drops, hurts if vol spikes.  \n"
        "**Net** (blue): theta + gamma + vega ≈ actual P&L.  \n\n"
        "The strategy is profitable whenever cumulative theta > cumulative |gamma|."
    )

    # Rolling 30D cumulative of each component
    window = min(30, T_days)
    cum_theta = daily["theta_pnl"].cumsum()
    cum_gamma = daily["gamma_pnl"].cumsum()
    cum_vega  = daily["vega_pnl"].cumsum()
    cum_total = daily["total_pnl"].cumsum()

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=cum_theta.index, y=cum_theta,
                              name="Θ Theta", line=dict(color="#2a9d8f", width=2)))
    fig3.add_trace(go.Scatter(x=cum_gamma.index, y=cum_gamma,
                              name="Γ Gamma", line=dict(color="#e63946", width=2)))
    fig3.add_trace(go.Scatter(x=cum_vega.index, y=cum_vega,
                              name="𝒱 Vega", line=dict(color="#f4a261", width=2)))
    fig3.add_trace(go.Scatter(x=cum_total.index, y=cum_total,
                              name="Net P&L", line=dict(color="#00b4d8", width=2.5, dash="dot")))
    fig3.add_hline(y=0, line_dash="dash", line_color="gray")
    fig3.update_layout(height=360, margin=dict(t=10, b=30),
                       yaxis_title="Cumulative ($)", hovermode="x unified",
                       legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig3, use_container_width=True)

    # Bar: total per component
    st.subheader("Total P&L by Source")
    components = {
        "Theta (collected)":  theta_total,
        "Gamma (paid)":       gamma_total,
        "Vega (vol changes)": vega_total,
        "Net P&L":            total_pnl,
    }
    bar_colors = ["#2a9d8f", "#e63946", "#f4a261", "#00b4d8"]
    fig4 = go.Figure(go.Bar(
        x=list(components.keys()),
        y=list(components.values()),
        marker_color=bar_colors,
        text=[f"${v:,.0f}" for v in components.values()],
        textposition="outside",
    ))
    fig4.add_hline(y=0, line_dash="dash", line_color="gray")
    fig4.update_layout(height=320, margin=dict(t=10, b=30), yaxis_title="Total $ over period")
    st.plotly_chart(fig4, use_container_width=True)

    # Daily realized return vs implied
    st.subheader("Daily |Return| vs Implied Breakeven")
    st.caption(
        "Each dot is one day. Above the line = spot moved more than the straddle's "
        "daily breakeven → gamma hurt more than theta helped that day."
    )
    be_daily = daily["iv"] / 100 / np.sqrt(252) * 100   # daily breakeven move %
    fig5 = go.Figure()
    colors_pts = ["#e63946" if r > b else "#2a9d8f"
                  for r, b in zip(daily["rv_daily_pct"], be_daily)]
    fig5.add_trace(go.Scatter(
        x=daily.index, y=daily["rv_daily_pct"],
        mode="markers", name="|Daily Return| %",
        marker=dict(color=colors_pts, size=4, opacity=0.7),
    ))
    fig5.add_trace(go.Scatter(
        x=daily.index, y=be_daily,
        name="Daily breakeven (IV/√252)", line=dict(color="gray", dash="dash", width=1.5),
    ))
    fig5.update_layout(height=280, margin=dict(t=10, b=30),
                       yaxis_title="|Return| (%)", hovermode="x unified")
    st.plotly_chart(fig5, use_container_width=True)


# ── Tab 3: Greeks over time ───────────────────────────────────────────
with tab_greek:
    st.subheader("Greeks During Active Positions")
    st.caption(
        "**Delta**: should stay near 0 between rebalances — spikes show when rebalancing is needed.  "
        "**Gamma**: your short-gamma exposure (higher = more sensitive to spot moves).  "
        "**Vega**: your IV exposure (negative for short straddle — you lose if vol rises)."
    )

    fig6 = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.33, 0.33, 0.33],
        vertical_spacing=0.06,
        subplot_titles=("Delta (BTC)", "Gamma ($/BTC²)", "Vega ($/vol-pt)"),
    )
    fig6.add_trace(go.Scatter(x=daily.index, y=daily["delta"],
                              name="Δ Delta", line=dict(color="#00b4d8", width=1.5)),
                   row=1, col=1)
    fig6.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

    fig6.add_trace(go.Scatter(x=daily.index, y=-daily["gamma"],  # flip: show as positive exposure
                              name="Γ Gamma exposure", line=dict(color="#e63946", width=1.5),
                              fill="tozeroy", fillcolor="rgba(230,57,70,0.1)"),
                   row=2, col=1)

    fig6.add_trace(go.Scatter(x=daily.index, y=-daily["vega"],   # flip: negative for short
                              name="𝒱 Vega exposure", line=dict(color="#f4a261", width=1.5),
                              fill="tozeroy", fillcolor="rgba(244,162,97,0.1)"),
                   row=3, col=1)

    fig6.update_layout(height=560, margin=dict(t=40, b=20), hovermode="x unified",
                       legend=dict(orientation="h", y=1.04))
    st.plotly_chart(fig6, use_container_width=True)


# ── Tab 4: Trade log ──────────────────────────────────────────────────
with tab_trades:
    st.subheader("Trade Log")

    def _color_trades(row):
        c = "background-color: #d4edda" if row["Win"] else "background-color: #f8d7da"
        return [c] * len(row)

    st.dataframe(
        trades.style.apply(_color_trades, axis=1),
        use_container_width=True, hide_index=True,
    )

    # P&L distribution
    st.subheader("P&L Distribution per Trade")
    fig7 = go.Figure()
    fig7.add_trace(go.Histogram(
        x=trades["P&L ($)"], nbinsx=max(5, n_trades // 2),
        marker_color="#2a9d8f", opacity=0.8, name="P&L",
    ))
    fig7.add_vline(x=0, line_dash="dash", line_color="gray")
    fig7.add_vline(x=avg_pnl, line_dash="dot", line_color="#f4a261",
                   annotation_text=f"Mean ${avg_pnl:,.0f}", annotation_position="top right")
    fig7.update_layout(height=280, margin=dict(t=10, b=30),
                       xaxis_title="P&L ($)", yaxis_title="# Trades")
    st.plotly_chart(fig7, use_container_width=True)

    st.info(
        "**Reading the results:**  \n"
        "- Consistent wins → variance risk premium is real and stable  \n"
        "- Large losses in specific months → check what vol did (tab 1) — likely a vol spike  \n"
        "- If win rate is high but Sharpe is low → tail risk is the issue, consider stop-loss rules  \n"
        "- Rebalancing less often (5D, 7D) reduces transaction costs but increases delta drift"
    )
