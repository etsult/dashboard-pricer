# pages/4_VolStrategy.py
"""
ETH/BTC Vol Premium Strategy — Backtest & Risk Monitor.

The edge
────────
ETH implied vol (DVOL) trades structurally 10-30% richer than BTC DVOL.
But ETH and BTC are highly correlated (ρ ≈ 0.7–0.9).
→ The IV premium is systematically overstated vs the realized vol premium.
→ Harvest by selling ETH vol / buying BTC vol, vega-neutral, when spread is rich.

P&L model  (simplified var-swap analogy)
────────────────────────────────────────
  Daily MTM = −ΔETH_DVOL + ΔBTC_DVOL       (short ETH vol, long BTC vol)
  Cumulative = spread_entry − spread_exit   (profit when IV premium narrows)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from market_data.providers.deribit_history import (
    fetch_dvol_pair,
    fetch_realized_vol,
    fetch_spot_history,
)

st.set_page_config(page_title="Vol Strategy", layout="wide")
st.title("ETH/BTC Vol Premium Strategy")

# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("Data")
    days     = st.selectbox("History", [365, 500, 730], index=2, format_func=lambda d: f"{d} days")
    rv_win   = st.selectbox("Realized vol window", [14, 21, 30], index=2, format_func=lambda w: f"{w}D")
    refresh  = st.button("↺  Refresh", use_container_width=True)

    st.divider()
    st.header("Backtest Parameters")
    z_window  = st.slider("Z-score window (days)", 30, 120, 60, 5)
    entry_z   = st.slider("Entry z-score",          0.5, 3.0, 1.5, 0.1)
    exit_z    = st.slider("Exit z-score",           -2.0, 1.0, 0.0, 0.1)
    max_hold  = st.slider("Max hold (days)",         5, 60, 21, 1)
    vega_notl = st.number_input("Vega notional ($k per vol-pt)", value=10, step=1,
                                help="P&L per 1 vol-point move in the spread")


# ═══════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def _load(days: int, rv_win: int):
    dvol = fetch_dvol_pair(days)
    rv_btc = fetch_realized_vol("BTC", days, rv_win).rename("BTC_rv")
    rv_eth = fetch_realized_vol("ETH", days, rv_win).rename("ETH_rv")
    spot_btc = fetch_spot_history("BTC", days)["close"].rename("BTC_close")
    spot_eth = fetch_spot_history("ETH", days)["close"].rename("ETH_close")
    merged = dvol.join([rv_btc, rv_eth, spot_btc, spot_eth], how="left")
    merged["rv_spread"] = merged["ETH_rv"] - merged["BTC_rv"]
    merged["rv_ratio"]  = merged["ETH_rv"] / merged["BTC_rv"]
    # rolling 30D correlation of log-returns
    lr_btc = np.log(merged["BTC_close"] / merged["BTC_close"].shift(1))
    lr_eth = np.log(merged["ETH_close"] / merged["ETH_close"].shift(1))
    merged["corr30"] = lr_btc.rolling(30).corr(lr_eth)
    return merged.dropna(subset=["BTC_dvol", "ETH_dvol"])

if refresh:
    st.cache_data.clear()

with st.spinner("Loading DVOL history from Deribit…"):
    try:
        df = _load(days, rv_win)
    except Exception as exc:
        st.error(f"Data error: {exc}")
        st.stop()


# ═══════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════

def run_backtest(data: pd.DataFrame, window: int, entry_z: float,
                 exit_z: float, max_hold: int) -> tuple[pd.Series, pd.DataFrame]:
    """
    Backtest the ETH/BTC vol spread strategy.

    Entry : z-score of spread > entry_z   (spread is rich → sell ETH, buy BTC vol)
    Exit  : z-score < exit_z  OR  days_held >= max_hold

    Returns
    -------
    daily_pnl : pd.Series  (vol-points per day, multiply by vega_notional for $)
    trades    : pd.DataFrame
    """
    spread     = data["spread"]
    roll_mean  = spread.rolling(window).mean()
    roll_std   = spread.rolling(window).std()
    zscore     = (spread - roll_mean) / roll_std

    position   = 0
    entry_idx  = None
    entry_sprd = None

    daily_pnl  = pd.Series(0.0, index=data.index)
    trades     = []

    eth_dvol = data["ETH_dvol"]
    btc_dvol = data["BTC_dvol"]

    for i in range(window, len(data)):
        z = zscore.iloc[i]
        if np.isnan(z):
            continue

        if position == 0:
            if z >= entry_z:
                position   = 1
                entry_idx  = i
                entry_sprd = spread.iloc[i]

        else:
            # Daily MTM: short ETH, long BTC
            d_eth = eth_dvol.iloc[i] - eth_dvol.iloc[i - 1]
            d_btc = btc_dvol.iloc[i] - btc_dvol.iloc[i - 1]
            daily_pnl.iloc[i] = -d_eth + d_btc

            days_held   = i - entry_idx
            exit_reason = None
            if z <= exit_z:
                exit_reason = "mean-revert"
            elif days_held >= max_hold:
                exit_reason = "max hold"

            if exit_reason:
                exit_sprd = spread.iloc[i]
                trades.append({
                    "Entry Date":   data.index[entry_idx].date(),
                    "Exit Date":    data.index[i].date(),
                    "Days Held":    days_held,
                    "Entry Spread": round(entry_sprd, 2),
                    "Exit Spread":  round(exit_sprd, 2),
                    "P&L (vol-pts)": round(entry_sprd - exit_sprd, 2),
                    "Exit Reason":  exit_reason,
                })
                position  = 0
                entry_idx = None

    return daily_pnl, pd.DataFrame(trades)


daily_pnl, trades = run_backtest(df, z_window, entry_z, exit_z, max_hold)

# Z-score series for display
roll_mean = df["spread"].rolling(z_window).mean()
roll_std  = df["spread"].rolling(z_window).std()
zscore    = (df["spread"] - roll_mean) / roll_std

cum_pnl        = (daily_pnl * vega_notl).cumsum()
total_pnl      = cum_pnl.iloc[-1]
n_trades       = len(trades)
win_rate       = (trades["P&L (vol-pts)"] > 0).mean() * 100 if n_trades else 0
avg_pnl        = trades["P&L (vol-pts)"].mean() if n_trades else 0
sharpe         = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)) if daily_pnl.std() > 0 else 0
roll_dd        = cum_pnl - cum_pnl.cummax()
max_dd         = roll_dd.min()
current_ratio  = df["ratio"].iloc[-1]
current_z      = zscore.iloc[-1]
current_pctile = (df["ratio"] <= current_ratio).mean() * 100


# ═══════════════════════════════════════════════════════════════════════
# TOP METRICS
# ═══════════════════════════════════════════════════════════════════════

st.caption(f"Data: {df.index[0].date()} → {df.index[-1].date()}  ·  {len(df)} trading days")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("BTC DVOL",          f"{df['BTC_dvol'].iloc[-1]:.1f}%")
c2.metric("ETH DVOL",          f"{df['ETH_dvol'].iloc[-1]:.1f}%")
c3.metric("IV Spread (ETH−BTC)", f"{df['spread'].iloc[-1]:.1f} vpts",
          delta=f"{current_pctile:.0f}th pct")
c4.metric("Ratio (ETH/BTC)",   f"{current_ratio:.3f}",
          delta=f"z = {current_z:.2f}")
c5.metric("Cum P&L",           f"${total_pnl:,.0f}",
          delta=f"{n_trades} trades")
c6.metric("Sharpe",            f"{sharpe:.2f}",
          delta=f"Win rate {win_rate:.0f}%")

signal_active = current_z >= entry_z
if signal_active:
    st.warning(
        f"**Signal active:** z-score = {current_z:.2f} ≥ {entry_z}  →  "
        f"Consider selling ETH vol / buying BTC vol.  "
        f"Spread at {current_pctile:.0f}th percentile over {days} days."
    )
else:
    st.info(f"No signal. z-score = {current_z:.2f}  (threshold = {entry_z})")

st.divider()

# ═══════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════

tab_ratio, tab_bt, tab_risk = st.tabs([
    "Ratio Tracker", "Backtest", "Risk Monitor"
])


# ── Tab 1: Ratio Tracker ──────────────────────────────────────────────
with tab_ratio:

    # ── DVOL levels ────────────────────────────────────────────────
    st.subheader("BTC vs ETH Implied Vol (DVOL)")
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.45, 0.3, 0.25],
        vertical_spacing=0.04,
        subplot_titles=("DVOL Levels (%)", "IV Spread (ETH − BTC, vol-pts)", "IV Ratio (ETH / BTC)"),
    )

    fig.add_trace(go.Scatter(x=df.index, y=df["BTC_dvol"], name="BTC DVOL",
                             line=dict(color="#00b4d8", width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ETH_dvol"], name="ETH DVOL",
                             line=dict(color="#f4a261", width=1.8)), row=1, col=1)
    if "BTC_rv" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BTC_rv"], name=f"BTC RV{rv_win}D",
                                 line=dict(color="#00b4d8", dash="dot", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["ETH_rv"], name=f"ETH RV{rv_win}D",
                                 line=dict(color="#f4a261", dash="dot", width=1)), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["spread"], name="IV Spread",
                             line=dict(color="#e63946", width=1.8),
                             fill="tozeroy", fillcolor="rgba(230,57,70,0.08)"), row=2, col=1)
    if "rv_spread" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["rv_spread"], name=f"RV Spread",
                                 line=dict(color="#e63946", dash="dot", width=1)), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["ratio"], name="IV Ratio",
                             line=dict(color="#2a9d8f", width=1.8)), row=3, col=1)
    # Historical mean ± 1 std band on ratio
    r_mean = df["ratio"].mean()
    r_std  = df["ratio"].std()
    fig.add_hline(y=r_mean,          line_dash="dash", line_color="gray",   row=3, col=1)
    fig.add_hline(y=r_mean + r_std,  line_dash="dot",  line_color="#e63946", row=3, col=1)
    fig.add_hline(y=r_mean - r_std,  line_dash="dot",  line_color="#2a9d8f", row=3, col=1)

    fig.update_layout(height=620, margin=dict(t=40, b=20), hovermode="x unified",
                      legend=dict(orientation="h", y=1.04))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Ratio mean (2Y)",     f"{r_mean:.3f}")
    col2.metric("Ratio +1σ",           f"{r_mean + r_std:.3f}")
    col3.metric("Current vs mean",     f"{(current_ratio - r_mean)/r_std:+.2f}σ")

    st.caption(
        "**IV Spread**: structural ETH vol premium in vol-points.  "
        "**Dashed lines**: rolling realized vol — when IV spread >> RV spread, the premium is harvestable.  "
        "**Ratio band**: ±1σ historical range; current ratio above +1σ is the entry zone."
    )


# ── Tab 2: Backtest ───────────────────────────────────────────────────
with tab_bt:
    st.subheader("Backtest: Sell ETH Vol / Buy BTC Vol")
    st.caption(
        f"Entry when z-score ≥ **{entry_z}** · Exit when z ≤ **{exit_z}** or after **{max_hold}** days  "
        f"· Z-score window: **{z_window}D** · {n_trades} trades"
    )

    # ── Cumulative P&L + drawdown ─────────────────────────────────
    fig2 = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.45, 0.3, 0.25],
        vertical_spacing=0.04,
        subplot_titles=("Cumulative P&L ($)", "Drawdown ($)", "Z-Score + Signals"),
    )

    # shade in-trade periods
    in_trade = (daily_pnl != 0)
    trade_starts = df.index[in_trade & ~in_trade.shift(1, fill_value=False)]
    trade_ends   = df.index[in_trade & ~in_trade.shift(-1, fill_value=False)]
    for ts, te in zip(trade_starts, trade_ends):
        for row in [1, 2, 3]:
            fig2.add_vrect(x0=ts, x1=te, fillcolor="#2a9d8f",
                           opacity=0.08, line_width=0, row=row, col=1)

    fig2.add_trace(go.Scatter(x=cum_pnl.index, y=cum_pnl,
                              name="Cum P&L", line=dict(color="#2a9d8f", width=2),
                              fill="tozeroy", fillcolor="rgba(42,157,143,0.1)"),
                   row=1, col=1)
    fig2.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

    fig2.add_trace(go.Scatter(x=roll_dd.index, y=roll_dd * vega_notl,
                              name="Drawdown", line=dict(color="#e63946", width=1.5),
                              fill="tozeroy", fillcolor="rgba(230,57,70,0.1)"),
                   row=2, col=1)

    fig2.add_trace(go.Scatter(x=zscore.index, y=zscore, name="Z-score",
                              line=dict(color="#00b4d8", width=1.5)), row=3, col=1)
    fig2.add_hline(y=entry_z, line_dash="dash", line_color="#e63946", row=3, col=1,
                   annotation_text=f"Entry {entry_z}", annotation_position="right")
    fig2.add_hline(y=exit_z,  line_dash="dot",  line_color="#2a9d8f", row=3, col=1,
                   annotation_text=f"Exit {exit_z}", annotation_position="right")

    fig2.update_layout(height=640, margin=dict(t=40, b=20), hovermode="x unified",
                       legend=dict(orientation="h", y=1.04))
    st.plotly_chart(fig2, use_container_width=True)

    # ── Summary stats ─────────────────────────────────────────────
    if n_trades:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total P&L",     f"${total_pnl:,.0f}")
        c2.metric("Sharpe ratio",  f"{sharpe:.2f}")
        c3.metric("Max drawdown",  f"${max_dd*vega_notl:,.0f}")
        c4.metric("Avg P&L / trade", f"{avg_pnl:.2f} vpts  (${avg_pnl*vega_notl:,.0f})")

        st.subheader("Trade Log")
        trades_disp = trades.copy()
        trades_disp["P&L ($)"] = (trades_disp["P&L (vol-pts)"] * vega_notl).round(0).astype(int)
        trades_disp["Win"] = trades_disp["P&L (vol-pts)"] > 0

        def _color_trades(row):
            color = "background-color: #d4edda" if row["Win"] else "background-color: #f8d7da"
            return [color] * len(row)

        st.dataframe(
            trades_disp.style.apply(_color_trades, axis=1),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No trades generated with current parameters.")

    st.caption(
        "**P&L model:** daily MTM = −ΔETH_DVOL + ΔBTC_DVOL (vega-neutral vol spread).  "
        "This is a simplified model — real P&L also includes theta collected, "
        "gamma realization from delta hedging, and transaction costs.  "
        "Green shading = periods in trade."
    )


# ── Tab 3: Risk Monitor ────────────────────────────────────────────────
with tab_risk:
    st.subheader("Risk Monitor")

    # ── Rolling correlation ───────────────────────────────────────
    st.markdown("#### BTC / ETH Rolling 30D Correlation")
    st.caption(
        "When correlation is high (> 0.7), the hedge works well — ETH and BTC move together "
        "so selling ETH vol and buying BTC vol is well-balanced.  "
        "When correlation drops, the two legs decouple and position risk increases."
    )

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df.index, y=df["corr30"], name="30D Corr",
        line=dict(color="#f4a261", width=2),
        fill="tozeroy", fillcolor="rgba(244,162,97,0.1)",
    ))
    fig3.add_hline(y=0.7, line_dash="dash", line_color="#2a9d8f",
                   annotation_text="0.7 — reduce size below", annotation_position="right")
    fig3.add_hline(y=0.5, line_dash="dash", line_color="#e63946",
                   annotation_text="0.5 — stop trading", annotation_position="right")
    fig3.update_layout(yaxis=dict(range=[0, 1.05]), height=280,
                       margin=dict(t=20, b=30), hovermode="x unified")
    st.plotly_chart(fig3, use_container_width=True)

    current_corr = df["corr30"].iloc[-1]
    corr_color   = "normal" if current_corr >= 0.7 else "inverse"
    st.metric("Current 30D Correlation", f"{current_corr:.3f}",
              delta="hedge ok" if current_corr >= 0.7 else "⚠️ hedge degraded",
              delta_color=corr_color)

    st.divider()

    # ── IV premium vs RV premium ──────────────────────────────────
    st.markdown("#### Implied Premium vs Realized Premium")
    st.caption(
        "The IV spread (blue) should exceed the RV spread (orange) on average — "
        "that gap is the **harvestable vol risk premium**.  "
        "When they converge, the edge is low."
    )

    if "rv_spread" in df.columns:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=df.index, y=df["spread"].rolling(5).mean(),
                                  name="IV Spread (5D MA)", line=dict(color="#00b4d8", width=2)))
        fig4.add_trace(go.Scatter(x=df.index, y=df["rv_spread"].rolling(5).mean(),
                                  name=f"RV Spread (5D MA)", line=dict(color="#f4a261", width=2)))
        iv_premium = (df["spread"] - df["rv_spread"]).rolling(30).mean()
        fig4.add_trace(go.Scatter(x=df.index, y=iv_premium,
                                  name="IV − RV spread (30D MA)",
                                  line=dict(color="#2a9d8f", width=1.5, dash="dot"),
                                  fill="tozeroy", fillcolor="rgba(42,157,143,0.07)"))
        fig4.add_hline(y=0, line_dash="dash", line_color="gray")
        fig4.update_layout(height=300, margin=dict(t=20, b=30),
                            yaxis_title="Vol-points", hovermode="x unified")
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    # ── Position sizing guide ─────────────────────────────────────
    st.markdown("#### Position Sizing Guide")
    st.caption(
        "Size the position so that a 1σ adverse move in the spread "
        "results in a tolerable drawdown. "
        "Reduce notional when correlation is low."
    )

    corr_scalar = max(0.0, min(1.0, (current_corr - 0.5) / 0.5))  # 0 at ρ=0.5, 1 at ρ=1.0
    spread_std  = df["spread"].rolling(z_window).std().iloc[-1]
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Spread 1σ (60D)", f"{spread_std:.1f} vpts")
    col_b.metric("Correlation scalar", f"{corr_scalar:.2f}",
                 help="Reduces size when hedge is weaker")
    col_c.metric("Suggested notional ($k / vpt)",
                 f"{vega_notl * corr_scalar:.1f}",
                 delta=f"vs {vega_notl} input")

    st.info(
        "**Risk rules of thumb:**\n"
        "- Stop if 30D correlation < 0.5\n"
        "- Reduce 50% if spread z-score moves further against you after entry\n"
        "- Rebalance delta daily using BTC/ETH perpetual futures\n"
        "- Max vega exposure: 2× normal size during high-conviction entries"
    )
