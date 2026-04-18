"""
Live Vol Monitor — Crypto Option Term Structure & Calendar Arb Scanner.

Data: Deribit public API (no key required).
Refresh: manual button or auto every N seconds.
"""

from __future__ import annotations

import time
from datetime import datetime

import plotly.graph_objects as go
import streamlit as st

from market_data.providers.deribit import DeribitProvider
from market_data.curves.vol_term_structure import build_term_structure

st.set_page_config(page_title="Live Vol Monitor", layout="wide")
st.title("Live Vol Monitor")

# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("Settings")
    currency    = st.selectbox("Underlying", ["BTC", "ETH", "SOL"])
    rate        = st.number_input(
        "Funding rate", value=0.05, step=0.01, format="%.2f",
        help="Crypto lending/funding rate used to compute the forward price.",
    )
    auto_label  = st.selectbox("Auto-refresh", ["Off", "30s", "1min", "5min"])
    refresh_btn = st.button("↺  Refresh now", use_container_width=True)

    st.divider()
    st.caption("Data: Deribit public API · No key required")


# ═══════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════

_TTL = {"Off": 3600, "30s": 30, "1min": 60, "5min": 300}


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch(currency: str, auto_label: str) -> tuple:
    # auto_label is part of the cache key: changing the refresh rate forces
    # a re-fetch immediately rather than waiting for the old TTL to expire.
    provider = DeribitProvider()
    spot     = provider.get_forward(currency)
    quotes   = provider.get_option_chain(currency)
    fetched  = datetime.utcnow().strftime("%H:%M:%S UTC")
    return spot, quotes, fetched


if refresh_btn:
    _fetch.clear()

with st.spinner(f"Fetching {currency} option chain from Deribit…"):
    try:
        spot, quotes, fetched_at = _fetch(currency, auto_label)
    except Exception as exc:
        st.error(f"Deribit API error: {exc}")
        st.stop()

ts = build_term_structure(quotes, spot=spot, rate=rate)

if ts.empty:
    st.warning("Not enough data to build term structure.")
    st.stop()

# Reset to integer index and pre-compute prev_days for forward vol midpoint
ts = ts.reset_index(drop=True)
ts["prev_days"] = ts["days"].shift(1)

# Pre-compute filtered views used across multiple tabs
fwd_valid = ts[ts["fwd_vol"].notna() & ~ts["is_calendar_arb"]]
fwd_arb   = ts[ts["is_calendar_arb"]]
clean_idx = ts[~ts["is_calendar_arb"]].index
arb_idx   = ts[ts["is_calendar_arb"]].index

n_arb = len(arb_idx)
slope = (ts.iloc[-1]["atm_iv"] - ts.iloc[0]["atm_iv"]) * 100

# ═══════════════════════════════════════════════════════════════════════
# TOP METRICS
# ═══════════════════════════════════════════════════════════════════════

st.caption(f"Last updated: **{fetched_at}**  ·  {len(quotes):,} option quotes  ·  {len(ts)} expiries")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(f"{currency} Spot",     f"${spot:,.0f}")
c2.metric("Front ATM IV",         f"{ts.iloc[0]['atm_iv']*100:.1f}%",
          delta=f"{ts.iloc[0]['tenor_label']}")
c3.metric("Back ATM IV",          f"{ts.iloc[-1]['atm_iv']*100:.1f}%",
          delta=f"{ts.iloc[-1]['tenor_label']}")
c4.metric("Term Structure Slope", f"{slope:+.1f}%",
          delta="contango" if slope > 0 else "backwardation",
          delta_color="normal" if slope > 0 else "inverse")
c5.metric("Calendar Arb Alerts",  str(n_arb),
          delta="⚠️ check scanner" if n_arb > 0 else "✓ clean",
          delta_color="inverse" if n_arb > 0 else "off")

st.divider()

# ═══════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════

tab_ts, tab_fwd, tab_arb, tab_tvar = st.tabs([
    "Vol Term Structure",
    "Forward Vol Strip",
    "Calendar Arb Scanner",
    "Total Variance",
])


# ── Tab 1: Vol Term Structure ─────────────────────────────────────────
with tab_ts:
    st.subheader("ATM Implied Vol — Spot vs Forward")
    st.caption(
        "**Spot vol** σ(T): market's average vol from today to T.  "
        "**Forward vol** σ_fwd(T₁→T₂): implied vol *between* two future dates.  "
        "When forward vol < spot vol → term structure is in **backwardation** (vol spike expected to mean-revert)."
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ts["days"],
        y=ts["atm_iv"] * 100,
        mode="lines+markers",
        name="Spot ATM IV",
        line=dict(color="#00b4d8", width=2.5),
        marker=dict(size=8),
        hovertemplate="%{text}<br>ATM IV: %{y:.2f}%<extra></extra>",
        text=ts["tenor_label"],
    ))

    if not fwd_valid.empty:
        mid_days = (fwd_valid["days"] + fwd_valid["prev_days"]) / 2
        fig.add_trace(go.Scatter(
            x=mid_days,
            y=fwd_valid["fwd_vol"] * 100,
            mode="markers",
            name="Forward Vol",
            marker=dict(symbol="diamond", size=10, color="#f4a261"),
            hovertemplate="%{text}<br>Fwd Vol: %{y:.2f}%<extra></extra>",
            text=fwd_valid["fwd_label"],
        ))

    if not fwd_arb.empty:
        fig.add_trace(go.Scatter(
            x=fwd_arb["days"],
            y=fwd_arb["atm_iv"] * 100,
            mode="markers",
            name="Calendar Arb ⚠️",
            marker=dict(symbol="x", size=14, color="#e63946", line=dict(width=2)),
            hovertemplate="%{text}<br>⚠️ Calendar arb<extra></extra>",
            text=fwd_arb["tenor_label"],
        ))

    fig.update_layout(
        xaxis=dict(title="Tenor", type="log",
                   tickvals=ts["days"].tolist(), ticktext=ts["tenor_label"].tolist()),
        yaxis_title="Implied Vol (%)",
        legend=dict(orientation="h", y=1.08),
        height=420, margin=dict(t=20, b=40), hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 2: Forward Vol Strip ──────────────────────────────────────────
with tab_fwd:
    st.subheader("Forward Vol Strip")
    st.caption(
        "Each bar shows σ_fwd(T₁→T₂) — the vol implied for the **future period** between two consecutive expiries.  "
        "A low bar next to a high spot vol means the market expects vol to collapse — a potential **calendar spread** opportunity.  "
        "Red bars = calendar arbitrage (negative forward variance), excluded from the bar height."
    )

    if fwd_valid.empty:
        st.info("Not enough expiries to compute forward vols.")
    else:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=fwd_valid["fwd_label"],
            y=fwd_valid["fwd_vol"] * 100,
            marker_color="#2a9d8f",
            name="Forward Vol",
            hovertemplate="%{x}<br>Fwd Vol: %{y:.2f}%<extra></extra>",
        ))
        if not fwd_arb.empty:
            fig2.add_trace(go.Bar(
                x=fwd_arb["fwd_label"],
                y=[-1.0] * len(fwd_arb),
                marker_color="#e63946",
                name="⚠️ Calendar Arb",
                hovertemplate="%{x}<br>⚠️ Negative forward variance<extra></extra>",
            ))
        fig2.add_trace(go.Scatter(
            x=fwd_valid["fwd_label"],
            y=fwd_valid["atm_iv"] * 100,
            mode="markers+lines",
            name="Spot ATM IV (end of period)",
            line=dict(color="#00b4d8", dash="dot", width=1.5),
            marker=dict(size=6),
        ))
        fig2.update_layout(
            xaxis_title="Forward Period", yaxis_title="Vol (%)",
            barmode="group",
            legend=dict(orientation="h", y=1.08),
            height=400, margin=dict(t=20, b=60),
        )
        fig2.update_xaxes(tickangle=-35)
        st.plotly_chart(fig2, use_container_width=True)


# ── Tab 3: Calendar Arb Scanner ──────────────────────────────────────
with tab_arb:
    st.subheader("Calendar Arbitrage Scanner")
    st.caption(
        "No-arb requires **w(T) = σ(T)² × T strictly increasing**.  "
        "A dip means the forward variance is negative — the spread between those two expiries "
        "can be sold for a risk-free profit (ignoring bid-ask and execution risk)."
    )

    display = ts[[
        "tenor_label", "days", "atm_iv", "total_var",
        "fwd_label", "fwd_vol", "is_calendar_arb",
    ]].copy()
    display["atm_iv"]    = (display["atm_iv"]  * 100).round(2)
    display["fwd_vol"]   = (display["fwd_vol"] * 100).round(2)
    display["total_var"] = display["total_var"].round(6)
    display["days"]      = display["days"].round(1)
    display.rename(columns={
        "tenor_label":     "Tenor",
        "days":            "Days",
        "atm_iv":          "ATM IV (%)",
        "total_var":       "Total Var w(T)",
        "fwd_label":       "Fwd Period",
        "fwd_vol":         "Fwd Vol (%)",
        "is_calendar_arb": "⚠️ Arb?",
    }, inplace=True)

    def _color_arb(row):
        if row["⚠️ Arb?"]:
            return ["background-color: #ffcccc"] * len(row)
        return [""] * len(row)

    styled = display.style.apply(_color_arb, axis=1).format(
        {"ATM IV (%)": "{:.2f}", "Fwd Vol (%)": "{:.2f}", "Total Var w(T)": "{:.6f}"}
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    if n_arb > 0:
        st.error(
            f"**{n_arb} calendar arbitrage violation{'s' if n_arb > 1 else ''} detected.**  "
            "These are the expiry pairs where total variance is non-monotone. "
            "In practice: check bid-ask spread before trading — the violation may be within transaction costs."
        )
    else:
        st.success("No calendar arbitrage violations detected. Total variance is monotone increasing.")


# ── Tab 4: Total Variance ─────────────────────────────────────────────
with tab_tvar:
    st.subheader("Total Variance w(T) = σ(T)² × T")
    st.caption(
        "This curve **must be non-decreasing** for a no-arbitrage vol surface. "
        "Any local dip creates a calendar spread arbitrage. "
        "The slope of this curve between two points equals the **forward variance** for that period."
    )

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=ts.loc[clean_idx, "days"],
        y=ts.loc[clean_idx, "total_var"],
        mode="lines+markers",
        name="w(T)  — no arb",
        line=dict(color="#2a9d8f", width=2.5),
        marker=dict(size=7),
        hovertemplate="%{text}<br>w(T) = %{y:.5f}<extra></extra>",
        text=ts.loc[clean_idx, "tenor_label"],
    ))

    if not arb_idx.empty:
        fig4.add_trace(go.Scatter(
            x=ts.loc[arb_idx, "days"],
            y=ts.loc[arb_idx, "total_var"],
            mode="markers",
            name="w(T)  — ⚠️ arb",
            marker=dict(symbol="x", size=14, color="#e63946", line=dict(width=2)),
            hovertemplate="%{text}<br>w(T) = %{y:.5f}  ⚠️<extra></extra>",
            text=ts.loc[arb_idx, "tenor_label"],
        ))

    fig4.update_layout(
        xaxis=dict(title="Tenor", type="log",
                   tickvals=ts["days"].tolist(), ticktext=ts["tenor_label"].tolist()),
        yaxis_title="w(T) = σ² × T",
        height=400, margin=dict(t=20, b=40), hovermode="x unified",
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.caption(
        "**How to read this:** The slope between two points = forward variance.  "
        "A steep slope = high forward vol. A flat slope = low forward vol.  "
        "A **negative slope** = calendar arbitrage."
    )


# ═══════════════════════════════════════════════════════════════════════
# AUTO REFRESH  (time.sleep is unavoidable in synchronous Streamlit)
# ═══════════════════════════════════════════════════════════════════════

if auto_label != "Off":
    time.sleep(_TTL[auto_label])
    st.rerun()
