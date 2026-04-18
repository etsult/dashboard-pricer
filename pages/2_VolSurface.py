"""
Volatility Surface Viewer.

Data    : Equity via Yahoo Finance · Crypto via Deribit (no key required)
Models  : SVI (per-expiry, fast) · SSVI (global, smooth) · Heston (full model, slow)
Display : 3-D surface · per-expiry smile slices · ATM term structure · model params
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st

from market_data.router import MarketRouter
from surface.builders import build_vol_surface
from surface.visualization import plot_surface_3d, plot_smile_slices, plot_term_structure

st.set_page_config(page_title="Vol Surface", layout="wide")
st.title("Volatility Surface")

# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("Data Source")
    asset_class = st.radio("Asset class", ["Equity (Yahoo)", "Crypto (Deribit)"], horizontal=True)

    if asset_class == "Equity (Yahoo)":
        provider_key = "yahoo"
        ticker = st.text_input("Ticker", "SPY")
        rate   = st.number_input("Risk-free rate", value=0.042, step=0.005, format="%.3f")
        div    = st.number_input("Dividend yield", value=0.013, step=0.005, format="%.3f")
    else:
        provider_key = "deribit"
        ticker = st.selectbox("Currency", ["BTC", "ETH", "SOL"])
        rate   = st.number_input("Funding / borrow rate", value=0.05, step=0.005, format="%.3f")
        div    = 0.0

    st.divider()
    st.header("Surface Model")

    model = st.selectbox(
        "Fitting model",
        ["SVI", "SSVI", "Heston"],
        help=(
            "**SVI** — Stochastic Volatility Inspired (Gatheral 2004). "
            "Fits per expiry independently, ~1–2s. Best general choice.\n\n"
            "**SSVI** — Surface SVI (Gatheral & Jacquier 2014). "
            "Global 3-parameter fit, inherently no-arb across T, ~2–4s.\n\n"
            "**Heston** — Full stochastic vol model. "
            "Global 5-parameter fit via Lewis formula, ~15–40s."
        ),
    )

    if model == "Heston":
        st.warning("Heston calibration may take 15–40 seconds.", icon="⏳")

    n_slices = st.slider("Smile slices to show", 3, 9, 6)

    st.divider()
    load_btn = st.button("🔄  Load & Fit Surface", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# DATA FETCHING  (cached per ticker + provider)
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def _fetch(ticker: str, provider_key: str) -> tuple:
    router = MarketRouter(provider_key)
    return router.get_option_chain(ticker), router.get_forward(ticker)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if load_btn:
    with st.spinner(f"Fetching {ticker} option chain…"):
        try:
            quotes, spot = _fetch(ticker, provider_key)
        except Exception as exc:
            st.error(f"Data fetch error: {exc}")
            st.stop()

    st.caption(f"Spot: **{spot:,.2f}** · {len(quotes):,} raw quotes fetched")

    with st.spinner(f"Fitting {model} surface…"):
        t0 = time.perf_counter()
        surf_df, params_df, raw_df = build_vol_surface(
            quotes, spot=spot, rate=rate, dividend_yield=div, model=model
        )
        elapsed = time.perf_counter() - t0

    if surf_df.empty:
        st.error(
            "Surface is empty after cleaning — not enough liquid options. "
            "Try a more liquid ticker (SPY, QQQ, BTC…) or check the data source."
        )
        st.stop()

    n_pts   = len(raw_df)
    n_exp   = raw_df["T"].nunique() if not raw_df.empty else 0
    k_range = (raw_df["k"].min(), raw_df["k"].max()) if not raw_df.empty else (0, 0)

    st.success(
        f"**{model}** surface fitted in **{elapsed:.1f}s** · "
        f"{n_pts:,} clean quotes · {n_exp} expiries · "
        f"moneyness [{k_range[0]:.2f}, {k_range[1]:.2f}]"
    )

    # Store in session so tabs stay visible after sidebar interaction
    st.session_state["surf_df"]   = surf_df
    st.session_state["params_df"] = params_df
    st.session_state["raw_df"]    = raw_df
    st.session_state["model"]     = model
    st.session_state["ticker"]    = ticker
    st.session_state["spot"]      = spot


# ── Show tabs only when a surface is loaded ───────────────────────────
if "surf_df" not in st.session_state or st.session_state["surf_df"].empty:
    st.info("Configure the data source and model in the sidebar, then click **Load & Fit Surface**.")
    st.stop()

surf_df   = st.session_state["surf_df"]
params_df = st.session_state["params_df"]
raw_df    = st.session_state["raw_df"]
fitted_model = st.session_state.get("model", model)
fitted_ticker = st.session_state.get("ticker", ticker)


tab_3d, tab_slices, tab_ts, tab_params = st.tabs([
    "🌐 3D Surface",
    "📉 Smile Slices",
    "📈 ATM Term Structure",
    "🔢 Model Parameters",
])


# ── 3D surface ────────────────────────────────────────────────────────
with tab_3d:
    st.plotly_chart(
        plot_surface_3d(surf_df, title=f"{fitted_ticker} — {fitted_model} Implied Vol Surface"),
        use_container_width=True,
    )
    st.caption(
        "Surface axes: **log-moneyness** ln(K/F) on X (negative = OTM puts, positive = OTM calls), "
        "**days to expiry** on Y, **implied vol (%)** on Z. "
        "Contour lines projected on the floor show the vol smile shape by expiry."
    )


# ── Per-expiry smile slices ───────────────────────────────────────────
with tab_slices:
    st.subheader("Per-Expiry Smile — Model vs Market")
    st.caption(
        "**Lines** = fitted model. **Dots** = market implied vols (OTM calls + puts, liquid quotes only). "
        "A good fit has dots sitting on the line."
    )
    fig_sl = plot_smile_slices(surf_df, raw_df, n_slices=n_slices)
    st.plotly_chart(fig_sl, use_container_width=True)

    # Allow user to inspect a specific expiry in detail
    T_opts = sorted(raw_df["T"].unique()) if not raw_df.empty else []
    if T_opts:
        st.divider()
        st.subheader("Zoom into one expiry")
        T_sel = st.selectbox(
            "Select expiry",
            T_opts,
            format_func=lambda t: f"{t*365:.0f}d ({t:.3f}Y)",
        )
        mkt_sel = raw_df[np.abs(raw_df["T"] - T_sel) < 0.01]
        model_k = surf_df.index.values.astype(float)

        # Nearest column in surf_df
        T_cols = surf_df.columns.values.astype(float)
        nearest_col = T_cols[np.argmin(np.abs(T_cols - T_sel))]

        import plotly.graph_objects as go
        fig_z = go.Figure()
        fig_z.add_trace(go.Scatter(
            x=model_k, y=surf_df[nearest_col].values * 100,
            mode="lines", line=dict(color="#00b4d8", width=2.5), name="Model",
        ))
        if not mkt_sel.empty:
            # Color by option type
            for otype, color in [("call", "#2a9d8f"), ("put", "#e63946")]:
                sub = mkt_sel[mkt_sel["type"] == otype]
                if not sub.empty:
                    fig_z.add_trace(go.Scatter(
                        x=sub["k"].values, y=sub["iv"].values * 100,
                        mode="markers",
                        marker=dict(color=color, size=8, symbol="circle",
                                    line=dict(color="white", width=0.5)),
                        name=f"Market {otype}",
                    ))
        fig_z.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="ATM")
        fig_z.update_layout(
            xaxis_title="log(K/F)", yaxis_title="IV (%)",
            height=360, margin=dict(t=20, b=40),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_z, use_container_width=True)


# ── ATM term structure ────────────────────────────────────────────────
with tab_ts:
    st.subheader("ATM Implied Vol — Term Structure")
    st.plotly_chart(plot_term_structure(surf_df, raw_df), use_container_width=True)

    # Vol of vol proxy: slope of the smile at each expiry
    if not raw_df.empty:
        st.subheader("Skew Term Structure  (25Δ skew ≈ ∂IV/∂k at k=−0.1)")
        T_vals = surf_df.columns.values.astype(float)
        k_vals = surf_df.index.values.astype(float)

        skews = []
        for T in T_vals:
            iv_col = surf_df[T].values
            k_p25  = -0.10   # ~25-delta put proxy in log-moneyness
            k_c25  =  0.10
            iv_p   = float(np.interp(k_p25, k_vals, iv_col))
            iv_c   = float(np.interp(k_c25, k_vals, iv_col))
            skews.append((T * 365, (iv_p - iv_c) * 100))  # put IV > call IV → positive skew

        skew_df = pd.DataFrame(skews, columns=["Days", "25Δ Skew (pp)"])

        import plotly.graph_objects as go
        fig_sk = go.Figure(go.Scatter(
            x=skew_df["Days"], y=skew_df["25Δ Skew (pp)"],
            mode="lines+markers",
            line=dict(color="#f4a261", width=2),
            marker=dict(size=5),
        ))
        fig_sk.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_sk.update_layout(
            xaxis_title="Days to Expiry",
            yaxis_title="OTM put IV − OTM call IV (pp)",
            height=300, margin=dict(t=10, b=40),
        )
        st.plotly_chart(fig_sk, use_container_width=True)


# ── Model params ──────────────────────────────────────────────────────
with tab_params:
    st.subheader(f"{fitted_model} Calibrated Parameters")

    if params_df is not None and not params_df.empty:
        st.dataframe(params_df.round(6), use_container_width=True, hide_index=True)

        if fitted_model == "SVI":
            st.caption(
                "**a** = minimum total variance level · "
                "**b** = slope (wing steepness) · "
                "**ρ** = skew · "
                "**m** = ATM shift · "
                "**σ** = smile curvature (convexity). "
                "Fitted independently per expiry."
            )
        elif fitted_model == "SSVI":
            st.caption(
                "**ρ** = global skew parameter · "
                "**η** = power-law scale · "
                "**γ** = power-law exponent. "
                "Single surface calibration across all expiries."
            )
        elif fitted_model == "Heston":
            st.caption(
                "**v0** = initial variance · "
                "**κ** = mean-reversion speed · "
                "**θ** = long-run variance · "
                "**σ** = vol-of-vol · "
                "**ρ** = spot-vol correlation. "
                "Feller condition: 2κθ > σ² (ensures variance stays positive)."
            )
    else:
        st.info("No parameter output available.")

    # Raw market data table
    if not raw_df.empty:
        with st.expander("Clean market data used for calibration"):
            disp = raw_df[["expiry", "type", "strike", "k", "iv", "bid", "ask"]].copy()
            disp["iv"]  = (disp["iv"] * 100).round(2)
            disp["k"]   = disp["k"].round(4)
            disp["bid"] = disp["bid"].round(3)
            disp["ask"] = disp["ask"].round(3)
            disp.columns = ["Expiry", "Type", "Strike", "log(K/F)", "IV (%)", "Bid", "Ask"]
            st.dataframe(disp, use_container_width=True, hide_index=True)
