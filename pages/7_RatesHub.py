# pages/7_RatesHub.py
"""
Rates Hub — Fixed-Income Analytics (à la Hull, Ch. 4–6).

Tabs
────
1. Bond Pricer       : dirty price, YTM, Macaulay / modified duration, convexity, DV01
2. Curve Analysis    : zero vs forward vs par yield; forward rate strip
3. Scenario Analysis : parallel shift, steepener/flattener, butterfly twist
4. FRA & Futures     : FRA fair rate, P&L, SOFR futures convexity adjustment
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.optimize import brentq

from market_data.providers.fred import fetch_usd_curve
from market_data.curves.rate_curve import RateCurve

st.set_page_config(page_title="Rates Hub", layout="wide")
st.title("Rates Hub — Fixed Income Analytics")

# ═══════════════════════════════════════════════════════════════════════
# TENOR LABELS
# ═══════════════════════════════════════════════════════════════════════

_TENOR_LABELS = {
    1/12: "1M", 2/12: "2M", 3/12: "3M", 6/12: "6M",
    1.0: "1Y", 2.0: "2Y", 3.0: "3Y", 5.0: "5Y",
    7.0: "7Y", 10.0: "10Y", 15.0: "15Y", 20.0: "20Y", 30.0: "30Y",
}

def _tenor_label(t: float) -> str:
    return _TENOR_LABELS.get(t, f"{t:g}Y")


# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR — curve source
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("Yield Curve")
    curve_source = st.radio("Source", ["FRED (live USD)", "Manual"])

    if curve_source == "FRED (live USD)":
        env_key = os.environ.get("FRED_API_KEY", "")
        if env_key:
            st.success("FRED key loaded from .env", icon="🔑")
            fred_key = env_key
        else:
            fred_key = st.text_input("FRED API Key", type="password",
                                      help="Get a free key at fred.stlouisfed.org")
    else:
        fred_key = None

    st.divider()
    st.caption("Hull, *Options, Futures, and Other Derivatives* — Ch. 4–6")


# ═══════════════════════════════════════════════════════════════════════
# CURVE BUILDING
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner="Fetching FRED curve…")
def _load_fred(key: str) -> dict:
    return fetch_usd_curve(api_key=key or None)


_DEFAULT_PAR = {
    1/12: 0.0433, 3/12: 0.0427, 6/12: 0.0420,
    1.0:  0.0410, 2.0:  0.0405, 3.0:  0.0408,
    5.0:  0.0420, 7.0:  0.0430, 10.0: 0.0445, 30.0: 0.0470,
}

if curve_source == "FRED (live USD)":
    par_yields = _load_fred(fred_key or "")
else:
    st.sidebar.subheader("Manual Par Yields (%)")
    _rows = [{"Tenor": _tenor_label(t), "Rate (%)": round(r * 100, 3)}
             for t, r in _DEFAULT_PAR.items()]
    _tenor_map = {_tenor_label(t): t for t in _DEFAULT_PAR}
    edited = st.sidebar.data_editor(
        pd.DataFrame(_rows), use_container_width=True, hide_index=True,
    )
    par_yields = {
        _tenor_map[row["Tenor"]]: row["Rate (%)"] / 100.0
        for _, row in edited.iterrows()
        if row["Tenor"] in _tenor_map
    }

curve = RateCurve(par_yields)


# ═══════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════

tab_bond, tab_curves, tab_scenario, tab_fra = st.tabs([
    "Bond Pricer",
    "Curve Analysis",
    "Scenario Analysis",
    "FRA & Futures",
])


# ══════════════════════════════════════════════════════════════════════
# TAB 1  ─  Bond Pricer
# ══════════════════════════════════════════════════════════════════════

with tab_bond:
    st.subheader("Fixed-Rate Bond Pricer")
    st.caption(
        "Prices a bond using the current zero curve. Duration and convexity "
        "use **continuously compounded** yields (Hull's convention)."
    )

    col_left, col_right = st.columns([1, 1.6])

    with col_left:
        st.markdown("#### Bond Parameters")
        face_val    = st.number_input("Face Value ($)", value=1_000_000, step=100_000)
        coupon_pct  = st.number_input("Annual Coupon Rate (%)", value=4.0, step=0.25, format="%.3f")
        maturity_y  = st.number_input("Maturity (years)", value=10.0, min_value=0.5, max_value=50.0, step=0.5)
        freq_lbl    = st.selectbox("Coupon Frequency", ["Semi-annual (2/yr)", "Annual (1/yr)", "Quarterly (4/yr)"])
        freq_map    = {"Semi-annual (2/yr)": 2, "Annual (1/yr)": 1, "Quarterly (4/yr)": 4}
        freq        = freq_map[freq_lbl]
        coupon_rate = coupon_pct / 100.0

        st.markdown("---")
        st.markdown("#### Yield Override *(optional)*")
        use_custom_y = st.checkbox("Price at a specific yield instead of curve")
        if use_custom_y:
            custom_y_pct = st.number_input("YTM to price at (%)", value=round(curve.zero_rate(maturity_y)*100, 3), step=0.05, format="%.4f")
        else:
            custom_y_pct = None

    # ── Compute ─────────────────────────────────────────────────────
    if use_custom_y and custom_y_pct is not None:
        y_cc   = custom_y_pct / 100.0
        period = 1.0 / freq
        coupon = face_val * coupon_rate / freq
        t_grid = np.arange(period, maturity_y + 1e-9, period)
        cfs    = [coupon] * len(t_grid)
        cfs[-1] += face_val
        price = float(sum(cf * np.exp(-y_cc * t) for cf, t in zip(cfs, t_grid)))
        ytm   = y_cc
    else:
        price = curve.price_bond(coupon_rate, maturity_y, face_val, freq)
        ytm   = curve.bond_yield(coupon_rate, maturity_y, face_val, freq)

    d_mac  = curve.macaulay_duration(coupon_rate, maturity_y, face_val, freq)
    d_mod  = d_mac                          # continuous compounding: D_mod = D_mac
    conv   = curve.convexity_measure(coupon_rate, maturity_y, face_val, freq)
    dv01   = curve.dv01(coupon_rate, maturity_y, face_val, freq)
    price_pct = price / face_val * 100.0    # price as % of face

    with col_right:
        st.markdown("#### Results")
        r1c1, r1c2, r1c3 = st.columns(3)
        r1c1.metric("Dirty Price ($)", f"${price:,.0f}")
        r1c2.metric("Price (% of face)", f"{price_pct:.4f}%")
        r1c3.metric("YTM (cont. comp.)", f"{ytm*100:.4f}%")

        r2c1, r2c2, r2c3 = st.columns(3)
        r2c1.metric("Macaulay Duration", f"{d_mac:.4f} yr")
        r2c2.metric("Modified Duration", f"{d_mod:.4f} yr")
        r2c3.metric("Convexity", f"{conv:.4f}")

        r3c1, r3c2 = st.columns(2)
        r3c1.metric("DV01 (per face)", f"${dv01:,.2f}")
        r3c2.metric("BPV01 (total)", f"${dv01:,.2f}")

        st.caption(
            "**DV01**: price change for a 1 bp parallel shift in yield.  "
            "**Convexity**: always positive for a plain bond — it works *in your favour* "
            "for both up and down moves (Jensen's inequality)."
        )

    # ── Cash-flow diagram ────────────────────────────────────────────
    st.markdown("#### Cash Flow Timeline")
    period = 1.0 / freq
    t_grid = np.arange(period, maturity_y + 1e-9, period)
    coupon = face_val * coupon_rate / freq
    cfs    = [coupon] * len(t_grid)
    cfs[-1] += face_val
    pv_cfs = [cf * curve.discount_factor(t) for cf, t in zip(cfs, t_grid)]

    fig_cf = go.Figure()
    fig_cf.add_trace(go.Bar(
        x=np.round(t_grid, 4), y=cfs,
        name="Cash Flow",
        marker_color="#00b4d8",
        opacity=0.6,
        hovertemplate="t=%{x:.2f}y  CF=$%{y:,.0f}<extra></extra>",
    ))
    fig_cf.add_trace(go.Bar(
        x=np.round(t_grid, 4), y=pv_cfs,
        name="PV (zero curve)",
        marker_color="#f4a261",
        hovertemplate="t=%{x:.2f}y  PV=$%{y:,.0f}<extra></extra>",
    ))
    fig_cf.update_layout(
        barmode="overlay", xaxis_title="Time (years)", yaxis_title="$",
        height=300, margin=dict(t=20, b=40),
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig_cf, use_container_width=True)

    # ── Price vs Yield ───────────────────────────────────────────────
    st.markdown("#### Price / Yield Relationship")
    st.caption(
        "The curvature of this line *is* convexity. For the same Δy, "
        "a price fall is smaller than a price rise — always good to be long convexity."
    )
    y_range    = np.linspace(max(ytm - 0.03, 0.001), ytm + 0.03, 200)
    prices_y   = [
        sum(cf * np.exp(-y * t) for cf, t in zip(cfs, t_grid))
        for y in y_range
    ]
    # Duration approximation
    approx_p   = [price * (1 - d_mod * (y - ytm) + 0.5 * conv * (y - ytm)**2)
                  for y in y_range]

    fig_py = go.Figure()
    fig_py.add_trace(go.Scatter(
        x=y_range * 100, y=prices_y,
        mode="lines", name="Exact price",
        line=dict(color="#00b4d8", width=2.5),
    ))
    fig_py.add_trace(go.Scatter(
        x=y_range * 100, y=approx_p,
        mode="lines", name="Duration+Convexity approx",
        line=dict(color="#e63946", dash="dash", width=1.5),
    ))
    fig_py.add_trace(go.Scatter(
        x=[ytm * 100], y=[price],
        mode="markers", name="Current",
        marker=dict(size=12, color="#f4a261", symbol="star"),
    ))
    fig_py.update_layout(
        xaxis_title="YTM (%)", yaxis_title="Price ($)",
        height=340, margin=dict(t=20, b=40),
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig_py, use_container_width=True)

    # ── Taylor approximation explainer ──────────────────────────────
    with st.expander("Price change approximation (Hull, Ch. 4)"):
        st.latex(r"\Delta P \approx -D_{mod} \cdot P \cdot \Delta y + \tfrac{1}{2} \cdot C \cdot P \cdot (\Delta y)^2")
        st.markdown(
            f"- Modified Duration **D_mod = {d_mod:.4f}** yr  \n"
            f"- Convexity **C = {conv:.4f}**  \n"
            f"- Current price **P = ${price:,.2f}**  \n\n"
            "A 1% (100 bp) parallel yield rise would change price by approximately:  \n"
        )
        dp_dur   = -d_mod * price * 0.01
        dp_conv  = 0.5 * conv * price * 0.01**2
        st.markdown(
            f"  Duration term: **${dp_dur:+,.0f}**  \n"
            f"  Convexity term: **${dp_conv:+,.0f}**  \n"
            f"  **Total ≈ ${dp_dur + dp_conv:+,.0f}**  ({(dp_dur + dp_conv)/price*100:+.3f}%)"
        )


# ══════════════════════════════════════════════════════════════════════
# TAB 2  ─  Curve Analysis
# ══════════════════════════════════════════════════════════════════════

with tab_curves:
    st.subheader("Zero Rate · Forward Rate · Par Yield")
    st.caption(
        "**Zero rate** z(T): rate for a single cash flow at T.  "
        "**Instantaneous forward** f(T): marginal rate at T implied by the zero curve.  "
        "**Par yield**: coupon rate for a T-year bond priced at par.  "
        "All rates are **continuously compounded**."
    )

    fwd_df = curve.forward_curve_df()
    t_vals = fwd_df["Tenor"].tolist()
    t_text = [_tenor_label(t) for t in t_vals]

    fig_curves = go.Figure()
    fig_curves.add_trace(go.Scatter(
        x=t_vals, y=fwd_df["Zero Rate (%)"],
        mode="lines+markers", name="Zero Rate",
        line=dict(color="#00b4d8", width=2.5),
        marker=dict(size=7),
        hovertemplate="%{text}: zero=%{y:.3f}%<extra></extra>",
        text=t_text,
    ))
    fig_curves.add_trace(go.Scatter(
        x=t_vals, y=fwd_df["Forward Rate (%)"],
        mode="lines+markers", name="Instantaneous Fwd",
        line=dict(color="#f4a261", width=2, dash="dot"),
        marker=dict(size=6, symbol="diamond"),
        hovertemplate="%{text}: fwd=%{y:.3f}%<extra></extra>",
        text=t_text,
    ))
    fig_curves.add_trace(go.Scatter(
        x=t_vals, y=fwd_df["Par Yield (%)"],
        mode="lines+markers", name="Par Yield",
        line=dict(color="#2a9d8f", width=2, dash="dash"),
        marker=dict(size=6, symbol="square"),
        hovertemplate="%{text}: par=%{y:.3f}%<extra></extra>",
        text=t_text,
    ))
    fig_curves.update_layout(
        xaxis=dict(
            title="Tenor", type="log",
            tickvals=t_vals, ticktext=t_text,
        ),
        yaxis_title="Rate (%, continuously compounded)",
        height=420, margin=dict(t=20, b=40),
        legend=dict(orientation="h", y=1.10),
        hovermode="x unified",
    )
    st.plotly_chart(fig_curves, use_container_width=True)

    st.caption(
        "**Key insight (Hull Ch. 4):** When the zero curve is upward sloping, "
        "the forward curve lies *above* the zero curve. "
        "When flat, all three coincide. The par yield always lies *between* zero and forward."
    )

    # ── Forward Rate Strip (simple, not instantaneous) ───────────────
    st.subheader("6-Month Forward Rate Strip")
    st.caption(
        "Forward rates f(T₁→T₂) for consecutive 6-month periods, "
        "i.e. the rate locked in today for a 6-month deposit starting at T₁."
    )
    strip_ends   = np.arange(0.5, min(curve._tenors[-1], 30.0) + 0.01, 0.5)
    strip_starts = strip_ends - 0.5
    strip_fwds   = []
    for t1, t2 in zip(strip_starts, strip_ends):
        try:
            strip_fwds.append(curve.forward_rate(max(t1, 1e-4), t2) * 100)
        except Exception:
            strip_fwds.append(float("nan"))

    fig_strip = go.Figure(go.Bar(
        x=[f"{t:.1f}Y" for t in strip_ends],
        y=strip_fwds,
        marker_color="#0077b6",
        hovertemplate="Period ending %{x}<br>Fwd: %{y:.3f}%<extra></extra>",
    ))
    fig_strip.update_layout(
        xaxis_title="Period End", yaxis_title="6M Fwd Rate (%)",
        height=320, margin=dict(t=20, b=60),
    )
    fig_strip.update_xaxes(tickangle=-45)
    st.plotly_chart(fig_strip, use_container_width=True)

    # ── Table ────────────────────────────────────────────────────────
    with st.expander("Full curve table"):
        disp = fwd_df.copy()
        disp.insert(0, "Tenor Label", t_text)
        disp = disp.drop(columns=["Tenor"]).round(4)
        st.dataframe(disp, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 3  ─  Scenario Analysis
# ══════════════════════════════════════════════════════════════════════

with tab_scenario:
    st.subheader("Curve Scenario Analysis")
    st.caption(
        "Standard risk scenarios used by rates desks. "
        "**DV01** = dollar change per bp for your bond position."
    )

    col_scen, col_bond = st.columns([1, 1])

    with col_scen:
        st.markdown("#### Scenario Selector")

        scen_type = st.selectbox(
            "Scenario type",
            ["Parallel Shift", "Bull Steepener", "Bear Steepener",
             "Bull Flattener", "Bear Flattener", "Butterfly (Belly Up)",
             "Butterfly (Belly Down)"],
            help="Bull = rates fall, Bear = rates rise.  "
                 "Steepener = long end rises more than short end.",
        )

        shift_bp = st.slider("Magnitude (bp)", -300, 300, 100, 10)

        _scenarios = {
            "Parallel Shift":         lambda c, bp: c.shifted(bp),
            "Bull Steepener":         lambda c, bp: c.steepened(-abs(bp)*0.3, abs(bp)),
            "Bear Steepener":         lambda c, bp: c.steepened(abs(bp)*0.3, abs(bp)),
            "Bull Flattener":         lambda c, bp: c.steepened(-abs(bp), -abs(bp)*0.3),
            "Bear Flattener":         lambda c, bp: c.steepened(abs(bp), abs(bp)*0.3),
            "Butterfly (Belly Up)":   lambda c, bp: c.twisted(abs(bp)),
            "Butterfly (Belly Down)": lambda c, bp: c.twisted(-abs(bp)),
        }

        curve_shocked = _scenarios[scen_type](curve, shift_bp)

    with col_bond:
        st.markdown("#### Bond Position (for DV01 / P&L)")
        pos_face     = st.number_input("Face Value ($)", value=10_000_000, step=1_000_000, key="s_face")
        pos_coupon   = st.number_input("Coupon (%)", value=4.0, step=0.25, format="%.3f", key="s_coup")
        pos_maturity = st.number_input("Maturity (yr)", value=10.0, min_value=0.5, max_value=50.0, step=0.5, key="s_mat")
        pos_freq     = 2
        pos_long     = st.radio("Long / Short", ["Long", "Short"], horizontal=True)
        sign         = 1 if pos_long == "Long" else -1

    # ── Compute P&L ─────────────────────────────────────────────────
    p_base    = curve.price_bond(pos_coupon / 100, pos_maturity, pos_face, pos_freq)
    p_shocked = curve_shocked.price_bond(pos_coupon / 100, pos_maturity, pos_face, pos_freq)
    pnl       = sign * (p_shocked - p_base)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Base Price ($)",    f"${p_base:,.0f}")
    c2.metric("Shocked Price ($)", f"${p_shocked:,.0f}", delta=f"${p_shocked - p_base:+,.0f}")
    c3.metric("Position P&L ($)",  f"${pnl:+,.0f}", delta_color="normal" if pnl >= 0 else "inverse")
    c4.metric("P&L (%)",           f"{pnl / abs(p_base) * 100:+.3f}%")

    # ── Curve comparison chart ───────────────────────────────────────
    t_fine  = np.linspace(curve._tenors[0], curve._tenors[-1], 300)
    z_base  = [curve.zero_rate(t) * 100 for t in t_fine]
    z_shock = [curve_shocked.zero_rate(t) * 100 for t in t_fine]

    fig_scen = go.Figure()
    fig_scen.add_trace(go.Scatter(
        x=t_fine, y=z_base,
        mode="lines", name="Base Curve",
        line=dict(color="#00b4d8", width=2),
    ))
    fig_scen.add_trace(go.Scatter(
        x=t_fine, y=z_shock,
        mode="lines", name=f"Shocked ({scen_type}, {shift_bp:+d}bp)",
        line=dict(color="#e63946", width=2, dash="dash"),
    ))
    # Shade the difference
    fig_scen.add_trace(go.Scatter(
        x=np.concatenate([t_fine, t_fine[::-1]]),
        y=np.concatenate([z_shock, z_base[::-1]]),
        fill="toself", fillcolor="rgba(230,57,70,0.10)",
        line=dict(width=0), name="Shift", showlegend=False,
    ))
    tv = [t for t in t_vals if t <= t_fine[-1]]
    tt = [_tenor_label(t) for t in tv]
    fig_scen.update_layout(
        xaxis=dict(title="Tenor", type="log", tickvals=tv, ticktext=tt),
        yaxis_title="Zero Rate (%)",
        height=360, margin=dict(t=20, b=40),
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
    )
    st.plotly_chart(fig_scen, use_container_width=True)

    # ── DV01 ladder ─────────────────────────────────────────────────
    st.subheader("DV01 Ladder — Key Tenors")
    st.caption("DV01 per $1M face value of an ATM par bond for each tenor.")

    tenors_ladder = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
    tenors_ladder = [t for t in tenors_ladder if t <= curve._tenors[-1]]
    dv01_ladder   = []
    for T in tenors_ladder:
        par = curve.zero_rate(T)
        dv = curve.dv01(par, T, 1_000_000, 2 if T > 1 else 1)
        dv01_ladder.append(dv)

    fig_ladder = go.Figure(go.Bar(
        x=[_tenor_label(t) for t in tenors_ladder],
        y=dv01_ladder,
        marker_color="#0077b6",
        hovertemplate="%{x}: DV01=$%{y:,.2f}<extra></extra>",
    ))
    fig_ladder.update_layout(
        xaxis_title="Tenor", yaxis_title="DV01 ($ per $1M face)",
        height=280, margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig_ladder, use_container_width=True)

    with st.expander("Multiple scenario comparison"):
        bps_range = [-200, -100, -50, -25, 0, +25, +50, +100, +200]
        rows_scen = []
        for bp in bps_range:
            c_sh  = curve.shifted(bp)
            p_sh  = c_sh.price_bond(pos_coupon / 100, pos_maturity, pos_face, pos_freq)
            pnl_v = sign * (p_sh - p_base)
            rows_scen.append({
                "Shift (bp)": bp,
                "Shocked Price ($)": round(p_sh, 0),
                "P&L ($)": round(pnl_v, 0),
                "P&L (%)": round(pnl_v / abs(p_base) * 100, 4),
            })
        st.dataframe(pd.DataFrame(rows_scen), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 4  ─  FRA & Futures
# ══════════════════════════════════════════════════════════════════════

with tab_fra:
    st.subheader("FRA & Interest Rate Futures")

    sub_fra, sub_fut = st.tabs(["FRA Pricer", "SOFR Futures Convexity"])

    # ── FRA ──────────────────────────────────────────────────────────
    with sub_fra:
        st.markdown(
            "A **Forward Rate Agreement** (FRA) locks in an interest rate for a future period.  \n"
            "The holder pays fixed (FRA rate) and receives floating (LIBOR/SOFR) on notional for [T₁, T₂]."
        )

        col_fra_in, col_fra_out = st.columns([1, 1.2])

        with col_fra_in:
            fra_t1      = st.number_input("T₁ — Reset date (yr)", value=1.0, min_value=0.0, max_value=29.0, step=0.25)
            fra_t2      = st.number_input("T₂ — Settlement date (yr)", value=1.5, min_value=0.25, max_value=30.0, step=0.25)
            fra_k       = st.number_input("FRA Fixed Rate K (%)", value=0.0, step=0.05, format="%.4f",
                                           help="Set to 0 to display the fair (ATM) FRA rate only.")
            fra_notional = st.number_input("Notional ($)", value=10_000_000, step=1_000_000)
            fra_side    = st.radio("Side", ["Pay Fixed", "Receive Fixed"], horizontal=True)

        # Fair FRA rate (simply compounded over [T1,T2])
        tau_fra   = fra_t2 - fra_t1
        if tau_fra <= 0:
            col_fra_out.error("T₂ must be > T₁")
        else:
            df1       = curve.discount_factor(fra_t1)
            df2       = curve.discount_factor(fra_t2)
            fra_fair  = (df1 / df2 - 1.0) / tau_fra           # simply compounded
            K_fra     = fra_k / 100.0 if fra_k != 0 else fra_fair

            # Value of FRA at inception (or now) = (fair - K) × τ × df(T2) × N
            sign_fra  = 1.0 if fra_side == "Pay Fixed" else -1.0
            fra_value = sign_fra * (fra_fair - K_fra) * tau_fra * df2 * fra_notional

            with col_fra_out:
                st.markdown("#### FRA Results")
                m1, m2 = st.columns(2)
                m1.metric("Fair FRA Rate",   f"{fra_fair*100:.4f}%")
                m2.metric("FRA Fixed Rate K", f"{K_fra*100:.4f}%")

                m3, m4 = st.columns(2)
                m3.metric("df(T₁)", f"{df1:.6f}")
                m4.metric("df(T₂)", f"{df2:.6f}")

                m5, m6 = st.columns(2)
                m5.metric("FRA Value ($)", f"${fra_value:+,.0f}")
                m6.metric("ITM/OTM",
                           "ATM" if abs(fra_fair - K_fra) < 1e-5
                           else ("ITM" if fra_value > 0 else "OTM"))

                st.caption(
                    f"**FRA formula (Hull Ch. 6):**  \n"
                    r"Value = (R_K − R_F) × τ × L × df(T₂)  "
                    "where R_F is the fair rate, R_K is the fixed rate, τ = T₂ − T₁, L = notional."
                )

        # ── Rate sensitivity scan ────────────────────────────────────
        st.markdown("#### FRA Value vs Yield Shift")
        bp_range_fra = np.linspace(-200, 200, 100)
        fra_values   = []
        for bp in bp_range_fra:
            c_sh     = curve.shifted(bp)
            df1_sh   = c_sh.discount_factor(fra_t1)
            df2_sh   = c_sh.discount_factor(fra_t2)
            fair_sh  = (df1_sh / df2_sh - 1.0) / max(tau_fra, 1e-6)
            pv       = sign_fra * (fair_sh - K_fra) * tau_fra * df2_sh * fra_notional
            fra_values.append(pv)

        fig_fra = go.Figure()
        fig_fra.add_trace(go.Scatter(
            x=bp_range_fra, y=fra_values,
            mode="lines", name="FRA Value",
            line=dict(color="#00b4d8", width=2),
        ))
        fig_fra.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_fra.update_layout(
            xaxis_title="Parallel Shift (bp)", yaxis_title="FRA Value ($)",
            height=300, margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig_fra, use_container_width=True)

    # ── Futures Convexity Adjustment ─────────────────────────────────
    with sub_fut:
        st.markdown(
            "### SOFR / Eurodollar Futures Convexity Adjustment  \n"
            "Futures rates are *higher* than equivalent forward rates because "
            "futures are marked-to-market daily (Hull's result, Ch. 6 / Ch. 31):  \n"
        )
        st.latex(r"R_{fwd} = R_{fut} - \tfrac{1}{2}\,\sigma^2\,T_1\,T_2")
        st.markdown(
            "where σ is the short-rate volatility, T₁ is the futures expiry, "
            "T₂ = T₁ + contract tenor."
        )

        col_cv1, col_cv2 = st.columns([1, 1.2])
        with col_cv1:
            fut_t1    = st.number_input("Futures expiry T₁ (yr)", value=2.0, min_value=0.1, max_value=10.0, step=0.25)
            fut_tenor = st.number_input("Contract tenor (yr, e.g. 0.25 for 3M)", value=0.25, min_value=0.08, max_value=1.0, step=0.08)
            fut_sigma = st.slider("Short-rate vol σ (%/yr)", 10, 200, 100, 5) / 100.0
            fut_rate  = st.number_input("Observed futures rate (%)", value=4.0, step=0.05, format="%.4f")

        fut_t2   = fut_t1 + fut_tenor
        conv_adj = 0.5 * fut_sigma**2 * fut_t1 * fut_t2      # in decimal
        fwd_rate = fut_rate / 100.0 - conv_adj

        with col_cv2:
            st.markdown("#### Results")
            n1, n2 = st.columns(2)
            n1.metric("Futures Rate",        f"{fut_rate:.4f}%")
            n2.metric("Convexity Adjustment", f"{conv_adj*100:.4f}% = {conv_adj*10000:.2f} bp",
                      delta="subtract from futures",
                      delta_color="off")
            n3, n4 = st.columns(2)
            n3.metric("Implied Forward Rate", f"{fwd_rate*100:.4f}%")
            n4.metric("Curve Fwd Rate",
                      f"{curve.forward_rate(max(fut_t1, 1e-4), fut_t2)*100:.4f}%")

            st.caption(
                "The convexity adjustment *increases* with σ and with time to expiry T₁.  "
                "For a 10Y futures contract, the adjustment can be 30–50+ bp.  "
                "This is why you cannot directly use futures rates for curve bootstrapping "
                "without applying this correction."
            )

        # ── Convexity adj vs expiry ──────────────────────────────────
        t1_range = np.linspace(0.1, 10.0, 100)
        ca_range = 0.5 * fut_sigma**2 * t1_range * (t1_range + fut_tenor) * 10_000  # in bps

        fig_ca = go.Figure()
        fig_ca.add_trace(go.Scatter(
            x=t1_range, y=ca_range,
            mode="lines", name=f"Adj (σ={fut_sigma*100:.0f}%)",
            line=dict(color="#f4a261", width=2),
        ))
        # Also show for 50% and 150% vol
        for sig_alt, col_alt in [(fut_sigma * 0.5, "#2a9d8f"), (fut_sigma * 1.5, "#e63946")]:
            ca_alt = 0.5 * sig_alt**2 * t1_range * (t1_range + fut_tenor) * 10_000
            fig_ca.add_trace(go.Scatter(
                x=t1_range, y=ca_alt,
                mode="lines", name=f"σ={sig_alt*100:.0f}%",
                line=dict(color=col_alt, width=1.5, dash="dot"),
            ))
        fig_ca.add_vline(x=fut_t1, line_dash="dash", line_color="gray",
                          annotation_text=f"T₁={fut_t1}Y", annotation_position="top")
        fig_ca.update_layout(
            xaxis_title="Futures Expiry T₁ (yr)",
            yaxis_title="Convexity Adjustment (bp)",
            height=320, margin=dict(t=20, b=40),
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig_ca, use_container_width=True)

        with st.expander("Why does this convexity adjustment exist? (Hull Ch. 6)"):
            st.markdown(
                "**Futures vs Forwards — the key difference:**  \n\n"
                "A forward contract is settled at maturity — no daily cash flows. "
                "A futures contract is **marked-to-market daily**, meaning gains/losses "
                "are settled in cash every day.  \n\n"
                "Because futures gains can be reinvested when rates rise (and losses occur "
                "when rates fall), the futures price is *biased upward* relative to the "
                "equivalent forward.  \n\n"
                "Mathematically, under a simple Gaussian short-rate model:  \n"
                r"$$R_{fwd}(T_1, T_2) = R_{fut}(T_1, T_2) - \tfrac{1}{2}\sigma^2 T_1 T_2$$"
                "  \n\nThis means using futures rates directly in a bootstrap overstates "
                "the implied forward rates. The bigger the vol σ and the longer the "
                "expiry T₁, the larger the correction."
            )
