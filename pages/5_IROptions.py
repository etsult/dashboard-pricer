"""
IR Options — Pricer, Book Monitor & Curve Simulator.

Instruments : Cap / Floor (caplet strip) · Payer / Receiver Swaption
Indexes     : SOFR · Term SOFR 1/3/6/12M · SOFR 3M · LIBOR 3M
              EURIBOR 1/3/6M · €STR · SONIA · SONIA 3M
Vol model   : Flat normal vol  |  ZABR smile (Antonov-Piterbarg)
Discounting : Two-curve  OIS discount + index basis spread
Scenarios   : Parallel shift · Tilt (front/back) · Belly twist
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.special import ndtr

from market_data.providers.fred import fetch_usd_curve
from market_data.curves.rate_curve import RateCurve
from pricer.ir.indexes import INDEX_CATALOG
from pricer.ir.instruments import IRPosition, Book
from pricer.ir.engine import BookEngine
from pricer.ir.zabr import zabr_atm_vol, smile_vol_strip

st.set_page_config(page_title="IR Options", layout="wide")
st.title("IR Options — Pricer & Book Monitor")

# ═══════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════

if "book" not in st.session_state:
    st.session_state["book"] = Book()

book: Book = st.session_state["book"]

# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR — Curve
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
            fred_key = st.text_input(
                "FRED API Key", type="password",
                help="Free at fred.stlouisfed.org",
            )
    else:
        fred_key = None

    st.divider()
    st.header("Vol Model")
    use_zabr = st.toggle("ZABR smile", value=False,
                          help="Use ZABR normal vol per caplet/swaption. Slower but smile-aware.")


# ═══════════════════════════════════════════════════════════════════════
# CURVE BUILDING
# ═══════════════════════════════════════════════════════════════════════

_DEFAULT_PAR: dict[float, float] = {
    1/12: 0.0433, 3/12: 0.0427, 6/12: 0.0420,
    1.0: 0.0410,  2.0: 0.0405,  3.0: 0.0408,
    5.0: 0.0420,  7.0: 0.0430, 10.0: 0.0445, 30.0: 0.0470,
}


@st.cache_data(ttl=300, show_spinner="Fetching FRED curve…")
def _load_fred(key: str) -> dict[float, float]:
    return fetch_usd_curve(api_key=key or None)


if curve_source == "FRED (live USD)":
    par_yields = _load_fred(fred_key or "")
else:
    st.subheader("Manual Yield Curve (USD)")
    tenor_map = {
        "1M": 1/12, "3M": 3/12, "6M": 6/12,
        "1Y": 1.0, "2Y": 2.0, "3Y": 3.0,
        "5Y": 5.0, "7Y": 7.0, "10Y": 10.0, "30Y": 30.0,
    }
    default_rows = [
        {"Tenor": k, "Rate (%)": round(_DEFAULT_PAR[v] * 100, 3)}
        for k, v in tenor_map.items()
    ]
    edited = st.data_editor(
        pd.DataFrame(default_rows), use_container_width=True, hide_index=True
    )
    par_yields = {
        tenor_map[row["Tenor"]]: row["Rate (%)"] / 100.0
        for _, row in edited.iterrows()
        if row["Tenor"] in tenor_map
    }

curve = RateCurve(par_yields)

# ═══════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════

tab_curve, tab_pricer, tab_book, tab_sim = st.tabs([
    "📈 Yield Curve",
    "🎯 Single Instrument",
    "📋 Option Book",
    "🔀 Curve Simulator",
])


# ── Tab 1: Yield Curve ────────────────────────────────────────────────
with tab_curve:
    st.subheader("USD Zero-Rate Curve")
    zdf = curve.zero_curve_df()

    _LABELS = {
        1/12: "1M", 2/12: "2M", 3/12: "3M", 6/12: "6M",
        1.0: "1Y", 2.0: "2Y", 3.0: "3Y", 5.0: "5Y",
        7.0: "7Y", 10.0: "10Y", 15.0: "15Y", 20.0: "20Y", 30.0: "30Y",
    }
    tick_vals = list(zdf["Tenor"])
    tick_text = [_LABELS.get(t, f"{t:.0f}Y") for t in tick_vals]

    fig_z = go.Figure()
    fig_z.add_trace(go.Scatter(
        x=zdf["Tenor"], y=zdf["Zero Rate (%)"],
        mode="lines+markers",
        line=dict(color="#00b4d8", width=2.5),
        marker=dict(size=7),
        hovertemplate="%{text}: %{y:.3f}%<extra></extra>",
        text=tick_text,
    ))
    fig_z.update_layout(
        xaxis=dict(title="Tenor", type="log", tickvals=tick_vals, ticktext=tick_text),
        yaxis_title="Zero Rate (%)",
        height=360, margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig_z, use_container_width=True)

    fdf = curve.forward_curve_df()
    fdf["Tenor Label"] = [_LABELS.get(t, f"{t:.0f}Y") for t in fdf["Tenor"]]
    st.dataframe(
        fdf[["Tenor Label", "Zero Rate (%)", "Forward Rate (%)", "Par Yield (%)"]].rename(
            columns={"Tenor Label": "Tenor"}
        ).round(4),
        use_container_width=True, hide_index=True,
    )


# ── Tab 2: Single Instrument ──────────────────────────────────────────
with tab_pricer:
    st.subheader("Single Instrument Pricer")

    c_l, c_r = st.columns([1, 1.8])
    with c_l:
        instr = st.selectbox(
            "Instrument",
            ["Cap", "Floor", "Payer Swaption", "Receiver Swaption"],
        )
        idx_key = st.selectbox(
            "Index",
            list(INDEX_CATALOG.keys()),
            format_func=lambda k: INDEX_CATALOG[k]["label"],
        )
        idx_info = INDEX_CATALOG[idx_key]
        freq     = 1.0 / idx_info["reset_freq"]

        notional = st.number_input("Notional", value=10_000_000, step=1_000_000)

        if instr in ("Cap", "Floor"):
            maturity = st.number_input(
                "Cap/Floor Maturity (Y)", value=5.0, min_value=0.25, max_value=30.0, step=0.25
            )
            expiry_y = maturity
            tenor_y  = maturity
            atm_ref  = curve.forward_rate(max(freq * 0.5, 1e-4), freq) * 100
        else:
            exp_labels = {"1M": 1/12, "3M": 3/12, "6M": 6/12, "1Y": 1.0, "2Y": 2.0, "5Y": 5.0}
            ten_labels = {"1Y": 1.0, "2Y": 2.0, "5Y": 5.0, "7Y": 7.0, "10Y": 10.0, "20Y": 20.0, "30Y": 30.0}
            exp_lbl = st.selectbox("Expiry", list(exp_labels.keys()))
            ten_lbl = st.selectbox("Swap Tenor", list(ten_labels.keys()), index=2)
            expiry_y = exp_labels[exp_lbl]
            tenor_y  = ten_labels[ten_lbl]
            atm_ref  = curve.par_swap_rate(expiry_y, expiry_y + tenor_y, freq) * 100

        vol_bps = st.slider("ATM Normal Vol (bps/yr)", 10, 300, 65, 5)
        sigma_n = vol_bps / 10_000.0

        st.caption(f"ATM ref: **{atm_ref:.4f}%** · Basis: {idx_info['basis_bps']}bps")
        strike_pct = st.number_input(
            "Strike (%)", value=round(atm_ref, 3), step=0.05, format="%.4f"
        )
        K = strike_pct / 100.0

        direction = st.radio("Direction", ["Long", "Short"], horizontal=True)
        dir_int = 1 if direction == "Long" else -1

    instr_key_map = {
        "Cap": "cap", "Floor": "floor",
        "Payer Swaption": "payer_swaption", "Receiver Swaption": "receiver_swaption",
    }
    pos = IRPosition(
        instrument=instr_key_map[instr],
        index_key=idx_key,
        notional=notional,
        strike=K,
        expiry_y=expiry_y,
        tenor_y=tenor_y,
        sigma_n=sigma_n,
        direction=dir_int,
    )
    engine = BookEngine(curve, Book([pos]), use_zabr=use_zabr)
    result = engine.price_book()

    with c_r:
        if not result.empty:
            pv   = result.iloc[0]["pv"]
            pv_bps = pv / notional * 10_000
            atm  = result.iloc[0]["atm_fwd_pct"]

            cols = st.columns(3)
            cols[0].metric(f"{instr} PV", f"${pv:,.0f}")
            cols[1].metric("PV (bps notional)", f"{pv_bps:.2f}")
            cols[2].metric("ATM Forward", f"{atm:.4f}%")

        # ZABR smile chart
        st.subheader("Vol Smile (ZABR)")
        alpha, nu, rho = zabr_atm_vol(expiry_y, tenor_y, idx_info["ccy"])
        tau_smile = expiry_y if instr not in ("Cap", "Floor") else max(freq, 0.25)
        f_ref = atm_ref / 100
        k_range = np.linspace(max(f_ref - 0.02, 0.0005), f_ref + 0.02, 100)
        smile  = smile_vol_strip(f_ref, k_range, tau_smile, alpha, nu, rho) * 10_000

        fig_sm = go.Figure()
        fig_sm.add_trace(go.Scatter(
            x=k_range * 100, y=smile,
            mode="lines", line=dict(color="#f4a261", width=2), name="ZABR σ_N",
        ))
        fig_sm.add_vline(x=atm_ref, line_dash="dash", line_color="#00b4d8",
                         annotation_text="ATM")
        fig_sm.add_vline(x=strike_pct, line_dash="dot", line_color="gray",
                         annotation_text="Strike")
        fig_sm.update_layout(
            xaxis_title="Strike (%)",
            yaxis_title="Normal Vol (bps/yr)",
            height=320, margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig_sm, use_container_width=True)

    # Sensitivity strips
    st.divider()
    cola, colb = st.columns(2)

    with cola:
        st.subheader("PV vs Strike")
        k_strip = np.linspace(max(f_ref * 0.5, 0.0005), f_ref * 1.5, 80)
        pvs_k = []
        for k_ in k_strip:
            p_ = IRPosition(instrument=pos.instrument, index_key=idx_key, notional=notional,
                            strike=k_, expiry_y=expiry_y, tenor_y=tenor_y,
                            sigma_n=sigma_n, direction=dir_int)
            r_ = BookEngine(curve, Book([p_]), use_zabr=False).price_book()
            pvs_k.append(r_.iloc[0]["pv"] if not r_.empty else 0.0)

        fig_k = go.Figure()
        fig_k.add_trace(go.Scatter(x=k_strip * 100, y=pvs_k,
                                    mode="lines", line=dict(color="#e63946")))
        fig_k.add_vline(x=strike_pct, line_dash="dash", line_color="gray")
        fig_k.update_layout(xaxis_title="Strike (%)", yaxis_title="PV ($)",
                             height=280, margin=dict(t=10, b=40))
        st.plotly_chart(fig_k, use_container_width=True)

    with colb:
        st.subheader("PV vs Vol (Vega)")
        vol_strip = np.linspace(10, 250, 60) / 10_000.0
        pvs_v = []
        for v_ in vol_strip:
            p_ = IRPosition(instrument=pos.instrument, index_key=idx_key, notional=notional,
                            strike=K, expiry_y=expiry_y, tenor_y=tenor_y,
                            sigma_n=v_, direction=dir_int)
            r_ = BookEngine(curve, Book([p_]), use_zabr=False).price_book()
            pvs_v.append(r_.iloc[0]["pv"] if not r_.empty else 0.0)

        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(x=vol_strip * 10_000, y=pvs_v,
                                    mode="lines", line=dict(color="#2a9d8f")))
        fig_v.add_vline(x=vol_bps, line_dash="dash", line_color="gray")
        fig_v.update_layout(xaxis_title="Normal Vol (bps/yr)", yaxis_title="PV ($)",
                             height=280, margin=dict(t=10, b=40))
        st.plotly_chart(fig_v, use_container_width=True)

    # Add to book
    st.divider()
    if st.button("➕  Add to Option Book", use_container_width=True):
        book.add(pos)
        st.success(f"Added: {pos.label}")
        st.rerun()


# ── Tab 3: Option Book ────────────────────────────────────────────────
with tab_book:
    st.subheader("Option Book")

    if book.is_empty():
        st.info("Book is empty. Price an instrument in the **Single Instrument** tab and click **Add to Book**.")
    else:
        eng = BookEngine(curve, book, use_zabr=use_zabr)
        price_df  = eng.price_book()
        greeks_df = eng.greeks_book(bump_bp=1.0)

        # Position table
        display_cols = ["label", "instrument", "index", "expiry_y", "tenor_y",
                        "strike_pct", "atm_fwd_pct", "sigma_bps", "pv"]
        disp = price_df[display_cols].copy()
        disp.columns = ["Position", "Type", "Index", "Expiry (Y)", "Tenor (Y)",
                        "Strike (%)", "ATM (%)", "Vol (bps)", "PV ($)"]
        disp["PV ($)"]     = disp["PV ($)"].map(lambda x: f"${x:,.0f}")
        disp["Strike (%)"] = disp["Strike (%)"].map(lambda x: f"{x:.4f}%")
        disp["ATM (%)"]    = disp["ATM (%)"].map(lambda x: f"{x:.4f}%")
        disp["Vol (bps)"]  = disp["Vol (bps)"].map(lambda x: f"{x:.1f}")

        st.dataframe(disp, use_container_width=True, hide_index=True)

        # Remove position
        rem_idx = st.number_input(
            "Remove position (row #, 0-indexed)", min_value=0,
            max_value=max(len(book) - 1, 0), step=1, value=0,
        )
        if st.button("🗑  Remove position"):
            book.remove(rem_idx)
            st.rerun()

        st.divider()

        # Book-level metrics
        total_pv   = price_df["pv"].sum()
        total_dv01 = greeks_df["DV01 ($)"].sum()
        total_vega = greeks_df["Vega/bp ($)"].sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total PV", f"${total_pv:,.0f}")
        c2.metric("Total DV01 (1bp)", f"${total_dv01:,.0f}")
        c3.metric("Total Vega (1bp σ)", f"${total_vega:,.0f}")

        # Greeks table
        st.subheader("Greeks per Position")
        g_disp = greeks_df[["label", "PV ($)", "DV01 ($)", "Vega/bp ($)"]].copy()
        g_disp.columns = ["Position", "PV ($)", "DV01 / 1bp", "Vega / 1bp σ"]
        st.dataframe(g_disp, use_container_width=True, hide_index=True)

        # PV bar chart
        fig_book = go.Figure(go.Bar(
            x=price_df["label"],
            y=price_df["pv"],
            marker_color=["#2a9d8f" if v >= 0 else "#e63946" for v in price_df["pv"]],
        ))
        fig_book.update_layout(
            xaxis_title="Position", yaxis_title="PV ($)",
            height=320, margin=dict(t=10, b=80),
        )
        fig_book.update_xaxes(tickangle=-30)
        st.plotly_chart(fig_book, use_container_width=True)

        if st.button("🗑  Clear entire book"):
            st.session_state["book"] = Book()
            st.rerun()


# ── Tab 4: Curve Simulator ─────────────────────────────────────────────
with tab_sim:
    st.subheader("Curve Simulator — Real-Time Greeks & P&L")
    st.caption(
        "Shift the yield curve and see how the book P&L and Greeks change live. "
        "Uses the same multi-curve engine as the pricer."
    )

    col_sliders, col_results = st.columns([1, 2])

    with col_sliders:
        shift_bp   = st.slider("Parallel Shift (bp)",  -200, 200, 0, 5)
        front_bp   = st.slider("Front Tilt (bp)",       -100, 100, 0, 5)
        back_bp    = st.slider("Back Tilt (bp)",        -100, 100, 0, 5)
        belly_bp   = st.slider("Belly Twist (bp)",      -100, 100, 0, 5)

    # Build scenario curve
    scenario = (
        curve
        .shifted(shift_bp)
        .steepened(front_bp, back_bp)
        .twisted(belly_bp)
    )

    with col_results:
        # Zero curve comparison
        zdf_base = curve.zero_curve_df()
        zdf_sc   = scenario.zero_curve_df()
        labels   = [_LABELS.get(t, f"{t:.0f}Y") for t in zdf_base["Tenor"]]

        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=zdf_base["Tenor"], y=zdf_base["Zero Rate (%)"],
            mode="lines", name="Base", line=dict(color="#00b4d8", dash="dot", width=1.5),
        ))
        fig_sc.add_trace(go.Scatter(
            x=zdf_sc["Tenor"], y=zdf_sc["Zero Rate (%)"],
            mode="lines+markers", name="Scenario", line=dict(color="#f4a261", width=2.5),
            text=labels, hovertemplate="%{text}: %{y:.3f}%<extra></extra>",
        ))
        fig_sc.update_layout(
            xaxis=dict(title="Tenor", type="log", tickvals=tick_vals, ticktext=tick_text),
            yaxis_title="Zero Rate (%)",
            height=300, margin=dict(t=10, b=40),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    if book.is_empty():
        st.info("Add positions to the book to see scenario P&L.")
    else:
        eng_base = BookEngine(curve,    book, use_zabr=False)
        eng_sc   = BookEngine(scenario, book, use_zabr=False)

        pv_base = eng_base.price_book()["pv"]
        pv_sc   = eng_sc.price_book()["pv"]
        pnl     = pv_sc - pv_base

        # Metrics
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Base Book PV",    f"${pv_base.sum():,.0f}")
        mc2.metric("Scenario PV",     f"${pv_sc.sum():,.0f}")
        mc3.metric("Scenario P&L",    f"${pnl.sum():,.0f}",
                   delta=f"{pnl.sum():+,.0f}")

        # P&L per position
        pnl_df = pd.DataFrame({
            "Position":    [p.label for p in book.positions],
            "Base PV ($)": pv_base.values,
            "Scen PV ($)": pv_sc.values,
            "P&L ($)":     pnl.values,
        })
        pnl_df["P&L ($)"] = pnl_df["P&L ($)"].round(0)

        fig_pnl = go.Figure(go.Bar(
            x=pnl_df["Position"],
            y=pnl_df["P&L ($)"],
            marker_color=["#2a9d8f" if v >= 0 else "#e63946" for v in pnl_df["P&L ($)"]],
        ))
        fig_pnl.update_layout(
            xaxis_title="Position", yaxis_title="P&L ($)",
            title="Scenario P&L by Position",
            height=320, margin=dict(t=40, b=80),
        )
        fig_pnl.update_xaxes(tickangle=-30)
        st.plotly_chart(fig_pnl, use_container_width=True)

        # Full scenario sweep — book PV vs parallel shift
        st.subheader("Book PV vs Parallel Shift")
        sweep_bps = np.arange(-200, 205, 10)
        sweep_pvs = [
            BookEngine(curve.shifted(float(b)), book, use_zabr=False).price_book()["pv"].sum()
            for b in sweep_bps
        ]
        base_pv_total = pv_base.sum()

        fig_sweep = go.Figure()
        fig_sweep.add_trace(go.Scatter(
            x=sweep_bps, y=sweep_pvs,
            mode="lines", line=dict(color="#00b4d8", width=2.5), name="Book PV",
        ))
        fig_sweep.add_hline(y=base_pv_total, line_dash="dash", line_color="gray",
                             annotation_text="Base")
        fig_sweep.add_vline(x=shift_bp, line_dash="dot", line_color="#f4a261",
                             annotation_text="Current")
        fig_sweep.update_layout(
            xaxis_title="Parallel Shift (bp)", yaxis_title="PV ($)",
            height=300, margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig_sweep, use_container_width=True)

        with st.expander("Full scenario P&L table"):
            st.dataframe(
                pnl_df.style.format({"Base PV ($)": "${:,.0f}", "Scen PV ($)": "${:,.0f}", "P&L ($)": "${:+,.0f}"}),
                use_container_width=True, hide_index=True,
            )
