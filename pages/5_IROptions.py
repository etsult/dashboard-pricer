"""
IR Options — Pricer, Book Monitor & Curve Simulator.

Instruments : Cap / Floor (caplet strip) · Payer / Receiver Swaption
Indexes     : SOFR · Term SOFR 1/3/6/12M · SOFR 3M · LIBOR 3M
              EURIBOR 1/3/6M · €STR · SONIA · SONIA 3M
Vol model   : Flat normal vol  |  ZABR smile (Antonov-Piterbarg)
Engines     : Fast (numpy) · QuantLib benchmark · Neural Network (SwaptionNetV2)
Discounting : Two-curve  OIS discount + index basis spread
Scenarios   : Parallel shift · Tilt (front/back) · Belly twist
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_data.providers.fred import fetch_usd_curve
from market_data.curves.rate_curve import RateCurve
from pricer.ir.indexes import INDEX_CATALOG
from pricer.ir.instruments import IRPosition, Book
from pricer.ir.fast_engine import FastBookEngine
from pricer.ir.zabr import zabr_atm_vol, smile_vol_strip
from api.client import get_client

_client = get_client()

st.set_page_config(page_title="IR Options", layout="wide")
st.title("IR Options — Pricer & Book Monitor")

# ═══════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════

if "book" not in st.session_state:
    st.session_state["book"] = Book()

book: Book = st.session_state["book"]

# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR
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
    use_zabr = st.toggle(
        "ZABR smile", value=False,
        help="ZABR normal vol per caplet/swaption. Applied in Fast engine only.",
    )

    st.divider()
    st.header("Pricer Engine")
    engine_choice = st.radio(
        "Engine",
        ["Fast (Numpy)", "QuantLib", "Neural Network"],
        captions=[
            "Vectorized Bachelier · analytical Greeks",
            "QuantLib exact benchmark · PV only · slow",
            "SwaptionNetV2 · swaptions only · PV + Greeks",
        ],
    )

    # Availability guards
    if engine_choice == "QuantLib":
        from pricer.ir.ql_engine import ql_available
        if not ql_available():
            st.error("`pip install QuantLib` then restart Streamlit.")
    elif engine_choice == "Neural Network":
        try:
            import torch as _t; del _t
        except ImportError:
            st.error("`pip install torch` then restart Streamlit.")


# ═══════════════════════════════════════════════════════════════════════
# CURVE BUILDING
# ═══════════════════════════════════════════════════════════════════════

_DEFAULT_PAR: dict[float, float] = {
    1/12: 0.0433, 3/12: 0.0427, 6/12: 0.0420,
    1.0: 0.0410,  2.0: 0.0405,  3.0: 0.0408,
    5.0: 0.0420,  7.0: 0.0430, 10.0: 0.0445, 30.0: 0.0470,
}

_LABELS = {
    1/12: "1M", 2/12: "2M", 3/12: "3M", 6/12: "6M",
    1.0: "1Y", 2.0: "2Y", 3.0: "3Y", 5.0: "5Y",
    7.0: "7Y", 10.0: "10Y", 15.0: "15Y", 20.0: "20Y", 30.0: "30Y",
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

# Shared tick info (used in Yield Curve and Curve Simulator tabs)
_zdf_base   = curve.zero_curve_df()
_tick_vals  = list(_zdf_base["Tenor"])
_tick_text  = [_LABELS.get(t, f"{t:.0f}Y") for t in _tick_vals]

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

    fig_z = go.Figure()
    fig_z.add_trace(go.Scatter(
        x=_zdf_base["Tenor"], y=_zdf_base["Zero Rate (%)"],
        mode="lines+markers",
        line=dict(color="#00b4d8", width=2.5),
        marker=dict(size=7),
        hovertemplate="%{text}: %{y:.3f}%<extra></extra>",
        text=_tick_text,
    ))
    fig_z.update_layout(
        xaxis=dict(title="Tenor", type="log",
                   tickvals=_tick_vals, ticktext=_tick_text),
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
    st.caption(
        f"Engine: **{engine_choice}**"
        + (" · ZABR smile active (Fast only)" if use_zabr else "")
    )

    c_l, c_r = st.columns([1, 1.8])
    with c_l:
        instr = st.selectbox(
            "Instrument",
            ["Cap", "Floor", "Payer Swaption", "Receiver Swaption"],
        )
        idx_key  = st.selectbox(
            "Index",
            list(INDEX_CATALOG.keys()),
            format_func=lambda k: INDEX_CATALOG[k]["label"],
        )
        idx_info = INDEX_CATALOG[idx_key]
        freq     = 1.0 / idx_info["reset_freq"]
        notional = st.number_input("Notional", value=10_000_000, step=1_000_000)

        instr_key_map = {
            "Cap": "cap", "Floor": "floor",
            "Payer Swaption": "payer_swaption",
            "Receiver Swaption": "receiver_swaption",
        }
        instr_key   = instr_key_map[instr]
        is_swaption = instr_key in ("payer_swaption", "receiver_swaption")

        if not is_swaption:
            maturity = st.number_input(
                "Cap/Floor Maturity (Y)", value=5.0,
                min_value=0.25, max_value=30.0, step=0.25,
            )
            expiry_y = maturity
            tenor_y  = maturity
            atm_ref  = curve.forward_rate(max(freq * 0.5, 1e-4), freq) * 100
        else:
            exp_labels = {
                "1M": 1/12, "3M": 3/12, "6M": 6/12,
                "1Y": 1.0,  "2Y": 2.0,  "5Y": 5.0,
            }
            ten_labels = {
                "1Y": 1.0, "2Y": 2.0,  "5Y": 5.0,
                "7Y": 7.0, "10Y": 10.0, "20Y": 20.0, "30Y": 30.0,
            }
            exp_lbl  = st.selectbox("Expiry", list(exp_labels.keys()))
            ten_lbl  = st.selectbox("Swap Tenor", list(ten_labels.keys()), index=2)
            expiry_y = exp_labels[exp_lbl]
            tenor_y  = ten_labels[ten_lbl]
            atm_ref  = curve.par_swap_rate(expiry_y, expiry_y + tenor_y, freq) * 100

        vol_bps  = st.slider("ATM Normal Vol (bps/yr)", 10, 300, 65, 5)
        sigma_n  = vol_bps / 10_000.0

        st.caption(f"ATM ref: **{atm_ref:.4f}%** · Basis: {idx_info['basis_bps']}bps")
        strike_pct = st.number_input(
            "Strike (%)", value=round(atm_ref, 3), step=0.05, format="%.4f",
        )
        K = strike_pct / 100.0

        direction = st.radio("Direction", ["Long", "Short"], horizontal=True)
        dir_int   = 1 if direction == "Long" else -1

    pos = IRPosition(
        instrument=instr_key, index_key=idx_key, notional=notional,
        strike=K, expiry_y=expiry_y, tenor_y=tenor_y,
        sigma_n=sigma_n, direction=dir_int,
    )

    # ── Price with selected engine ─────────────────────────────────────────
    with c_r:
        # NN only supports swaptions — fall back to Fast for caps/floors
        effective_engine = engine_choice
        if engine_choice == "Neural Network" and not is_swaption:
            st.info("NN engine supports swaptions only — showing Fast engine result.")
            effective_engine = "Fast (Numpy)"

        try:
            if effective_engine == "Fast (Numpy)":
                # Honour ZABR toggle via BookEngine (which has the ZABR path)
                from pricer.ir.engine import BookEngine
                res = BookEngine(curve, Book([pos]), use_zabr=use_zabr).price_book()
                pv  = float(res.iloc[0]["pv"])
                atm = float(res.iloc[0]["atm_fwd_pct"])

            elif effective_engine == "QuantLib":
                if use_zabr:
                    st.caption("ℹ ZABR not applied to QuantLib — using flat vol.")
                cmp  = _client.benchmark_book(Book([pos]), curve)
                fast = _client.price_book(Book([pos]), curve)
                pv   = float(cmp.iloc[0]["pv_ql"])
                atm  = float(fast.iloc[0]["atm_pct"])

            else:  # Neural Network
                if use_zabr:
                    st.caption("ℹ ZABR not applied to NN — using flat vol.")
                nn = _client.nn_price_book(Book([pos]), curve)
                pv  = float(nn.iloc[0]["pv"])
                atm = float(nn.iloc[0]["atm_pct"])

            pv_bps = pv / notional * 10_000
            cols   = st.columns(3)
            cols[0].metric(f"{instr} PV  [{effective_engine}]", f"${pv:,.0f}")
            cols[1].metric("PV (bps notional)", f"{pv_bps:.2f}")
            cols[2].metric("ATM Forward", f"{atm:.4f}%")
            f_ref = atm / 100.0

        except Exception as exc:
            st.error(f"Pricing error: {exc}")
            pv, atm, f_ref = 0.0, atm_ref, atm_ref / 100.0

        # ZABR smile chart — always shown, independent of pricer engine
        st.subheader("Vol Smile (ZABR)")
        alpha, nu, rho = zabr_atm_vol(expiry_y, tenor_y, idx_info["ccy"])
        tau_smile = expiry_y if is_swaption else max(freq, 0.25)
        k_range   = np.linspace(max(f_ref - 0.02, 0.0005), f_ref + 0.02, 100)
        smile     = smile_vol_strip(f_ref, k_range, tau_smile, alpha, nu, rho) * 10_000

        fig_sm = go.Figure()
        fig_sm.add_trace(go.Scatter(
            x=k_range * 100, y=smile,
            mode="lines", line=dict(color="#f4a261", width=2), name="ZABR σ_N",
        ))
        fig_sm.add_vline(x=atm, line_dash="dash", line_color="#00b4d8",
                         annotation_text="ATM")
        fig_sm.add_vline(x=strike_pct, line_dash="dot", line_color="gray",
                         annotation_text="Strike")
        fig_sm.update_layout(
            xaxis_title="Strike (%)", yaxis_title="Normal Vol (bps/yr)",
            height=320, margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig_sm, use_container_width=True)

    # Sensitivity strips — batched via FastBookEngine (vectorized, engine-agnostic)
    st.divider()
    st.caption("Sensitivity strips always use the Fast engine (vectorized batch).")
    cola, colb = st.columns(2)

    with cola:
        st.subheader("PV vs Strike")
        k_strip = np.linspace(max(f_ref * 0.5, 0.0005), f_ref * 1.5, 80)
        pvs_k   = FastBookEngine(curve, Book([
            IRPosition(instrument=instr_key, index_key=idx_key, notional=notional,
                       strike=k_, expiry_y=expiry_y, tenor_y=tenor_y,
                       sigma_n=sigma_n, direction=dir_int)
            for k_ in k_strip
        ])).price_book()["pv"].tolist()

        fig_k = go.Figure(go.Scatter(
            x=k_strip * 100, y=pvs_k,
            mode="lines", line=dict(color="#e63946"),
        ))
        fig_k.add_vline(x=strike_pct, line_dash="dash", line_color="gray")
        fig_k.update_layout(xaxis_title="Strike (%)", yaxis_title="PV ($)",
                             height=280, margin=dict(t=10, b=40))
        st.plotly_chart(fig_k, use_container_width=True)

    with colb:
        st.subheader("PV vs Vol (Vega)")
        vol_strip = np.linspace(10, 250, 60) / 10_000.0
        pvs_v     = FastBookEngine(curve, Book([
            IRPosition(instrument=instr_key, index_key=idx_key, notional=notional,
                       strike=K, expiry_y=expiry_y, tenor_y=tenor_y,
                       sigma_n=v_, direction=dir_int)
            for v_ in vol_strip
        ])).price_book()["pv"].tolist()

        fig_v = go.Figure(go.Scatter(
            x=vol_strip * 10_000, y=pvs_v,
            mode="lines", line=dict(color="#2a9d8f"),
        ))
        fig_v.add_vline(x=vol_bps, line_dash="dash", line_color="gray")
        fig_v.update_layout(xaxis_title="Normal Vol (bps/yr)", yaxis_title="PV ($)",
                             height=280, margin=dict(t=10, b=40))
        st.plotly_chart(fig_v, use_container_width=True)

    st.divider()
    if st.button("➕  Add to Option Book", use_container_width=True):
        book.add(pos)
        st.success(f"Added: {pos.label}")
        st.rerun()


# ── Tab 3: Option Book ────────────────────────────────────────────────
with tab_book:
    st.subheader("Option Book")
    st.caption(f"Engine: **{engine_choice}**")

    if book.is_empty():
        st.info(
            "Book is empty. Price an instrument in **Single Instrument** "
            "and click **Add to Book**."
        )
    else:
        # Base position metadata always from fast engine
        price_df   = _client.price_book(book, curve)
        has_greeks = False

        try:
            if engine_choice == "Fast (Numpy)":
                risk_df, _ = _client.fast_risk(book, curve)
                price_df["pv"]   = risk_df["pv"].values
                price_df["dv01"] = risk_df["dv01"].values
                price_df["vega"] = risk_df["vega"].values
                has_greeks = True

            elif engine_choice == "QuantLib":
                cmp_df = _client.benchmark_book(book, curve)
                price_df["pv"]       = cmp_df["pv_ql"].values
                price_df["diff_bps"] = cmp_df["diff_bps"].values

            else:  # Neural Network
                nn_df = _client.nn_price_book(book, curve)
                price_df["pv"]   = nn_df["pv"].values
                price_df["dv01"] = nn_df["dv01"].values
                price_df["vega"] = nn_df["vega"].values
                has_greeks = True

        except Exception as exc:
            st.error(f"Engine error: {exc}")
            st.stop()

        # ── Position table ────────────────────────────────────────────────────
        disp_cols  = ["label", "instrument", "index_key",
                      "expiry_y", "tenor_y", "strike_pct", "atm_pct",
                      "sigma_bps", "pv"]
        if engine_choice == "QuantLib" and "diff_bps" in price_df.columns:
            disp_cols.append("diff_bps")

        disp = price_df[[c for c in disp_cols if c in price_df.columns]].copy()
        disp = disp.rename(columns={
            "label": "Position", "instrument": "Type", "index_key": "Index",
            "expiry_y": "Expiry (Y)", "tenor_y": "Tenor (Y)",
            "strike_pct": "Strike (%)", "atm_pct": "ATM (%)",
            "sigma_bps": "Vol (bps)", "pv": "PV ($)",
            "diff_bps": "Δ vs Fast (bps)",
        })
        fmt = {
            "PV ($)": "${:,.0f}", "Strike (%)": "{:.4f}%",
            "ATM (%)": "{:.4f}%", "Vol (bps)": "{:.1f}",
        }
        if "Δ vs Fast (bps)" in disp.columns:
            fmt["Δ vs Fast (bps)"] = "{:.2f}"

        st.dataframe(
            disp.style.format(fmt, na_rep="N/A"),
            use_container_width=True, hide_index=True,
        )

        rem_idx = st.number_input(
            "Remove position (row #, 0-indexed)", min_value=0,
            max_value=max(len(book) - 1, 0), step=1, value=0,
        )
        if st.button("🗑  Remove position"):
            book.remove(rem_idx)
            st.rerun()

        st.divider()

        # ── Book-level metrics ────────────────────────────────────────────────
        total_pv = price_df["pv"].sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total PV", f"${total_pv:,.0f}")

        if has_greeks:
            c2.metric("Total DV01 (1bp)", f"${price_df['dv01'].sum():,.0f}")
            c3.metric("Total Vega (1bp σ)", f"${price_df['vega'].sum():,.0f}")
        else:
            c2.metric("DV01", "—  (QL: no analytical Greeks)")
            c3.metric("Vega", "—")

        # ── Greeks table (Fast + NN only) ─────────────────────────────────────
        if has_greeks:
            st.subheader("Greeks per Position")
            g_disp = price_df[["label", "pv", "dv01", "vega"]].rename(columns={
                "label": "Position", "pv": "PV ($)",
                "dv01": "DV01 / 1bp ($)", "vega": "Vega / 1bp σ ($)",
            })
            st.dataframe(
                g_disp.style.format({
                    "PV ($)": "${:,.0f}",
                    "DV01 / 1bp ($)": "${:,.0f}",
                    "Vega / 1bp σ ($)": "${:,.0f}",
                }, na_rep="N/A"),
                use_container_width=True, hide_index=True,
            )

        # ── PV bar chart ──────────────────────────────────────────────────────
        fig_book = go.Figure(go.Bar(
            x=price_df["label"],
            y=price_df["pv"],
            marker_color=[
                "#2a9d8f" if (v == v and v >= 0) else "#e63946"
                for v in price_df["pv"]
            ],
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
    st.caption("Scenario repricing always uses the Fast engine (vectorized).")

    col_sliders, col_results = st.columns([1, 2])

    with col_sliders:
        shift_bp = st.slider("Parallel Shift (bp)",  -200, 200, 0, 5)
        front_bp = st.slider("Front Tilt (bp)",       -100, 100, 0, 5)
        back_bp  = st.slider("Back Tilt (bp)",        -100, 100, 0, 5)
        belly_bp = st.slider("Belly Twist (bp)",      -100, 100, 0, 5)

    scenario = curve.shifted(shift_bp).steepened(front_bp, back_bp).twisted(belly_bp)

    with col_results:
        zdf_sc = scenario.zero_curve_df()
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=_zdf_base["Tenor"], y=_zdf_base["Zero Rate (%)"],
            mode="lines", name="Base",
            line=dict(color="#00b4d8", dash="dot", width=1.5),
        ))
        fig_sc.add_trace(go.Scatter(
            x=zdf_sc["Tenor"], y=zdf_sc["Zero Rate (%)"],
            mode="lines+markers", name="Scenario",
            line=dict(color="#f4a261", width=2.5),
            text=_tick_text,
            hovertemplate="%{text}: %{y:.3f}%<extra></extra>",
        ))
        fig_sc.update_layout(
            xaxis=dict(title="Tenor", type="log",
                       tickvals=_tick_vals, ticktext=_tick_text),
            yaxis_title="Zero Rate (%)",
            height=300, margin=dict(t=10, b=40),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    if book.is_empty():
        st.info("Add positions to the book to see scenario P&L.")
        st.stop()

    pv_base = _client.price_book(book, curve)["pv"]
    pv_sc   = _client.price_book(book, scenario)["pv"]
    pnl     = pv_sc - pv_base

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Base Book PV",  f"${pv_base.sum():,.0f}")
    mc2.metric("Scenario PV",   f"${pv_sc.sum():,.0f}")
    mc3.metric("Scenario P&L",  f"${pnl.sum():,.0f}", delta=f"{pnl.sum():+,.0f}")

    pnl_df = pd.DataFrame({
        "Position":    [p.label for p in book.positions],
        "Base PV ($)": pv_base.values,
        "Scen PV ($)": pv_sc.values,
        "P&L ($)":     pnl.values,
    })

    fig_pnl = go.Figure(go.Bar(
        x=pnl_df["Position"], y=pnl_df["P&L ($)"],
        marker_color=["#2a9d8f" if v >= 0 else "#e63946" for v in pnl_df["P&L ($)"]],
    ))
    fig_pnl.update_layout(
        title="Scenario P&L by Position",
        xaxis_title="Position", yaxis_title="P&L ($)",
        height=320, margin=dict(t=40, b=80),
    )
    fig_pnl.update_xaxes(tickangle=-30)
    st.plotly_chart(fig_pnl, use_container_width=True)

    # Full parallel-shift sweep (batch — one FastBookEngine call per shift)
    st.subheader("Book PV vs Parallel Shift")
    sweep_bps = np.arange(-200, 205, 10)
    sweep_pvs = [
        _client.price_book(book, curve.shifted(float(b)))["pv"].sum()
        for b in sweep_bps
    ]
    base_pv_total = float(pv_base.sum())

    fig_sweep = go.Figure(go.Scatter(
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
            pnl_df.style.format({
                "Base PV ($)": "${:,.0f}",
                "Scen PV ($)": "${:,.0f}",
                "P&L ($)":     "${:+,.0f}",
            }),
            use_container_width=True, hide_index=True,
        )
