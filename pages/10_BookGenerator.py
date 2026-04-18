"""
IR Option Book Generator.

Generate a synthetic book of 1k–400k IR option positions instantly —
no manual entry. Covers all USD + EUR indexes, caps, floors,
payer/receiver swaptions across realistic maturities and notionals.

The generated book is stored in session state and is immediately
available in:
  • 5_IROptions   — single-position view
  • 9_PortfolioRisk — Monte Carlo simulation
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from market_data.curves.rate_curve import RateCurve
from market_data.providers.fred import fetch_usd_curve
from pricer.ir.instruments import Book
from pricer.ir.book_generator import book_summary, ALL_INDEXES
from pricer.ir.indexes import INDEX_CATALOG
from api.client import get_client

_client = get_client()

st.set_page_config(page_title="Book Generator", layout="wide")
st.title("IR Option Book Generator")
st.caption("Generate realistic synthetic books of 1k–400k positions instantly. "
           "Results stored in session state for the Portfolio Risk page.")

# ═══════════════════════════════════════════════════════════════════════
# BASE CURVE
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=600, show_spinner=False)
def _base_curve() -> RateCurve:
    try:
        return RateCurve(fetch_usd_curve())
    except Exception:
        return RateCurve({0.25: 0.043, 1.0: 0.042, 2.0: 0.041,
                          5.0: 0.042, 7.0: 0.043, 10.0: 0.044, 30.0: 0.047})

curve = _base_curve()


# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR — Generator controls
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("Book Parameters")

    n_positions = st.select_slider(
        "Number of positions",
        options=[500, 1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000, 400_000],
        value=10_000,
    )

    seed = st.number_input("Random seed", value=42, step=1)

    st.divider()
    st.subheader("Instrument Mix")
    p_payer = st.slider("Payer Swaptions (%)",   0, 100, 32, 1)
    p_recvr = st.slider("Receiver Swaptions (%)", 0, 100, 28, 1)
    p_cap   = st.slider("Caps (%)",               0, 100, 22, 1)
    p_floor = st.slider("Floors (%)",             0, 100, 18, 1)
    _total  = p_payer + p_recvr + p_cap + p_floor
    if _total == 0:
        st.error("Mix must sum to > 0"); _total = 100

    st.divider()
    st.subheader("Index Mix")
    usd_pct = st.slider("USD indexes (%)", 0, 100, 60, 5)
    add_hedges = st.toggle("Add delta hedges (~15%)", value=True)

    st.divider()
    gen_btn = st.button("⚡  Generate Book", use_container_width=True, type="primary")
    clr_btn = st.button("🗑  Clear Book",    use_container_width=True)

if clr_btn:
    st.session_state.pop("book", None)
    st.session_state.pop("fast_risk", None)
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════
# GENERATION
# ═══════════════════════════════════════════════════════════════════════

if gen_btn:
    # Normalise mix weights and patch generator for this run
    mix_raw = np.array([p_payer, p_recvr, p_cap, p_floor], dtype=float)
    from pricer.ir import book_generator as _bg
    _bg._INSTR_W = mix_raw / mix_raw.sum()

    t0 = time.perf_counter()
    with st.spinner(f"Generating {n_positions:,} positions…"):
        book = _client.generate_book(
            n=n_positions,
            seed=int(seed),
            usd_weight=usd_pct / 100.0,
            add_hedges=add_hedges,
        )
    t_gen = time.perf_counter() - t0

    t1 = time.perf_counter()
    with st.spinner(f"Pricing {len(book):,} positions (vectorized)…"):
        risk, agg = _client.fast_risk(book, curve)
    t_price = time.perf_counter() - t1

    st.session_state["book"]      = book
    st.session_state["fast_risk"] = risk
    st.session_state["fast_agg"]  = agg
    st.session_state["gen_stats"] = {
        "n": len(book), "t_gen": t_gen, "t_price": t_price,
    }
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════

book = st.session_state.get("book", Book())

if book.is_empty():
    st.info("Configure the book in the sidebar and click **⚡ Generate Book**.")
    st.stop()

risk    = st.session_state.get("fast_risk", pd.DataFrame())
agg     = st.session_state.get("fast_agg",  {})
gstats  = st.session_state.get("gen_stats", {})
summ    = book_summary(book)

# ── Header metrics ────────────────────────────────────────────────────
n_pos   = gstats.get("n", len(book))
t_gen   = gstats.get("t_gen",   0)
t_price = gstats.get("t_price", 0)

st.success(
    f"**{n_pos:,} positions** generated in **{t_gen:.2f}s** · "
    f"priced in **{t_price:.2f}s** · "
    f"{n_pos / max(t_price, 0.001):,.0f} positions/second"
)

total_pv   = float(risk["pv"].sum())  if not risk.empty else 0.0
total_dv01 = float(risk["dv01"].sum()) if not risk.empty else 0.0
total_vega = float(risk["vega"].sum()) if not risk.empty else 0.0
long_n     = int((risk["direction"] > 0).sum()) if not risk.empty else 0
short_n    = int((risk["direction"] < 0).sum()) if not risk.empty else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total PV",       f"${total_pv:,.0f}")
c2.metric("Total DV01",     f"${total_dv01:,.0f}")
c3.metric("Total Vega/bp",  f"${total_vega:,.0f}")
c4.metric("Long / Short",   f"{long_n:,} / {short_n:,}")
c5.metric("Total Notional", f"${summ['total_notional']/1e9:.2f}B")

st.divider()


# ── Tabs ──────────────────────────────────────────────────────────────
tab_comp, tab_risk, tab_ladder, tab_dist, tab_data = st.tabs([
    "📊 Composition", "⚠️ Aggregate Risk",
    "📅 Maturity Ladder", "📈 Distributions", "🗒 Sample Data",
])


# ── Tab 1: Composition ────────────────────────────────────────────────
with tab_comp:
    col_a, col_b = st.columns(2)

    with col_a:
        # Instrument pie
        bi = summ["by_instrument"].reset_index()
        bi.columns = ["Instrument", "# Positions", "Notional ($)"]
        fig_pie = go.Figure(go.Pie(
            labels=bi["Instrument"],
            values=bi["Notional ($)"],
            hole=0.4,
            marker=dict(colors=["#00b4d8", "#f4a261", "#2a9d8f", "#e63946"]),
        ))
        fig_pie.update_layout(
            title="Notional by Instrument",
            height=320, margin=dict(t=40, b=0),
            paper_bgcolor="#0e1117", font_color="white",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        # CCY bar
        bc = summ["by_ccy"].reset_index()
        bc.columns = ["CCY", "# Positions", "Notional ($)"]
        fig_ccy = go.Figure(go.Bar(
            x=bc["CCY"], y=bc["Notional ($)"] / 1e9,
            marker_color=["#00b4d8", "#f4a261", "#2a9d8f"],
        ))
        fig_ccy.update_layout(
            title="Notional by Currency (Bn)",
            xaxis_title="CCY", yaxis_title="Notional (Bn)",
            height=320, margin=dict(t=40, b=40),
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white",
        )
        st.plotly_chart(fig_ccy, use_container_width=True)

    # Index breakdown bar
    bi2 = summ["by_index"].reset_index()
    bi2.columns = ["Index", "# Positions", "Notional ($)"]
    bi2["Label"] = bi2["Index"].map(lambda k: INDEX_CATALOG.get(k, {}).get("label", k))
    fig_idx = go.Figure(go.Bar(
        x=bi2["Label"], y=bi2["Notional ($)"] / 1e9,
        marker_color="#00b4d8",
    ))
    fig_idx.update_layout(
        title="Notional by Index (Bn)",
        xaxis_title="Index", yaxis_title="Notional (Bn)",
        height=340, margin=dict(t=40, b=80),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white",
    )
    fig_idx.update_xaxes(tickangle=-35)
    st.plotly_chart(fig_idx, use_container_width=True)


# ── Tab 2: Aggregate Risk ─────────────────────────────────────────────
with tab_risk:
    if not agg:
        st.info("No risk data.")
    else:
        for label, df_agg in [
            ("By Instrument",    agg.get("by_instrument", pd.DataFrame())),
            ("By Index",         agg.get("by_index",      pd.DataFrame())),
            ("By Currency",      agg.get("by_ccy",        pd.DataFrame())),
        ]:
            if df_agg.empty:
                continue
            st.subheader(label)
            col1, col2 = st.columns([1.5, 2])

            with col1:
                disp = df_agg.copy()
                for c in ["PV ($)", "DV01 ($)", "Vega/bp ($)"]:
                    if c in disp.columns:
                        disp[c] = disp[c].map(lambda x: f"${x:,.0f}")
                st.dataframe(disp, use_container_width=True)

            with col2:
                # Stacked bar: PV + DV01 + Vega
                df_plot = agg.get(label.split()[-1].lower(), df_agg).reset_index()
                x_col = df_plot.columns[0]
                idx_labels = df_plot[x_col].map(
                    lambda k: INDEX_CATALOG.get(k, {}).get("label", k)
                    if label == "By Index" else k
                )
                fig_r = make_subplots(rows=1, cols=3,
                                      subplot_titles=["PV ($)", "DV01 ($)", "Vega/bp ($)"])
                for i_c, col_name in enumerate(["PV ($)", "DV01 ($)", "Vega/bp ($)"], start=1):
                    if col_name in df_agg.columns:
                        vals = df_agg[col_name].values
                        fig_r.add_trace(
                            go.Bar(x=idx_labels, y=vals,
                                   marker_color=["#2a9d8f" if v >= 0 else "#e63946" for v in vals],
                                   showlegend=False),
                            row=1, col=i_c,
                        )
                fig_r.update_layout(
                    height=300, margin=dict(t=40, b=60),
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white",
                )
                fig_r.update_xaxes(tickangle=-30)
                st.plotly_chart(fig_r, use_container_width=True)

            st.divider()


# ── Tab 3: Maturity Ladder ────────────────────────────────────────────
with tab_ladder:
    if not risk.empty:
        bins = [0, 1/12, 3/12, 1, 2, 5, 10, 30]
        lbls = ["<1M", "1-3M", "3M-1Y", "1-2Y", "2-5Y", "5-10Y", ">10Y"]
        risk["exp_bucket"] = pd.cut(risk["expiry_y"], bins=bins, labels=lbls)

        ladder = (
            risk.groupby(["exp_bucket", "instrument"], observed=True)[["pv", "dv01", "vega"]]
            .sum()
            .reset_index()
        )

        inst_colors = {
            "payer_swaption":    "#00b4d8",
            "receiver_swaption": "#f4a261",
            "cap":               "#2a9d8f",
            "floor":             "#e63946",
        }

        for metric, ylabel in [("pv", "PV ($)"), ("dv01", "DV01 ($)"), ("vega", "Vega/bp ($)")]:
            fig_lad = go.Figure()
            for instr in ladder["instrument"].unique():
                sub = ladder[ladder["instrument"] == instr]
                fig_lad.add_trace(go.Bar(
                    x=sub["exp_bucket"].astype(str),
                    y=sub[metric],
                    name=instr,
                    marker_color=inst_colors.get(instr, "#aaa"),
                ))
            fig_lad.update_layout(
                title=f"Maturity Ladder — {ylabel}",
                barmode="relative",
                xaxis_title="Expiry Bucket", yaxis_title=ylabel,
                height=340, margin=dict(t=40, b=40),
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white",
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig_lad, use_container_width=True)


# ── Tab 4: Distributions ──────────────────────────────────────────────
with tab_dist:
    if not risk.empty:
        col_p, col_q = st.columns(2)

        with col_p:
            fig_pv = go.Figure(go.Histogram(
                x=risk["pv"].clip(-5e6, 5e6), nbinsx=60,
                marker_color="#00b4d8", opacity=0.8,
            ))
            fig_pv.add_vline(x=float(risk["pv"].mean()),
                             line_dash="dash", line_color="#f4a261",
                             annotation_text="Mean PV")
            fig_pv.update_layout(
                title="PV Distribution per Position",
                xaxis_title="PV ($)", yaxis_title="Count",
                height=300, margin=dict(t=40, b=40),
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white",
            )
            st.plotly_chart(fig_pv, use_container_width=True)

        with col_q:
            fig_dv = go.Figure(go.Histogram(
                x=risk["dv01"].clip(-50_000, 50_000), nbinsx=60,
                marker_color="#2a9d8f", opacity=0.8,
            ))
            fig_dv.update_layout(
                title="DV01 Distribution per Position",
                xaxis_title="DV01 ($)", yaxis_title="Count",
                height=300, margin=dict(t=40, b=40),
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white",
            )
            st.plotly_chart(fig_dv, use_container_width=True)

        # Notional distribution
        fig_not = go.Figure(go.Histogram(
            x=np.log10(risk["notional"].clip(1e5, None)),
            nbinsx=40, marker_color="#f4a261", opacity=0.8,
        ))
        fig_not.update_layout(
            title="Notional Distribution (log₁₀ scale)",
            xaxis=dict(
                title="log₁₀(Notional)",
                tickvals=[6, 7, 8, 8.5],
                ticktext=["$1M", "$10M", "$100M", "$316M"],
            ),
            yaxis_title="Count",
            height=280, margin=dict(t=40, b=40),
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white",
        )
        st.plotly_chart(fig_not, use_container_width=True)


# ── Tab 5: Sample Data ────────────────────────────────────────────────
with tab_data:
    st.caption(f"Showing first 500 of {len(book):,} positions.")
    sample = risk.head(500).copy()
    for col in ["pv", "dv01", "vega"]:
        sample[col] = sample[col].round(0)
    sample["strike_pct"] = (risk.head(500)["expiry_y"] * 0 + 1)  # placeholder
    sample["notional"] = sample["notional"].map(lambda x: f"${x/1e6:.1f}M")

    st.dataframe(
        sample[["label", "instrument", "index_key", "ccy",
                "expiry_y", "tenor_y", "notional", "direction",
                "pv", "dv01", "vega"]].rename(columns={
            "label": "Position", "instrument": "Type", "index_key": "Index",
            "ccy": "CCY", "expiry_y": "Exp (Y)", "tenor_y": "Tenor (Y)",
            "notional": "Notional", "direction": "Dir",
            "pv": "PV ($)", "dv01": "DV01 ($)", "vega": "Vega/bp ($)",
        }),
        use_container_width=True, hide_index=True,
    )

    st.divider()
    st.page_link("pages/9_PortfolioRisk.py",
                 label="→ Go to Portfolio Risk & Monte Carlo",
                 icon="🎲")
