"""
QuantLib vs Fast Engine Benchmark.

Prices a small book with both engines and displays:
  • Price comparison table (absolute diff, bps of notional, relative %)
  • Error distribution by instrument type and expiry bucket
  • Speed comparison (positions/second)
  • Explanation of where and why differences arise

Run QuantLib install first:  pip install QuantLib
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
from pricer.ir.ql_engine import ql_available
from api.client import get_client

_client = get_client()

st.set_page_config(page_title="QuantLib Benchmark", layout="wide")
st.title("QuantLib vs Fast Engine — Benchmark")
st.caption(
    "Compares our vectorized numpy/Bachelier engine against QuantLib's exact formula "
    "with a full coupon schedule. Shows where and why the approximations diverge."
)

# ── QuantLib availability gate ────────────────────────────────────────────────

if not ql_available():
    st.error(
        "**QuantLib is not installed.**\n\n"
        "Run in your terminal:\n```\npip install QuantLib\n```\n"
        "Then restart Streamlit."
    )
    st.stop()

import QuantLib as ql  # noqa: E402 — only reached if available

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Benchmark Settings")
    n_pos = st.select_slider(
        "Book size",
        options=[50, 100, 200, 500, 1_000],
        value=200,
        help="QuantLib prices in a Python loop — keep this small (< 1000).",
    )
    seed = st.number_input("Seed", value=42, step=1)

    st.divider()
    st.subheader("Instrument Mix")
    p_sw  = st.slider("Swaptions (%)",  0, 100, 60, 5)
    p_cap = st.slider("Caps (%)",       0, 100, 25, 5)
    p_fl  = st.slider("Floors (%)",     0, 100, 15, 5)
    _tot = p_sw + p_cap + p_fl
    if _tot == 0:
        st.error("Must sum to > 0"); _tot = 100

    st.divider()
    run_btn = st.button("▶ Run Benchmark", use_container_width=True, type="primary")

# ── Base curve ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def _base_curve() -> RateCurve:
    try:
        return RateCurve(fetch_usd_curve())
    except Exception:
        return RateCurve({0.25: 0.043, 1.0: 0.042, 2.0: 0.041,
                          5.0: 0.042, 10.0: 0.044, 30.0: 0.048})

curve = _base_curve()


# ── Run ───────────────────────────────────────────────────────────────────────

if run_btn or "bench_result" not in st.session_state:
    # Patch instrument weights for this run
    mix_raw = np.array([p_sw / 2, p_sw / 2, p_cap, p_fl], dtype=float)
    from pricer.ir import book_generator as _bg
    _bg._INSTR_W = mix_raw / mix_raw.sum()

    with st.spinner(f"Generating {n_pos} positions…"):
        book = _client.generate_book(n=n_pos, seed=int(seed), add_hedges=False)

    # ── FastBookEngine timing
    from pricer.ir.fast_engine import FastBookEngine
    t0 = time.perf_counter()
    fast_df = FastBookEngine(curve, book).price_book()
    t_fast  = time.perf_counter() - t0

    # ── QuantLib timing
    from pricer.ir.ql_engine import QLBookEngine
    t1 = time.perf_counter()
    cmp_df = QLBookEngine(curve, book).compare(fast_df)
    t_ql   = time.perf_counter() - t1

    st.session_state["bench_result"] = {
        "cmp":    cmp_df,
        "fast":   fast_df,
        "t_fast": t_fast,
        "t_ql":   t_ql,
        "n":      n_pos,
    }

result   = st.session_state.get("bench_result")
if result is None:
    st.info("Click **▶ Run Benchmark** in the sidebar.")
    st.stop()

cmp_df  = result["cmp"]
fast_df = result["fast"]
t_fast  = result["t_fast"]
t_ql    = result["t_ql"]
n       = result["n"]


# ══════════════════════════════════════════════════════════════════════════════
# Speed comparison
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("Speed")
c1, c2, c3 = st.columns(3)
c1.metric("Fast engine",  f"{n/t_fast:,.0f} pos/s",  f"{t_fast*1000:.1f} ms total")
c2.metric("QuantLib",     f"{n/t_ql:,.0f} pos/s",    f"{t_ql*1000:.1f} ms total")
c3.metric("Speedup",      f"{t_ql/t_fast:.0f}×",     "Fast engine advantage")

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# Accuracy overview
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("Accuracy overview")

abs_diff  = cmp_df["diff"].abs()
pct_diff  = cmp_df["diff_pct"].abs().dropna()
bps_diff  = cmp_df["diff_bps"].abs()

ca, cb, cc, cd = st.columns(4)
ca.metric("Mean |Δ PV|",      f"${abs_diff.mean():,.0f}")
cb.metric("Max  |Δ PV|",      f"${abs_diff.max():,.0f}")
cc.metric("Mean |Δ %|",       f"{pct_diff.mean():.3f}%")
cd.metric("Mean |Δ bps notl|",f"{bps_diff.mean():.2f} bps")

# Error distribution histogram
fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(
    x=cmp_df["diff_pct"].dropna().values,
    nbinsx=40, marker_color="#00b4d8", opacity=0.85,
    name="PV error (%)",
))
fig_hist.add_vline(x=0, line_dash="dot", line_color="white", opacity=0.4)
fig_hist.update_layout(
    title="Distribution of PV error (QL − Fast)  in % of |PV_fast|",
    xaxis_title="Error (%)", yaxis_title="Count",
    height=280, margin=dict(t=50, b=40),
    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
)
st.plotly_chart(fig_hist, use_container_width=True)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# Breakdown by instrument
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("Error by instrument type")

cmp_aug = cmp_df.copy()
cmp_aug["instrument"] = fast_df["instrument"].values

err_by_instr = (
    cmp_aug.groupby("instrument")[["diff", "diff_bps"]]
    .agg(mean_abs_diff=("diff", lambda x: x.abs().mean()),
         mean_abs_bps=("diff_bps", lambda x: x.abs().mean()),
         max_abs_diff=("diff", lambda x: x.abs().max()))
    .rename(columns={
        "mean_abs_diff": "Mean |ΔPV| ($)",
        "mean_abs_bps":  "Mean |Δ| (bps of notl)",
        "max_abs_diff":  "Max |ΔPV| ($)",
    })
    .sort_values("Mean |ΔPV| ($)", ascending=False)
)
st.dataframe(err_by_instr.style.format("{:.2f}"), use_container_width=True)

# Bar chart
fig_instr = go.Figure()
for col, color in [("Mean |ΔPV| ($)", "#f4a261"), ("Max |ΔPV| ($)", "#e63946")]:
    fig_instr.add_trace(go.Bar(
        x=err_by_instr.index,
        y=err_by_instr[col],
        name=col, marker_color=color,
    ))
fig_instr.update_layout(
    barmode="group", title="PV error by instrument",
    xaxis_title="Instrument", yaxis_title="$ error",
    height=300, margin=dict(t=50, b=40),
    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
    legend=dict(orientation="h", y=1.12),
)
st.plotly_chart(fig_instr, use_container_width=True)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# Breakdown by expiry
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("Error by expiry bucket")

bins = [0, 1/12, 3/12, 1, 2, 5, 10, 30]
lbls = ["<1M", "1-3M", "3M-1Y", "1-2Y", "2-5Y", "5-10Y", ">10Y"]
cmp_aug["expiry_y"]   = fast_df["expiry_y"].values
cmp_aug["exp_bucket"] = pd.cut(cmp_aug["expiry_y"], bins=bins, labels=lbls)

err_by_exp = (
    cmp_aug.groupby("exp_bucket", observed=True)[["diff"]]
    .agg(mean_abs=("diff", lambda x: x.abs().mean()),
         max_abs =("diff", lambda x: x.abs().max()),
         count   =("diff", "count"))
    .rename(columns={"mean_abs": "Mean |ΔPV| ($)", "max_abs": "Max |ΔPV| ($)"})
)

fig_exp = go.Figure(go.Bar(
    x=err_by_exp.index.astype(str),
    y=err_by_exp["Mean |ΔPV| ($)"].values,
    marker_color="#2a9d8f",
))
fig_exp.update_layout(
    title="Mean absolute PV error by option expiry",
    xaxis_title="Expiry bucket", yaxis_title="Mean |ΔPV| ($)",
    height=280, margin=dict(t=50, b=40),
    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
)
st.plotly_chart(fig_exp, use_container_width=True)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# Scatter: PV fast vs QL
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("PV scatter — Fast engine vs QuantLib")

fig_sc = go.Figure()
for instr, color in [
    ("payer_swaption",    "#00b4d8"),
    ("receiver_swaption", "#f4a261"),
    ("cap",               "#06d6a0"),
    ("floor",             "#e63946"),
    ("payer_irs",         "#a8dadc"),
    ("receiver_irs",      "#f1a7c3"),
]:
    mask = cmp_aug["instrument"] == instr
    if mask.any():
        fig_sc.add_trace(go.Scatter(
            x=cmp_aug.loc[mask, "pv_fast"],
            y=cmp_aug.loc[mask, "pv_ql"],
            mode="markers",
            name=instr,
            marker=dict(color=color, size=5, opacity=0.7),
        ))

lo = min(cmp_aug["pv_fast"].min(), cmp_aug["pv_ql"].min())
hi = max(cmp_aug["pv_fast"].max(), cmp_aug["pv_ql"].max())
fig_sc.add_trace(go.Scatter(
    x=[lo, hi], y=[lo, hi],
    mode="lines", line=dict(color="white", dash="dot", width=1),
    showlegend=False, name="y = x",
))
fig_sc.update_layout(
    xaxis_title="PV — Fast engine ($)",
    yaxis_title="PV — QuantLib ($)",
    height=400, margin=dict(t=20, b=50),
    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
    legend=dict(orientation="h", y=1.05),
)
st.plotly_chart(fig_sc, use_container_width=True)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# Full comparison table
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("Full position-level comparison")

display_df = cmp_aug[["label", "instrument", "notional",
                       "pv_fast", "pv_ql", "diff", "diff_bps", "diff_pct"]].copy()
display_df = display_df.rename(columns={
    "pv_fast":  "PV Fast ($)",
    "pv_ql":    "PV QuantLib ($)",
    "diff":     "ΔPV ($)",
    "diff_bps": "Δ (bps notl)",
    "diff_pct": "Δ (%)",
})

st.dataframe(
    display_df.style.format({
        "notional":       "${:,.0f}",
        "PV Fast ($)":    "${:,.0f}",
        "PV QuantLib ($)":"${:,.0f}",
        "ΔPV ($)":        "${:,.0f}",
        "Δ (bps notl)":   "{:.2f}",
        "Δ (%)":          "{:.3f}%",
    }).background_gradient(
        subset=["Δ (bps notl)"], cmap="RdYlGn_r", axis=0,
    ),
    use_container_width=True,
    height=400,
)


# ══════════════════════════════════════════════════════════════════════════════
# Methodology note
# ══════════════════════════════════════════════════════════════════════════════

with st.expander("What drives the differences?"):
    st.markdown("""
**Swaption annuity**

| Fast engine | QuantLib benchmark |
|---|---|
| `Σ df_OIS(tᵢ) × Δtᵢ` — exact vectorized coupon sum | `Σ df(tᵢ) × Δtᵢ` — same exact sum |

Both engines compute the annuity identically. Residual error < 0.01 bps (floating-point ε).

**Cap / floor strip**

| Fast engine | QuantLib benchmark |
|---|---|
| 2-D caplet grid — all n caplets at exact reset/pay dates | All n individual caplets, each at its own reset/pay date |

Both price every caplet individually. Residual error < 0.01 bps.

**Bachelier formula itself**

Both use the exact Bachelier formula — our `scipy.special.ndtr` and QuantLib's `bachelierBlackFormula` give bit-identical results. Any error there is zero.

**Multi-curve discounting**

| Fast engine | QuantLib benchmark |
|---|---|
| Projection curve for forwards, OIS for discounting | Same dual-curve setup |

Curves are sampled from the same `CurveSet` object so the residual is pure interpolation noise (< 0.001 bps).

**What both engines ignore** (would require full QuantLib instrument setup):
- Business day conventions (Modified Following, etc.)
- Day count fraction differences (ACT/360 vs continuous)
- Settlement delays (T+2 for USD)
- Convexity adjustments for in-arrears caplets
""")
