"""
Portfolio Risk — Intraday Live Greeks Monitor.

Smooth animation via @st.fragment + st.rerun(scope="fragment"):
  • Main script runs once (sidebar, book filter, path generation).
  • The live_monitor() fragment reruns independently — only the charts
    inside it update; the sidebar and static content never flicker.

Per step: 4 × bump-and-reprice via API client:
  base / +dv01_bp / +gamma_bp / −gamma_bp
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_data.providers.fred import fetch_usd_curve
from market_data.curves.rate_curve import RateCurve
from pricer.ir.instruments import Book
from pricer.ir.indexes import INDEX_CATALOG
from api.client import get_client

_client = get_client()

st.set_page_config(page_title="Portfolio Risk", layout="wide")
st.title("Portfolio Risk — Intraday Greeks Monitor")

# ══════════════════════════════════════════════════════════════════════════════
# Book
# ══════════════════════════════════════════════════════════════════════════════

full_book: Book = st.session_state.get("book", Book())

if full_book.is_empty():
    st.info("No book loaded. Use **10 Book Generator** or **5 IR Options**.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar  (main script — runs on full rerun only)
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("Book Filter")
    all_indexes = sorted({p.index_key  for p in full_book.positions})
    all_instrs  = sorted({p.instrument for p in full_book.positions})
    all_ccys    = sorted({INDEX_CATALOG[p.index_key]["ccy"] for p in full_book.positions})

    sel_ccys  = st.multiselect("CCY",        all_ccys,    default=all_ccys)
    sel_idx   = st.multiselect("Index",      all_indexes, default=all_indexes)
    sel_instr = st.multiselect("Instrument", all_instrs,  default=all_instrs)

    st.divider()
    st.header("Simulation")
    open_h   = st.slider("Open  (h)",  7,  10, 9)
    close_h  = st.slider("Close (h)", 15,  18, 17)
    n_steps  = st.slider("Steps",     10, 120, 48, 4)
    vol_bps  = st.slider("Intraday vol (bps/√day)", 5, 80, 25, 5)
    kappa    = st.slider("Mean reversion κ", 0.0, 5.0, 0.5, 0.1)
    seed     = st.number_input("Seed", value=42, step=1)

    st.divider()
    st.header("Bumps")
    dv01_bp  = int(st.number_input("DV01 bump (bp)",  value=1, step=1, min_value=1))
    gamma_bp = int(st.number_input("Γ bump (bp)",     value=5, step=1, min_value=1))
    frame_ms = st.slider("Frame delay (ms)", 100, 2000, 600, 100)

    st.divider()
    run_btn   = st.button("▶ Run",   use_container_width=True, type="primary")
    reset_btn = st.button("↩ Reset", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Base curve  (main script)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=600, show_spinner=False)
def _base_curve() -> RateCurve:
    try:
        return RateCurve(fetch_usd_curve())
    except Exception:
        return RateCurve({0.25: 0.043, 1.0: 0.042, 2.0: 0.041,
                          5.0: 0.042, 10.0: 0.044, 30.0: 0.048})

base_curve = _base_curve()

# ══════════════════════════════════════════════════════════════════════════════
# Path generation helpers  (main script)
# ══════════════════════════════════════════════════════════════════════════════

def _filter_book(book: Book) -> Book:
    return Book(positions=[
        p for p in book.positions
        if (INDEX_CATALOG[p.index_key]["ccy"] in sel_ccys
            and p.index_key in sel_idx
            and p.instrument in sel_instr)
    ])


def _gen_path(n_steps, open_h, close_h, vol_bps, kappa, seed) -> tuple[np.ndarray, list[str]]:
    horizon_h  = close_h - open_h
    dt_frac    = (horizon_h / n_steps) / (252 * 8)
    sigma_step = vol_bps * np.sqrt(dt_frac * 252)
    rng   = np.random.default_rng(seed)
    r     = np.zeros(n_steps + 1)
    alpha = kappa * (horizon_h / n_steps) / horizon_h
    for s in range(n_steps):
        r[s + 1] = r[s] * (1 - alpha) + sigma_step * rng.standard_normal()
    total_min = horizon_h * 60
    mins  = [int(open_h * 60 + i * total_min / n_steps) for i in range(n_steps + 1)]
    lbls  = [f"{m // 60:02d}:{m % 60:02d}" for m in mins]
    return r, lbls

# ══════════════════════════════════════════════════════════════════════════════
# Main-script state management  (full rerun only)
# ══════════════════════════════════════════════════════════════════════════════

_KEYS = ("pr_path", "pr_labels", "pr_step", "pr_playing",
         "pr_history", "pr_df", "pr_pv_open",
         "pr_dv01_bp", "pr_gamma_bp", "pr_frame_ms", "pr_n_steps")

if reset_btn:
    for k in _KEYS:
        st.session_state.pop(k, None)
    st.rerun()

if run_btn:
    fbook = _filter_book(full_book)
    if fbook.is_empty():
        st.warning("No positions match the current filter.")
        st.stop()

    for k in _KEYS:
        st.session_state.pop(k, None)

    r, lbls = _gen_path(n_steps, open_h, close_h, vol_bps, kappa, int(seed))
    # Store everything the fragment needs in session state
    st.session_state.update({
        "pr_path":     r,
        "pr_labels":   lbls,
        "pr_step":     0,
        "pr_history":  [],
        "pr_playing":  False,
        "pr_book":     fbook,
        "pr_dv01_bp":  dv01_bp,
        "pr_gamma_bp": gamma_bp,
        "pr_frame_ms": frame_ms,
        "pr_n_steps":  n_steps,
    })
    st.rerun()

if "pr_path" not in st.session_state:
    st.info("Configure the simulation in the sidebar and click **▶ Run**.")
    st.stop()

n_pos = len(st.session_state["pr_book"].positions)
st.caption(
    f"Book: **{len(full_book.positions):,}** total · "
    f"**{n_pos:,}** in filter · "
    f"**{n_steps}** steps"
)

# ══════════════════════════════════════════════════════════════════════════════
# LIVE MONITOR FRAGMENT
# Only this fragment reruns during animation — sidebar never flickers.
# ══════════════════════════════════════════════════════════════════════════════

_EXP_BINS = [0, 1/12, 3/12, 1, 2, 5, 10, 30]
_EXP_LBLS = ["<1M", "1-3M", "3M-1Y", "1-2Y", "2-5Y", "5-10Y", ">10Y"]


@st.fragment
def live_monitor():
    # ── Read state ────────────────────────────────────────────────────────
    shifts   = st.session_state["pr_path"]
    labels   = st.session_state["pr_labels"]
    step     = st.session_state["pr_step"]
    history  = st.session_state["pr_history"]
    playing  = st.session_state.get("pr_playing", False)
    fbook    = st.session_state["pr_book"]
    _dv01_bp = st.session_state["pr_dv01_bp"]
    _gam_bp  = st.session_state["pr_gamma_bp"]
    _fms     = st.session_state["pr_frame_ms"]
    _nsteps  = st.session_state["pr_n_steps"]

    # ── Fragment controls (Play/Stop/Step live here so they don't full-rerun) ─
    col_p, col_x, col_s = st.columns([1, 1, 1])
    if col_p.button("⏵ Play",  use_container_width=True,
                    disabled=playing or step > _nsteps):
        st.session_state["pr_playing"] = True
        playing = True
    if col_x.button("⏹ Stop",  use_container_width=True, disabled=not playing):
        st.session_state["pr_playing"] = False
        playing = False
    if col_s.button("⏭ Step",  use_container_width=True,
                    disabled=playing or step > _nsteps):
        # advance exactly one step then stop
        _advance(fbook, base_curve, shifts, labels, step,
                 history, _dv01_bp, _gam_bp)
        st.rerun(scope="fragment")
        return

    # ── Advance one step if playing ───────────────────────────────────────
    if playing and step <= _nsteps:
        _advance(fbook, base_curve, shifts, labels, step,
                 history, _dv01_bp, _gam_bp)
        if st.session_state["pr_step"] > _nsteps:
            st.session_state["pr_playing"] = False
        # Redraw immediately, then schedule next frame
        _render(shifts, labels, history, _dv01_bp, _gam_bp, _nsteps)
        time.sleep(_fms / 1000.0)
        st.rerun(scope="fragment")
        return

    # ── Static render (not playing) ───────────────────────────────────────
    if history:
        _render(shifts, labels, history, _dv01_bp, _gam_bp, _nsteps)
    else:
        st.info("Press **⏵ Play** or **⏭ Step** to begin.")


# ── Step logic (separated so both Play and Step button can call it) ────────────

def _advance(fbook, curve, shifts, labels, step, history, dv01_bp, gam_bp):
    shift_now = float(shifts[step])
    with st.spinner(f"⏱ {labels[step]}  Δr = {shift_now:+.1f} bps"):
        df = _client.risk_book(fbook, curve, shift_bp=shift_now,
                               dv01_bp=dv01_bp, gamma_bp=gam_bp)
    df["exp_bucket"] = pd.cut(df["expiry_y"], bins=_EXP_BINS, labels=_EXP_LBLS)

    pv  = float(df["pv"].sum())
    dv  = float(df["dv01"].sum())
    gup = float(df["gamma_up"].sum())
    gdn = float(df["gamma_dn"].sum())

    if step == 0:
        st.session_state["pr_pv_open"] = pv

    history.append({"step": step, "time": labels[step],
                    "shift_bp": shift_now,
                    "pv": pv, "dv01": dv,
                    "gamma_up": gup, "gamma_dn": gdn})
    st.session_state["pr_history"] = history
    st.session_state["pr_df"]      = df
    st.session_state["pr_step"]    = step + 1


# ── Render (pure display, no state mutation) ──────────────────────────────────

def _render(shifts, labels, history, dv01_bp, gam_bp, n_steps):
    hist      = pd.DataFrame(history)
    last      = hist.iloc[-1]
    df_pos    = st.session_state.get("pr_df", pd.DataFrame())
    pv_open   = st.session_state.get("pr_pv_open", last["pv"])
    pnl_day   = last["pv"] - pv_open
    cur_shift = float(last["shift_bp"])
    cur_step  = int(last["step"])

    # ── Headline ───────────────────────────────────────────────────────
    st.subheader(
        f"🕐 {last['time']}  ·  step {cur_step}/{n_steps}  ·  "
        f"Δr = {cur_shift:+.1f} bps"
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Book PV",               f"${last['pv']:,.0f}")
    c2.metric("P&L since open",        f"${pnl_day:,.0f}",
              delta=f"{pnl_day / abs(pv_open) * 100:.2f}%" if pv_open else None)
    c3.metric(f"DV01 (+{dv01_bp}bp)",  f"${last['dv01']:,.0f}")
    c4.metric(f"Γ+ (+{gam_bp}bp)",     f"${last['gamma_up']:,.0f}")
    c5.metric(f"Γ− (−{gam_bp}bp)",     f"${last['gamma_dn']:,.0f}")

    # ── Live yield curve + rate path ───────────────────────────────────
    col_curve, col_path = st.columns(2)

    with col_curve:
        c_now  = base_curve.shifted(cur_shift)
        T_plot = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])
        z_base = np.interp(T_plot, base_curve._tenors, base_curve._zero_rates) * 100
        z_now  = np.interp(T_plot, c_now._tenors,      c_now._zero_rates)      * 100
        fig_cv = go.Figure()
        fig_cv.add_trace(go.Scatter(
            x=T_plot, y=z_base, mode="lines", name="Open",
            line=dict(color="#555", width=1.8, dash="dot"),
        ))
        fig_cv.add_trace(go.Scatter(
            x=T_plot, y=z_now, mode="lines+markers",
            name=f"Now ({cur_shift:+.1f}bp)",
            line=dict(color="#00b4d8", width=2.5), marker=dict(size=5),
            fill="tonexty", fillcolor="rgba(0,180,216,0.08)",
        ))
        fig_cv.update_layout(
            title="Live Yield Curve",
            xaxis_title="Tenor (Y)", yaxis_title="Zero rate (%)",
            height=260, margin=dict(t=40, b=30),
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_cv, use_container_width=True, key="cv")

    with col_path:
        future = list(range(cur_step + 1, n_steps + 1))
        fig_r  = go.Figure()
        if future:
            fig_r.add_trace(go.Scatter(
                x=[labels[i] for i in future], y=shifts[future],
                mode="lines", line=dict(color="#333", width=1, dash="dot"),
                showlegend=False,
            ))
        fig_r.add_trace(go.Scatter(
            x=hist["time"], y=hist["shift_bp"],
            mode="lines+markers", line=dict(color="#00b4d8", width=2.2),
            marker=dict(size=5, color=hist["shift_bp"].values,
                        colorscale="RdBu_r", cmin=-60, cmax=60, showscale=False),
            showlegend=False,
        ))
        fig_r.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.3,
                        annotation_text="open", annotation_position="right")
        fig_r.update_layout(
            title="Intraday Rate Path (bps vs open)",
            xaxis_title="Time", yaxis_title="Δr (bps)",
            height=260, margin=dict(t=40, b=30),
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
        )
        st.plotly_chart(fig_r, use_container_width=True, key="rp")

    # ── P&L + Greeks evolution ──────────────────────────────────────────
    col_pnl, col_gk = st.columns(2)
    with col_pnl:
        pnl_s = hist["pv"] - pv_open
        fig_pnl = go.Figure(go.Scatter(
            x=hist["time"], y=pnl_s,
            mode="lines+markers", line=dict(color="#f4a261", width=2.2),
            marker=dict(size=4),
            fill="tozeroy", fillcolor="rgba(244,162,97,0.12)",
        ))
        fig_pnl.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.3)
        fig_pnl.update_layout(
            title="P&L since open ($)",
            xaxis_title="Time", yaxis_title="$",
            height=210, margin=dict(t=40, b=30),
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
        )
        st.plotly_chart(fig_pnl, use_container_width=True, key="pnl")

    with col_gk:
        fig_gk = go.Figure()
        fig_gk.add_trace(go.Scatter(
            x=hist["time"], y=hist["dv01"],
            mode="lines", name=f"DV01 (+{dv01_bp}bp)",
            line=dict(color="#8338ec", width=2),
        ))
        fig_gk.add_trace(go.Scatter(
            x=hist["time"], y=hist["gamma_up"],
            mode="lines", name=f"Γ+ (+{gam_bp}bp)",
            line=dict(color="#06d6a0", width=1.6, dash="dash"),
        ))
        fig_gk.add_trace(go.Scatter(
            x=hist["time"], y=hist["gamma_dn"],
            mode="lines", name=f"Γ− (−{gam_bp}bp)",
            line=dict(color="#e63946", width=1.6, dash="dash"),
        ))
        fig_gk.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.3)
        fig_gk.update_layout(
            title="Greeks evolution",
            xaxis_title="Time", yaxis_title="$",
            height=210, margin=dict(t=40, b=30),
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
            legend=dict(orientation="h", y=1.12, font=dict(size=10)),
        )
        st.plotly_chart(fig_gk, use_container_width=True, key="gk")

    if df_pos.empty:
        return

    st.divider()

    # ── Greeks heatmaps ─────────────────────────────────────────────────
    def _heatmap(col, title):
        piv = (
            df_pos.groupby(["exp_bucket", "index_key"], observed=True)[col]
            .sum().unstack(fill_value=0.0)
        )
        text = [[f"${v:,.0f}" for v in row] for row in piv.values]
        fig  = go.Figure(go.Heatmap(
            z=piv.values, x=list(piv.columns),
            y=[str(r) for r in piv.index],
            text=text, texttemplate="%{text}",
            colorscale="RdYlGn", zmid=0,
            colorbar=dict(title="$", len=0.8),
        ))
        fig.update_layout(
            title=title,
            xaxis=dict(tickangle=-30), yaxis_title="Expiry",
            height=300, margin=dict(t=50, b=40, l=80),
            paper_bgcolor="#0e1117", font_color="white",
        )
        return fig

    def _bar(col, title):
        tots = df_pos.groupby("exp_bucket", observed=True)[col].sum()
        fig  = go.Figure(go.Bar(
            x=tots.index.astype(str), y=tots.values,
            marker_color=["#2a9d8f" if v >= 0 else "#e63946" for v in tots.values],
        ))
        fig.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.3)
        fig.update_layout(
            title=title, xaxis_title="Expiry", yaxis_title="$",
            height=230, margin=dict(t=40, b=30),
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
        )
        return fig

    tab_dv, tab_gup, tab_gdn, tab_pos = st.tabs([
        f"DV01 (+{dv01_bp}bp)",
        f"Γ+ (+{gam_bp}bp)",
        f"Γ− (−{gam_bp}bp)",
        "Positions",
    ])

    with tab_dv:
        st.plotly_chart(
            _heatmap("dv01", f"DV01 by expiry × index  (+{dv01_bp}bp)"),
            use_container_width=True, key="h_dv",
        )
        st.plotly_chart(
            _bar("dv01", "DV01 ladder"), use_container_width=True, key="b_dv",
        )

    with tab_gup:
        st.caption(f"P&L if rates +{gam_bp}bp · Total: **${df_pos['gamma_up'].sum():,.0f}**")
        st.plotly_chart(
            _heatmap("gamma_up", f"Γ+ by expiry × index  (+{gam_bp}bp)"),
            use_container_width=True, key="h_gup",
        )
        st.plotly_chart(
            _bar("gamma_up", f"Γ+ by expiry"), use_container_width=True, key="b_gup",
        )

    with tab_gdn:
        st.caption(f"P&L if rates −{gam_bp}bp · Total: **${df_pos['gamma_dn'].sum():,.0f}**")
        st.plotly_chart(
            _heatmap("gamma_dn", f"Γ− by expiry × index  (−{gam_bp}bp)"),
            use_container_width=True, key="h_gdn",
        )
        gdn_tots = df_pos.groupby("exp_bucket", observed=True)["gamma_dn"].sum()
        gup_tots = df_pos.groupby("exp_bucket", observed=True)["gamma_up"].sum()
        st.plotly_chart(
            _bar("gamma_dn", f"Γ− by expiry"), use_container_width=True, key="b_gdn",
        )
        asym = (gup_tots + gdn_tots).reindex(gup_tots.index, fill_value=0)
        fig_asym = go.Figure(go.Bar(
            x=asym.index.astype(str), y=asym.values,
            marker_color=["#8338ec" if v >= 0 else "#ffb703" for v in asym.values],
        ))
        fig_asym.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.4)
        fig_asym.update_layout(
            title="Convexity asymmetry Γ+ + Γ−  (purple = long gamma)",
            xaxis_title="Expiry", yaxis_title="$",
            height=220, margin=dict(t=40, b=30),
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
        )
        st.plotly_chart(fig_asym, use_container_width=True, key="asym")

    with tab_pos:
        rename = {
            "label": "Position", "instrument": "Instrument",
            "index_key": "Index", "exp_bucket": "Bucket",
            "expiry_y": "Expiry (Y)", "tenor_y": "Tenor (Y)",
            "notional": "Notional", "direction": "Dir",
            "pv": "PV ($)",
            "dv01": f"DV01 (+{dv01_bp}bp)",
            "gamma_up": f"Γ+ (+{gam_bp}bp)",
            "gamma_dn": f"Γ− (−{gam_bp}bp)",
        }
        disp = df_pos[list(rename)].rename(columns=rename)
        sort_col = st.selectbox("Sort by", list(rename.values())[8:] + ["Notional"],
                                key="sort_pos")
        disp = disp.sort_values(sort_col, key=abs, ascending=False)
        st.dataframe(
            disp.style.format({
                "Notional":              "${:,.0f}",
                "PV ($)":               "${:,.0f}",
                f"DV01 (+{dv01_bp}bp)": "${:,.0f}",
                f"Γ+ (+{gam_bp}bp)":    "${:,.0f}",
                f"Γ− (−{gam_bp}bp)":    "${:,.0f}",
                "Expiry (Y)":           "{:.2f}",
                "Tenor (Y)":            "{:.1f}",
            }),
            use_container_width=True, hide_index=True,
        )


# ── Entry point ───────────────────────────────────────────────────────────────
live_monitor()
