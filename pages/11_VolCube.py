"""
IR Vol Cube — synthetic ATM normal volatility surfaces.

Displays:
  Tab 1 — Swaption ATM surface (USD / EUR / GBP), 3-D and heatmap.
  Tab 2 — Cap/Floor ATM curves per index, grouped by CCY.
  Tab 3 — Smile slices (ZABR) for any (index, expiry).
  Tab 4 — Cross-index comparison at a fixed maturity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from pricer.ir.vol_cube import EXPIRY_GRID, TENOR_GRID, CAPMAT_GRID
from pricer.ir.indexes import INDEX_CATALOG
from pricer.ir.zabr import smile_vol_strip
from api.client import get_client

_client = get_client()

st.set_page_config(page_title="IR Vol Cube", layout="wide")
st.title("IR Vol Cube — Synthetic Normal Volatility")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Cube Settings")
    seed = st.number_input("Market scenario (seed)", value=42, step=1,
                           help="Different seeds → different synthetic market scenarios")
    regen = st.button("Regenerate cube", use_container_width=True)

    st.divider()
    st.header("Swaption Surface")
    sw_ccy = st.selectbox("CCY", ["USD", "EUR", "GBP"])

    st.divider()
    st.header("Smile Slice")
    all_indexes = list(INDEX_CATALOG.keys())
    sm_index = st.selectbox("Index", all_indexes,
                             index=all_indexes.index("TERM_SOFR_3M"))
    sm_expiry = st.select_slider(
        "Expiry (Y)",
        options=[f"{x*12:.0f}M" if x < 1 else f"{x:.0f}Y" for x in EXPIRY_GRID],
        value="2Y",
    )
    sm_expiry_y = EXPIRY_GRID[
        [f"{x*12:.0f}M" if x < 1 else f"{x:.0f}Y" for x in EXPIRY_GRID].index(sm_expiry)
    ]
    sm_product = st.radio("Product", ["Swaption", "Cap/Floor"])
    sm_tenor = st.select_slider("Swap tenor (Y)", options=[int(t) for t in TENOR_GRID],
                                 value=5,
                                 disabled=sm_product == "Cap/Floor")

# ── Cube (cached by seed) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Building vol cube…")
def _cube(s: int):
    return _client.vol_cube(seed=s)

if regen:
    st.cache_resource.clear()

cube = _cube(int(seed))

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Swaption Surface", "Cap/Floor Curves", "Smile Slices", "Index Comparison"
])


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Swaption ATM Surface
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    surf_df = cube.swaption_surface_df(sw_ccy)

    col_a, col_b = st.columns([1.6, 1])

    with col_a:
        # 3-D surface
        Z = surf_df.values   # bps
        fig3d = go.Figure(go.Surface(
            z=Z,
            x=[f"{t:.0f}Y" for t in TENOR_GRID],
            y=surf_df.index.tolist(),
            colorscale="Plasma",
            colorbar=dict(title="σ_N (bps)", len=0.6),
            contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)),
        ))
        fig3d.update_layout(
            title=f"{sw_ccy} Swaption ATM Normal Vol Surface",
            scene=dict(
                xaxis_title="Swap Tenor",
                yaxis_title="Expiry",
                zaxis_title="σ_N (bps)",
                bgcolor="#0e1117",
            ),
            paper_bgcolor="#0e1117", font_color="white",
            height=500, margin=dict(t=50, b=10),
        )
        st.plotly_chart(fig3d, use_container_width=True)

    with col_b:
        st.subheader(f"{sw_ccy} ATM Normal Vol (bps)")
        st.dataframe(
            surf_df.style.background_gradient(cmap="plasma", axis=None).format("{:.1f}"),
            use_container_width=True,
        )

    # Heatmap
    fig_hm = go.Figure(go.Heatmap(
        z=Z,
        x=[f"{t:.0f}Y" for t in TENOR_GRID],
        y=surf_df.index.tolist(),
        colorscale="Plasma",
        colorbar=dict(title="σ_N (bps)"),
        text=np.round(Z, 1),
        texttemplate="%{text}",
        textfont=dict(size=10),
    ))
    fig_hm.update_layout(
        title=f"{sw_ccy} Swaption ATM Vol Heatmap",
        xaxis_title="Swap Tenor", yaxis_title="Expiry",
        height=380, margin=dict(t=50, b=40),
        paper_bgcolor="#0e1117", font_color="white",
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # Expiry term structure slices
    st.subheader("Term structure by swap tenor")
    fig_ts = go.Figure()
    colors = ["#f4a261", "#2a9d8f", "#e63946", "#8338ec", "#00b4d8", "#06d6a0"]
    sel_tenors = [1, 2, 5, 10, 20, 30]
    for i, ten in enumerate(sel_tenors):
        col_lbl = f"{ten:.0f}Y"
        if col_lbl in surf_df.columns:
            fig_ts.add_trace(go.Scatter(
                x=surf_df.index.tolist(),
                y=surf_df[col_lbl].values,
                mode="lines+markers",
                name=f"{ten}Y swap",
                line=dict(color=colors[i % len(colors)], width=2),
            ))
    fig_ts.update_layout(
        xaxis_title="Expiry", yaxis_title="σ_N (bps)",
        height=320, margin=dict(t=10, b=40),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig_ts, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Cap/Floor ATM Curves
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    cf_df = cube.capfloor_surface_df()  # rows=maturities, cols=indexes (bps)

    # Group by CCY
    usd_cols = [k for k in cf_df.columns if INDEX_CATALOG[k]["ccy"] == "USD"]
    eur_cols = [k for k in cf_df.columns if INDEX_CATALOG[k]["ccy"] == "EUR"]
    gbp_cols = [k for k in cf_df.columns if INDEX_CATALOG[k]["ccy"] == "GBP"]

    def _cf_plot(cols, title):
        fig = go.Figure()
        pal = ["#f4a261", "#2a9d8f", "#e63946", "#8338ec", "#00b4d8", "#06d6a0"]
        for i, col in enumerate(cols):
            label = INDEX_CATALOG[col]["label"]
            fig.add_trace(go.Scatter(
                x=cf_df.index.tolist(),
                y=cf_df[col].values,
                mode="lines+markers",
                name=label,
                line=dict(color=pal[i % len(pal)], width=2.2),
                marker=dict(size=6),
            ))
        fig.update_layout(
            title=title, xaxis_title="Cap Maturity",
            yaxis_title="ATM Normal Vol (bps)",
            height=340, margin=dict(t=50, b=40),
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
            legend=dict(orientation="h", y=1.12),
        )
        return fig

    st.plotly_chart(_cf_plot(usd_cols, "USD Cap/Floor ATM Vols by Index"), use_container_width=True)
    st.plotly_chart(_cf_plot(eur_cols, "EUR Cap/Floor ATM Vols by Index"), use_container_width=True)
    st.plotly_chart(_cf_plot(gbp_cols, "GBP Cap/Floor ATM Vols by Index"), use_container_width=True)

    st.subheader("Cap/Floor ATM Vol Table (bps)")
    st.dataframe(
        cf_df.style.background_gradient(cmap="YlOrRd", axis=None).format("{:.1f}"),
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Smile Slices
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    is_cf = sm_product == "Cap/Floor"
    alpha, nu, rho = cube.smile_params(
        sm_index, sm_expiry_y, tenor_y=float(sm_tenor), is_capfloor=is_cf
    )
    ccy  = INDEX_CATALOG[sm_index]["ccy"]
    F_atm = {"USD": 0.0450, "EUR": 0.0235, "GBP": 0.0455}.get(ccy, 0.04)

    # Strike grid: ±3 sigma around ATM
    sigma_move = alpha * np.sqrt(sm_expiry_y)
    K_grid = np.linspace(
        max(F_atm - 3 * sigma_move, 0.001),
        F_atm + 3 * sigma_move,
        120,
    )
    vols_bps = smile_vol_strip(F_atm, K_grid, sm_expiry_y, alpha, nu, rho) * 10_000

    col1, col2 = st.columns([2, 1])
    with col1:
        fig_sm = go.Figure()
        fig_sm.add_trace(go.Scatter(
            x=K_grid * 100, y=vols_bps,
            mode="lines", name="ZABR smile",
            line=dict(color="#f4a261", width=2.5),
        ))
        fig_sm.add_vline(x=F_atm * 100, line_dash="dash", line_color="white",
                         opacity=0.5, annotation_text="ATM")
        fig_sm.update_layout(
            title=f"ZABR Smile — {INDEX_CATALOG[sm_index]['label']} · T={sm_expiry_y:.2f}Y · {sm_product}",
            xaxis_title="Strike (%)", yaxis_title="Normal Vol (bps)",
            height=380, margin=dict(t=50, b=40),
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
        )
        st.plotly_chart(fig_sm, use_container_width=True)

    with col2:
        st.subheader("ZABR Parameters")
        st.metric("α  (ATM vol)", f"{alpha*10_000:.1f} bps")
        st.metric("ν  (vol-of-vol)", f"{nu:.3f}")
        st.metric("ρ  (skew corr.)", f"{rho:.3f}")
        st.metric("F  (ATM fwd)", f"{F_atm*100:.2f}%")

    # Multi-expiry smile overlay
    st.subheader("Smile across all expiries")
    fig_multi = go.Figure()
    pal = ["#f4a261", "#2a9d8f", "#e63946", "#8338ec", "#00b4d8", "#06d6a0",
           "#ffb703", "#fb8500", "#023047", "#219ebc", "#8ecae6"]
    for i, T_exp in enumerate(EXPIRY_GRID):
        a_i, nu_i, rho_i = cube.smile_params(sm_index, T_exp, is_capfloor=is_cf)
        sm_i = alpha * np.sqrt(T_exp)
        K_i  = np.linspace(max(F_atm - 3 * sm_i, 0.001), F_atm + 3 * sm_i, 80)
        v_i  = smile_vol_strip(F_atm, K_i, T_exp, a_i, nu_i, rho_i) * 10_000
        lbl  = f"{T_exp*12:.0f}M" if T_exp < 1 else f"{T_exp:.0f}Y"
        fig_multi.add_trace(go.Scatter(
            x=K_i * 100, y=v_i, mode="lines", name=lbl,
            line=dict(color=pal[i % len(pal)], width=1.6),
        ))
    fig_multi.add_vline(x=F_atm * 100, line_dash="dot", line_color="white", opacity=0.3)
    fig_multi.update_layout(
        xaxis_title="Strike (%)", yaxis_title="Normal Vol (bps)",
        height=400, margin=dict(t=10, b=40),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
        legend=dict(orientation="h", y=1.05, font=dict(size=10)),
    )
    st.plotly_chart(fig_multi, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Cross-index Comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Cap/Floor ATM vol vs Equivalent Swaption vol (same maturity)")

    mat_y = st.slider("Maturity (Y)", 1, 20, 5)

    rows = []
    for idx, meta in INDEX_CATALOG.items():
        ccy = meta["ccy"]
        cf_vol  = cube.capfloor_atm(idx, mat_y) * 10_000
        sw_vol  = cube.swaption_atm(ccy, mat_y, 5.0) * 10_000
        rows.append({
            "Index":         meta["label"],
            "CCY":           ccy,
            "Cap/Floor (bps)": round(cf_vol, 1),
            "Swaption 5Y (bps)": round(sw_vol, 1),
            "Premium (bps)": round(cf_vol - sw_vol, 1),
        })

    comp_df = pd.DataFrame(rows).sort_values(["CCY", "Cap/Floor (bps)"], ascending=[True, False])

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(
        x=comp_df["Index"], y=comp_df["Cap/Floor (bps)"],
        name="Cap/Floor", marker_color="#f4a261",
    ))
    fig_cmp.add_trace(go.Bar(
        x=comp_df["Index"], y=comp_df["Swaption 5Y (bps)"],
        name="Swaption 5Y", marker_color="#2a9d8f",
    ))
    fig_cmp.update_layout(
        barmode="group",
        xaxis_title="Index", yaxis_title="ATM Normal Vol (bps)",
        height=380, margin=dict(t=10, b=100),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
        legend=dict(orientation="h", y=1.05),
    )
    fig_cmp.update_xaxes(tickangle=-30)
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.dataframe(
        comp_df.style.background_gradient(
            subset=["Cap/Floor (bps)", "Swaption 5Y (bps)"],
            cmap="YlOrRd", axis=None,
        ).format({
            "Cap/Floor (bps)": "{:.1f}",
            "Swaption 5Y (bps)": "{:.1f}",
            "Premium (bps)": "{:.1f}",
        }),
        use_container_width=True, hide_index=True,
    )
