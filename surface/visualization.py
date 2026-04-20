# surface/visualization.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


_COLORSCALE = "Plasma"


def plot_surface_3d(surf_df: pd.DataFrame, title: str = "Implied Volatility Surface") -> go.Figure:
    """
    3D surface plot from surf_df (index=log-moneyness, columns=T_years, values=IV).
    """
    k_vals = surf_df.index.values.astype(float)
    T_vals = surf_df.columns.values.astype(float)
    z = surf_df.values.astype(float) * 100  # → percent

    T_days = T_vals * 365

    fig = go.Figure(go.Surface(
        x=k_vals,
        y=T_days,
        z=z,
        colorscale=_COLORSCALE,
        showscale=True,
        colorbar=dict(title="IV (%)", thickness=15),
        opacity=0.92,
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True),
        ),
        hovertemplate="log-m: %{x:.3f}<br>Days: %{y:.0f}<br>IV: %{z:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis=dict(title="Log-moneyness ln(K/F)", dtick=0.1),
            yaxis=dict(title="Days to Expiry"),
            zaxis=dict(title="Implied Vol (%)"),
            camera=dict(eye=dict(x=1.5, y=-1.8, z=0.7)),
            aspectratio=dict(x=1.5, y=1.0, z=0.6),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=560,
    )
    return fig


def plot_smile_slices(
    surf_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    n_slices: int = 6,
) -> go.Figure:
    """
    Per-expiry smile: market scatter points + model fitted line.
    Shows up to n_slices expiries (evenly spaced).
    """
    T_vals = surf_df.columns.values.astype(float)
    k_vals = surf_df.index.values.astype(float)

    # Pick representative expiries
    idx = np.round(np.linspace(0, len(T_vals) - 1, min(n_slices, len(T_vals)))).astype(int)
    T_slices = T_vals[idx]

    n_cols = min(3, len(T_slices))
    n_rows = int(np.ceil(len(T_slices) / n_cols))
    subplot_titles = [f"T = {T*365:.0f}d ({T:.2f}Y)" for T in T_slices]

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
    )

    colors = [
        "#00b4d8", "#e63946", "#2a9d8f", "#f4a261", "#8338ec", "#06d6a0"
    ]

    for i, T in enumerate(T_slices):
        row = i // n_cols + 1
        col = i % n_cols + 1
        color = colors[i % len(colors)]

        # Model curve
        iv_fit = surf_df[T].values * 100  # percent
        fig.add_trace(
            go.Scatter(
                x=k_vals, y=iv_fit,
                mode="lines",
                line=dict(color=color, width=2.5),
                name=f"{T*365:.0f}d model",
                showlegend=(i == 0),
                legendgroup="model",
            ),
            row=row, col=col,
        )

        # Market scatter — match to nearest T
        if not raw_df.empty:
            T_raw = raw_df["T"].values
            nearest = T_raw[np.argmin(np.abs(T_raw - T))]
            if abs(nearest - T) < 0.05:  # within 18 days
                mkt = raw_df[np.abs(raw_df["T"] - nearest) < 0.01]
                if not mkt.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=mkt["k"].values, y=mkt["iv"].values * 100,
                            mode="markers",
                            marker=dict(color=color, size=6, opacity=0.7,
                                        line=dict(color="white", width=0.5)),
                            name=f"{T*365:.0f}d mkt",
                            showlegend=(i == 0),
                            legendgroup="market",
                        ),
                        row=row, col=col,
                    )

        fig.update_xaxes(title_text="log(K/F)", row=row, col=col)
        fig.update_yaxes(title_text="IV (%)", row=row, col=col)

    fig.update_layout(
        height=300 * n_rows + 60,
        margin=dict(t=60, b=30, l=40, r=10),
        legend=dict(orientation="h", y=-0.05),
    )
    return fig


def plot_term_structure(surf_df: pd.DataFrame, raw_df: pd.DataFrame) -> go.Figure:
    """ATM implied vol term structure: model + market ATM points."""
    T_vals = surf_df.columns.values.astype(float)
    k_vals = surf_df.index.values.astype(float)

    # ATM IV = interpolate at k=0 for each T
    atm_idx = np.argmin(np.abs(k_vals))
    atm_iv_model = surf_df.iloc[atm_idx].values * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=T_vals * 365, y=atm_iv_model,
        mode="lines",
        line=dict(color="#00b4d8", width=2.5),
        name="Model ATM IV",
    ))

    if not raw_df.empty:
        # Per expiry, take the point closest to k=0
        atm_mkt = raw_df.groupby("T").apply(
            lambda g: g.iloc[np.argmin(np.abs(g["k"].values))]
        ).reset_index(drop=True)
        fig.add_trace(go.Scatter(
            x=atm_mkt["T"].values * 365, y=atm_mkt["iv"].values * 100,
            mode="markers",
            marker=dict(color="#f4a261", size=9, symbol="diamond"),
            name="Market ATM",
        ))

    fig.update_layout(
        xaxis_title="Days to Expiry",
        yaxis_title="ATM IV (%)",
        height=320,
        margin=dict(t=20, b=40),
        legend=dict(orientation="h", y=1.08),
    )
    return fig
