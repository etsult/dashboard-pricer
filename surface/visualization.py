# surface/visualization.py

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata


# -------------------------------------------------------------------
# 1. Minimal cleaning: only sort axes, keep ALL data (no dropping)
# -------------------------------------------------------------------
def prepare_surface(surface_df: pd.DataFrame) -> pd.DataFrame:
    # Sort strikes (index)
    surface_df = surface_df.sort_index()

    # Sort expiries (columns)
    expiries = pd.to_datetime(surface_df.columns)
    expiries = np.sort(expiries)

    # Reindex columns in sorted order
    surface_df = surface_df.reindex(columns=expiries)

    return surface_df


def _regularize_grid(surface_df: pd.DataFrame):
    """Interpolate onto a regular grid for a smoother surface (linear with nearest fallback)."""

    surface_df = prepare_surface(surface_df)

    # Axes (index = log-moneyness, columns = expiries)
    log_m = surface_df.index.values
    expiries = pd.to_datetime(surface_df.columns).values
    expiries_numeric = (expiries - expiries.min()) / np.timedelta64(1, "D")

    # If the grid is very large, downsample expiries to keep plotting reasonable
    max_expiry_cols = 220
    if surface_df.shape[1] > max_expiry_cols:
        step = int(np.ceil(surface_df.shape[1] / max_expiry_cols))
        surface_df = surface_df.iloc[:, ::step]
        expiries = pd.to_datetime(surface_df.columns).values
        expiries_numeric = (expiries - expiries.min()) / np.timedelta64(1, "D")

    # Raw grid (rows=log-m, cols=expiries) → transpose so z[y, x]
    z_raw = surface_df.values.astype(float)
    z_raw = np.where((z_raw > 0) & (z_raw < 5.0), z_raw, np.nan)
    z = z_raw.T

    X_known, Y_known = np.meshgrid(log_m, expiries_numeric, indexing="xy")
    mask = ~np.isnan(z)

    # If too little data, return raw grid to avoid overfitting
    if mask.sum() < 6:
        return strikes, expiries_numeric, z

    points = np.column_stack((X_known[mask], Y_known[mask]))
    values = z[mask]

    # Target grid with light padding to allow extension beyond convex hull
    strike_span = log_m.max() - log_m.min()
    expiry_span = expiries_numeric.max() - expiries_numeric.min()
    strike_grid = np.linspace(log_m.min() - 0.05 * strike_span, log_m.max() + 0.05 * strike_span, 120)
    expiry_grid = np.linspace(expiries_numeric.min() - 0.05 * expiry_span, expiries_numeric.max() + 0.05 * expiry_span, 70)
    X_target, Y_target = np.meshgrid(strike_grid, expiry_grid, indexing="xy")

    # Linear interpolation (fast) with nearest fallback to fill holes
    z_interp = griddata(points, values, (X_target, Y_target), method="linear")
    if np.isnan(z_interp).any():
        z_nearest = griddata(points, values, (X_target, Y_target), method="nearest")
        z_interp = np.where(np.isnan(z_interp), z_nearest, z_interp)

    return strike_grid, expiry_grid, z_interp


# -------------------------------------------------------------------
# 2. Full 3D surface plot (no aggressive cleaning, no clipping)
# -------------------------------------------------------------------
def plot_surface(surface_df: pd.DataFrame):
    log_m, expiries_numeric, z = _regularize_grid(surface_df)

    fig = go.Figure(
        data=[
            go.Surface(
                x=log_m,
                y=expiries_numeric,
                z=z,
                colorscale="Viridis",
                showscale=True,
                opacity=0.95,
                hovertemplate="log-m: %{x:.3f}<br>Days: %{y:.0f}<br>IV: %{z:.2f}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title="Implied Volatility Surface",
        scene=dict(
            xaxis_title="Log-moneyness ln(K/F)",
            yaxis_title="Days to Expiry",
            zaxis_title="Implied Vol",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8)),
        ),
        autosize=True,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig
