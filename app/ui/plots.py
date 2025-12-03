# app/ui/plots.py

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from pricer.strategies.strategy import Strategy


def find_break_even(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return F-values where payoff_today crosses zero."""
    idx = np.where(np.diff(np.sign(y)))[0]
    if len(idx) == 0:
        return np.array([])
    # linear interpolation
    be = x[idx] - y[idx] * (x[idx+1] - x[idx]) / (y[idx+1] - y[idx])
    return be


def plot_payoff(F_range: np.ndarray, payoff_today: np.ndarray, payoff_expiry: np.ndarray) -> None:
    """Plot payoff today vs payoff at expiry with break-even detection."""

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=F_range, y=payoff_expiry,
        name="Payoff at Expiry",
        line=dict(width=2)
    ))

    fig.add_trace(go.Scatter(
        x=F_range, y=payoff_today,
        name="Payoff Today",
        line=dict(width=2, dash="dash")
    ))

    # Break-even
    be_points = find_break_even(F_range, payoff_today)

    for be in be_points:
        fig.add_vline(x=float(be), line_dash="dot", line_color="red")
        fig.add_annotation(x=float(be), y=0, text=f"BE {be:.1f}", showarrow=True)

    fig.add_hline(y=0, line_width=1, line_color="black")

    fig.update_layout(
        title="Payoff Diagram",
        xaxis_title="Forward Price",
        yaxis_title="Payoff",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_greek(F_range: np.ndarray, values: np.ndarray, greek: str) -> None:
    """Generic Plotly Greek plot."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=F_range, y=values, name=greek.capitalize()))
    fig.update_layout(
        title=f"{greek.capitalize()} vs Forward",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)
