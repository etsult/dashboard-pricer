# app/ui/inputs.py

from __future__ import annotations
from typing import Tuple
from datetime import datetime
import streamlit as st


def market_inputs() -> Tuple[float, float, float, float, datetime, str]:
    """
    Sidebar widget for market-level inputs.

    Returns
    -------
    F : float
    r : float
    q : float
    sigma_default : float
    valuation_date : datetime
    model_name : str
    """

    st.sidebar.header("Market Parameters")

    model_name: str = st.sidebar.selectbox(
        "Pricing Model",
        ["Black-76", "Black-Scholes", "Bachelier"],
        index=0,
    )

    F: float = st.sidebar.number_input("Forward / Spot (F or S)", value=5000.0, step=10.0)
    r: float = st.sidebar.number_input("Risk-free rate (r)", value=0.02, format="%.4f")
    q: float = st.sidebar.number_input("Dividend yield (q)", value=0.00, format="%.4f")
    sigma_default: float = st.sidebar.number_input("Default Volatility (σ)", value=0.20, format="%.4f")

    valuation_date: datetime = datetime.today()

    return F, r, q, sigma_default, valuation_date, model_name
