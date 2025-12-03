# app/pages/1_Strategies.py

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# --- UI MODULES ---
from app.ui.inputs import market_inputs
from app.ui.legs_editor import edit_legs
from app.ui.helpers import build_strategy_from_df
from app.ui.plots import plot_payoff, plot_greek
from app.ui.strategy_selector import strategy_selector


# --- PRICING ---
from pricer.strategies.strategy import Strategy


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(page_title="Option Strategies", layout="wide")
st.title("📘 Educational Option Strategy Builder")

st.markdown("""
Build multi-leg EQD option strategies and visualize:

- Payoff at expiry  
- Payoff today (premium included)  
- Greeks vs forward  
- Break-even levels  

Use Black-76, Black-Scholes, or Bachelier models.
""")


# ------------------------------------------------------------
# MARKET PARAMS (from sidebar)
# ------------------------------------------------------------

F, r, q, sigma_default, valuation_date, model_name = market_inputs()


# ------------------------------------------------------------
# LEGS EDITOR
# ------------------------------------------------------------

legs_df: pd.DataFrame = edit_legs(default_F=F, default_sigma=sigma_default)


# ------------------------------------------------------------
# STRATEGY BUILDING
# ------------------------------------------------------------

strategy: Strategy = strategy_selector(F, r, sigma_default)


# ------------------------------------------------------------
# COMPUTE PAYOFFS & GREEKS
# ------------------------------------------------------------

F_range: np.ndarray = np.linspace(0.5 * F, 1.5 * F, 400)

# Price(F) including premium
payoff_today_vals: np.ndarray = strategy.price_vs_forward(F_range)

# Intrinsic payoff at expiry
payoff_expiry_vals: np.ndarray = strategy.payoff_at_expiry_vs_forward(F_range)

# Greeks
greeks: dict[str, np.ndarray] = strategy.greeks_vs_forward(F_range)


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

st.subheader("📊 Strategy Metrics")

c1, c2, c3 = st.columns(3)
c1.metric("Price (Premium)", f"{strategy.price():.4f}")
c2.metric("Delta", f"{strategy.delta():.4f}")
c3.metric("Gamma", f"{strategy.gamma():.6f}")

c4, c5, c6 = st.columns(3)
c4.metric("Vega", f"{strategy.vega():.4f}")
c5.metric("Theta", f"{strategy.theta():.4f}")
c6.metric("Rho", f"{strategy.rho():.4f}")


# ------------------------------------------------------------
# TABS: PAYOFF + GREEKS
# ------------------------------------------------------------

tabs = st.tabs(["Payoff", "Price", "Delta", "Gamma", "Vega", "Theta", "Rho"])


# --- PAYOFF ---
with tabs[0]:
    plot_payoff(F_range, payoff_today_vals, payoff_expiry_vals)


# --- PRICE ---
with tabs[1]:
    plot_greek(F_range, greeks["price"], "price")


# --- DELTA ---
with tabs[2]:
    plot_greek(F_range, greeks["delta"], "delta")


# --- GAMMA ---
with tabs[3]:
    plot_greek(F_range, greeks["gamma"], "gamma")


# --- VEGA ---
with tabs[4]:
    plot_greek(F_range, greeks["vega"], "vega")


# --- THETA ---
with tabs[5]:
    plot_greek(F_range, greeks["theta"], "theta")


# --- RHO ---
with tabs[6]:
    plot_greek(F_range, greeks["rho"], "rho")
