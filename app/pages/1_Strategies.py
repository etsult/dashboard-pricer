# app/pages/1_Strategies.py

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any

from pricer.strategies.strategy import Strategy
from pricer.strategies.custom_strategy import CustomStrategy
from pricer.models.black76 import ForwardOption

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(page_title="Option Strategies", layout="wide")
st.title("📘 Educational Option Strategy Builder")

st.markdown("""
Build multi-leg EQD option strategies and visualize:
- Payoff at expiry
- Payoff today (including premium)
- Greeks vs Forward

All charts are independent for clean analysis.
""")

# ------------------------------------------------------------
# SIDEBAR — MARKET PARAMETERS
# ------------------------------------------------------------

st.sidebar.header("Market Parameters")

F: float = st.sidebar.number_input("Forward (F)", value=5000.0, step=10.0)
r: float = st.sidebar.number_input("Risk-free rate (r)", value=0.02, format="%.4f")
sigma_default: float = st.sidebar.number_input("Default Volatility (σ)", value=0.20, format="%.4f")
valuation_date: datetime = datetime.today()

# ------------------------------------------------------------
# SESSION STATE FOR STRATEGY LEGS
# ------------------------------------------------------------

if "legs_df" not in st.session_state:
    st.session_state.legs_df = pd.DataFrame([{
        "Type": "Call",
        "Strike": F * 1.02,
        "Qty": 1,
        "σ": sigma_default,
        "Expiry": datetime(2025, 12, 19).date()
    }])

st.subheader("Strategy Legs Editor")

edited_legs: pd.DataFrame = st.data_editor(
    st.session_state.legs_df,
    num_rows="dynamic",
    use_container_width=True
)
st.session_state.legs_df = edited_legs

# ------------------------------------------------------------
# BUILD STRATEGY INSTANCE FROM LEGS
# ------------------------------------------------------------

strategy: Strategy = Strategy("Custom Strategy")

for _, row in edited_legs.iterrows():
    expiry: Any = row["Expiry"]
    # Convert date to datetime if needed
    if isinstance(expiry, pd.Timestamp):
        expiry = expiry.to_pydatetime()
    
    opt: ForwardOption = ForwardOption(
        F=F,
        K=row["Strike"],
        r=r,
        sigma=row["σ"],
        expiry=expiry,
        option_type=row["Type"].lower(),
        valuation_date=valuation_date
    )
    strategy.add_leg(opt, qty=row["Qty"], label=f"{row['Type']} K={row['Strike']}")

# ------------------------------------------------------------
# PAYOFF COMPUTATION
# ------------------------------------------------------------

F_range: np.ndarray = np.linspace(0.5 * F, 1.5 * F, 400)

def payoff_today(Fval: float) -> float:
    """Price(F) minus strategy premium at current forward."""
    total: float = sum(
        qty * ForwardOption(
            Fval, opt.K, opt.r, opt.sigma, opt.expiry, opt.option_type, valuation_date
        ).price()
        for opt, qty, _ in strategy.legs
    )
    return total

def payoff_expiry(Fval: float) -> float:
    """Intrinsic payoff at expiry."""
    total: float = 0
    for opt, qty, _ in strategy.legs:
        if opt.option_type == "call":
            total += qty * max(Fval - opt.K, 0)
        else:
            total += qty * max(opt.K - Fval, 0)
    return total

payoff_today_vals: np.ndarray = np.vectorize(payoff_today)(F_range)
payoff_expiry_vals: np.ndarray = np.vectorize(payoff_expiry)(F_range)

premium_cost: float = strategy.price()

# ------------------------------------------------------------
# GREEKS VS FORWARD
# ------------------------------------------------------------

greeks: Dict[str, np.ndarray] = strategy.greeks_vs_forward(F_range)

# ------------------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------------------

st.subheader("📊 Strategy Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Price (Premium)", f"{strategy.price():.2f}")
col2.metric("Delta", f"{strategy.delta():.4f}")
col3.metric("Gamma", f"{strategy.gamma():.6f}")

col4, col5, col6 = st.columns(3)
col4.metric("Vega", f"{strategy.vega():.4f}")
col5.metric("Theta", f"{strategy.theta():.4f}")
col6.metric("Rho", f"{strategy.rho():.4f}")

# ------------------------------------------------------------
# VISUALISATION TABS
# ------------------------------------------------------------

tabs = st.tabs(["Payoff", "Price", "Delta", "Gamma", "Vega", "Theta", "Rho"])

# --- PAYOFF CHART ---
with tabs[0]:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,5))

    ax.plot(F_range, payoff_expiry_vals, label="Payoff at Expiry", linewidth=2)
    ax.plot(F_range, payoff_today_vals, label="Payoff Today (Premium Adjusted)", linestyle="--")
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_title("Payoff Diagram")
    ax.set_xlabel("Forward Price")
    ax.set_ylabel("Payoff")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

# --- PRICE ---
with tabs[1]:
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(F_range, greeks["price"])
    ax.set_title("Price vs Forward")
    ax.grid(True)
    st.pyplot(fig)

# --- DELTA ---
with tabs[2]:
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(F_range, greeks["delta"])
    ax.set_title("Delta vs Forward")
    ax.grid(True)
    st.pyplot(fig)

# --- GAMMA ---
with tabs[3]:
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(F_range, greeks["gamma"])
    ax.set_title("Gamma vs Forward")
    ax.grid(True)
    st.pyplot(fig)

# --- VEGA ---
with tabs[4]:
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(F_range, greeks["vega"])
    ax.set_title("Vega vs Forward")
    ax.grid(True)
    st.pyplot(fig)

# --- THETA ---
with tabs[5]:
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(F_range, greeks["theta"])
    ax.set_title("Theta vs Forward")
    ax.grid(True)
    st.pyplot(fig)

# --- RHO ---
with tabs[6]:
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(F_range, greeks["rho"])
    ax.set_title("Rho vs Forward")
    ax.grid(True)
    st.pyplot(fig)
