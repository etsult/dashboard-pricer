from __future__ import annotations
import streamlit as st
from datetime import datetime, date

from pricer.strategies.vanilla_structures import (
    Call, Put,
    Straddle, Strangle,
    CallSpread, PutSpread,
    CallButterfly, PutButterfly,
    CallCalendar, PutCalendar,
    RiskReversal,
    CustomStrategy
)


# ------------------------------------------------------------
# Helper to convert expiry input into datetime
# ------------------------------------------------------------
def _convert_expiry(exp):
    if isinstance(exp, datetime):
        return exp
    if isinstance(exp, date):
        return datetime(exp.year, exp.month, exp.day)
    raise TypeError("Expiry must be date or datetime")


# ------------------------------------------------------------
# Strategy Selector UI
# ------------------------------------------------------------
def strategy_selector(F: float, r: float, sigma_default: float):
    """
    Displays a UI for selecting and configuring a predefined EQD strategy.
    Returns a Strategy object.
    """

    st.sidebar.subheader("Choose a Strategy")

    strategy_name = st.sidebar.selectbox(
        "Strategy Type",
        [
            "Custom",
            "Call", "Put",
            "Straddle", "Strangle",
            "Call Spread", "Put Spread",
            "Call Butterfly", "Put Butterfly",
            "Call Calendar", "Put Calendar",
            "Risk Reversal",
        ]
    )

    # ---------------------------------------------------------
    # CUSTOM STRATEGY
    # ---------------------------------------------------------
    if strategy_name == "Custom":
        st.info("Custom Strategy: configure legs manually in the main app page.")
        return CustomStrategy()

    # ---------------------------------------------------------
    # Single Vanillas
    # ---------------------------------------------------------
    if strategy_name in ("Call", "Put"):
        K = st.sidebar.number_input("Strike", value=F, step=10.0)
        sigma = st.sidebar.number_input("Volatility", value=sigma_default, format="%.4f")
        expiry = _convert_expiry(st.sidebar.date_input("Expiry", date(2025, 12, 19)))

        if strategy_name == "Call":
            return Call(F, K, r, sigma, expiry)
        else:
            return Put(F, K, r, sigma, expiry)

    # ---------------------------------------------------------
    # Straddle / Strangle
    # ---------------------------------------------------------
    if strategy_name in ("Straddle", "Strangle"):
        K = st.sidebar.number_input("ATM Strike", value=F, step=10.0)
        sigma = st.sidebar.number_input("Volatility", value=sigma_default)
        expiry = _convert_expiry(st.sidebar.date_input("Expiry", date(2025, 12, 19)))

        if strategy_name == "Straddle":
            return Straddle(F, K, r, sigma, expiry)
        else:
            width = st.sidebar.number_input("Wings distance", value=50.0)
            return Strangle(F, K, width, r, sigma, expiry)

    # ---------------------------------------------------------
    # Vertical Spreads
    # ---------------------------------------------------------
    if strategy_name in ("Call Spread", "Put Spread"):
        K1 = st.sidebar.number_input("Lower Strike", value=F * 0.98)
        K2 = st.sidebar.number_input("Upper Strike", value=F * 1.02)
        sigma = st.sidebar.number_input("Volatility", value=sigma_default)
        expiry = _convert_expiry(st.sidebar.date_input("Expiry", date(2025, 12, 19)))

        if strategy_name == "Call Spread":
            return CallSpread(F, K1, K2, r, sigma, expiry)

        return PutSpread(F, K1, K2, r, sigma, expiry)

    # ---------------------------------------------------------
    # Butterflies
    # ---------------------------------------------------------
    if strategy_name in ("Call Butterfly", "Put Butterfly"):
        K_center = st.sidebar.number_input("Center Strike", value=F)
        width = st.sidebar.number_input("Wing Width", value=50.0)
        sigma = st.sidebar.number_input("Volatility", value=sigma_default)
        expiry = _convert_expiry(st.sidebar.date_input("Expiry", date(2025, 12, 19)))

        if strategy_name == "Call Butterfly":
            return CallButterfly(F, K_center, width, r, sigma, expiry)

        return PutButterfly(F, K_center, width, r, sigma, expiry)

    # ---------------------------------------------------------
    # Calendars
    # ---------------------------------------------------------
    if strategy_name in ("Call Calendar", "Put Calendar"):
        K = st.sidebar.number_input("Strike", value=F)
        sigma_front = st.sidebar.number_input("Front Vol", value=sigma_default)
        sigma_back = st.sidebar.number_input("Back Vol", value=sigma_default)

        expiry_front = _convert_expiry(st.sidebar.date_input("Front Expiry", date(2025, 6, 21)))
        expiry_back = _convert_expiry(st.sidebar.date_input("Back Expiry", date(2025, 12, 19)))

        if strategy_name == "Call Calendar":
            return CallCalendar(F, K, r, sigma_front, sigma_back, expiry_front, expiry_back)

        return PutCalendar(F, K, r, sigma_front, sigma_back, expiry_front, expiry_back)

    # ---------------------------------------------------------
    # Risk Reversal (25-delta or strike-based)
    # ---------------------------------------------------------
    if strategy_name == "Risk Reversal":
        K_call = st.sidebar.number_input("Call Strike", value=F * 1.02)
        K_put = st.sidebar.number_input("Put Strike", value=F * 0.98)
        sigma = st.sidebar.number_input("Volatility", value=sigma_default)
        expiry = _convert_expiry(st.sidebar.date_input("Expiry", date(2025, 12, 19)))

        return RiskReversal(F, K_call, K_put, r, sigma, expiry)

    # Should not happen
    st.error("Unknown strategy type selected.")
    return CustomStrategy()
