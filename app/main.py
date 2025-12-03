import sys, os
# Add parent directory (contains 'app' and 'pricer')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from ui.inputs import sidebar_inputs
from ui.legs_editor import leg_editor
from pricer.strategies.strategy import Strategy

st.set_page_config(page_title="Option Pricer", layout="wide")
st.title("Option Strategy Visualizer")

# --- SIDEBAR ---
ticker, params = sidebar_inputs()

# --- LEGS ---
if "legs" not in st.session_state:
    st.session_state.legs = pd.DataFrame([{
        "Type": "Call",
        "Strike": params["F"] * 1.02,
        "Qty": 1,
        "Sigma": params["sigma"],
        "Expiry": datetime(2025,12,19).date()
    }])

df_legs = leg_editor(st.session_state.legs)
st.session_state.legs = df_legs

# --- STRATEGY ---
strategy = Strategy.from_dataframe(
    df_legs,
    F=params["F"],
    r=params["r"],
    q=params["q"],
)

st.header("Results")
st.write(f"Data source: {params['source'].upper()}")

F_range = np.linspace(0.5 * params["F"], 1.5 * params["F"], 200)
st.line_chart(strategy.price_vs_forward(F_range))
