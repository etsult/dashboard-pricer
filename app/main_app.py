import sys, os
# ajoute le dossier parent (celui qui contient 'app' et 'pricer')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import uuid  # 👈 for stable leg IDs

from pricer.strategies.strategy import Strategy


# --- PAGE SETUP ---
st.set_page_config(page_title="Option Pricer", layout="centered")
st.title("📈 Option Pricer – Black 76 Model")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Model Parameters")
strategy_type = st.sidebar.selectbox(
    "Strategy",
    ["Custom", "Call", "Put", "Call Calendar", "Put Calendar"]
)

F = st.sidebar.number_input("Forward (F)", value=5500.0)
r = st.sidebar.number_input("Risk-free rate (r)", value=0.02, format="%.4f")
sigma = st.sidebar.number_input("Volatility (σ)", value=0.15, format="%.4f")
valuation_date = datetime.today()

if strategy_type != "Custom":
    st.info("Only the 'Custom' strategy is interactive for now.")
    st.stop()


# --- INIT SESSION STATE ---
if "legs" not in st.session_state:
    st.session_state.legs = []  # list of dicts


# --- UTILS ---
def new_leg():
    """Return a default leg dict with a unique ID."""
    return {
        "id": str(uuid.uuid4()),
        "Type": "Call",
        "Strike": 5600.0,
        "Qty": 1,
        "σ": sigma,
        "Expiry": datetime(2025, 12, 19).date(),
    }


def display_leg(leg, idx):
    """Display a leg editor card with insert/delete buttons."""
    with st.expander(f"Leg {idx + 1}: {leg['Type']} K={leg['Strike']}", expanded=True):
        cols = st.columns(6)
        leg["Type"] = cols[0].selectbox("Type", ["Call", "Put"],
                                        index=0 if leg["Type"].lower() == "call" else 1,
                                        key=f"type_{leg['id']}")
        leg["Strike"] = cols[1].number_input("Strike", value=float(leg["Strike"]),
                                             key=f"K_{leg['id']}")
        leg["Qty"] = cols[2].number_input("Quantity", value=float(leg["Qty"]),
                                          key=f"qty_{leg['id']}")
        leg["σ"] = cols[3].number_input("Vol (σ)", value=float(leg["σ"]),
                                        key=f"sigma_{leg['id']}")
        leg["Expiry"] = cols[4].date_input("Expiry", value=leg["Expiry"],
                                           key=f"expiry_{leg['id']}")

        remove = cols[5].button("🗑️", key=f"delete_{leg['id']}")
        insert_above = st.button("⬆️ Insert Above", key=f"insert_above_{leg['id']}")
        insert_below = st.button("⬇️ Insert Below", key=f"insert_below_{leg['id']}")
    return leg, remove, insert_above, insert_below


# --- UI ---
st.markdown("### ⚙️ Strategy Legs")

# Global add button (for when there are no legs yet)
if st.button("➕ Add Leg"):
    st.session_state.legs.append(new_leg())

# No legs? show info
if not st.session_state.legs:
    st.info("No legs yet. Click ➕ Add Leg to start.")
else:
    updated_legs = []
    insertion_queue = []

    for idx, leg in enumerate(st.session_state.legs):
        updated_leg, remove, insert_above, insert_below = display_leg(leg, idx)

        # handle insertions
        if insert_above:
            insertion_queue.append(("above", idx))
        elif insert_below:
            insertion_queue.append(("below", idx))

        # handle removal
        if not remove:
            updated_legs.append(updated_leg)

    # apply insertions after rendering
    for direction, idx in reversed(insertion_queue):  # reversed to keep indices stable
        if direction == "above":
            updated_legs.insert(idx, new_leg())
        else:  # below
            updated_legs.insert(idx + 1, new_leg())

    st.session_state.legs = updated_legs


# --- BUILD STRATEGY ---
if st.session_state.legs:
    df = pd.DataFrame([
        {
            "Type": leg["Type"],
            "Strike": leg["Strike"],
            "Expiry": leg["Expiry"].strftime("%Y-%m-%d"),
            "Qty": leg["Qty"],
            "σ": leg["σ"],
            "r": r,
            "F": F,
            "Valuation Date": valuation_date.strftime("%Y-%m-%d"),
        }
        for leg in st.session_state.legs
    ])

    strategy = Strategy.from_dataframe(df)

    # --- RESULTS ---
    st.markdown("### 💰 Strategy Summary")

    cols = st.columns(3)
    cols[0].metric("Price", f"{strategy.price():.2f}")
    cols[1].metric("Delta", f"{strategy.delta():.2f}")
    cols[2].metric("Gamma", f"{strategy.gamma():.6f}")

    cols = st.columns(3)
    cols[0].metric("Vega", f"{strategy.vega():.2f}")
    cols[1].metric("Theta", f"{strategy.theta():.2f}")
    cols[2].metric("Rho", f"{strategy.rho():.2f}")

    # --- PLOTS ---
    st.markdown("### 📊 Greeks and Price vs Forward")
    F_values = np.linspace(0.5 * F, 1.5 * F, 200)
    data = strategy.greeks_vs_forward(F_values)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()
    for ax, greek in zip(axes, ["price", "delta", "gamma", "vega", "theta", "rho"]):
        ax.plot(data["F"], data[greek])
        ax.set_title(greek.capitalize())
        ax.grid(True)
    st.pyplot(fig)
