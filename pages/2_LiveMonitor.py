import streamlit as st
import pandas as pd
from market_data.router import MarketDataRouter
from pricer.strategies.strategy import Strategy
from app.ui.strategy_selector import strategy_selector

st.set_page_config(page_title="Live Monitor", layout="wide")
st.title("📡 Live Option Strategy Monitor")

st.markdown("""
Monitor live prices and Greeks for a multi-leg option strategy,
powered by **Bloomberg OMON via BLPAPI**.
""")

# ------------------------------------------------------------
# STRATEGY SELECTION
# ------------------------------------------------------------
st.subheader("1. Select Strategy")

F = st.number_input("Reference Forward (for Δ-adj calculation)", value=5800.0, step=25.0)
r = st.number_input("Risk-free rate", value=0.02)
sigma = st.number_input("Default vol", value=0.20)

strategy: Strategy = strategy_selector(F, r, sigma)

if strategy is None or len(strategy.legs) == 0:
    st.warning("Please build a strategy first.")
    st.stop()

# ------------------------------------------------------------
# DATA FETCH
# ------------------------------------------------------------
st.subheader("2. Fetch Live Bloomberg Data")

provider = MarketDataRouter(provider="bbg")
data = provider.get_quotes(strategy)

df = pd.DataFrame(data)

# Merge strategy quantities into table
df["Qty"] = [qty for _, qty, _ in strategy.legs]
df["Label"] = [label for _, _, label in strategy.legs]

df = df[
    ["Label", "Qty", "Bid", "BidSize", "Mid", "Ask", "AskSize", "Volume", "Delta", "IVMid"]
]

st.dataframe(df, use_container_width=True)

# ------------------------------------------------------------
# STRATEGY VALUATION
# ------------------------------------------------------------
st.subheader("3. Strategy Valuation")

net_price = (df["Mid"] * df["Qty"]).sum()
net_delta = (df["Delta"] * df["Qty"]).sum()

delta_adj_price = net_price + net_delta * F

col1, col2 = st.columns(2)
col1.metric("Net Mid Price", f"{net_price:.4f}")
col2.metric("Δ-Adjusted Price", f"{delta_adj_price:.4f}")

st.info(f"Total Delta: {net_delta:.4f}")

# ------------------------------------------------------------
# AUTO REFRESH
# ------------------------------------------------------------
refresh = st.slider("Auto-refresh (seconds)", 0, 30, 5)
if refresh > 0:
    import time
    time.sleep(refresh)
    st.rerun()
