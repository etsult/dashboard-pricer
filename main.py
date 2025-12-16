# app/main.py

import streamlit as st

st.set_page_config(page_title="Dashboard", layout="wide")

st.title("📈 Options Analytics Dashboard")

st.markdown("""
Welcome!

Use the navigation menu on the left to access tools:

### 📘 Option Strategy Builder
- Build multi-leg option strategies  
- Visualize payoff and Greeks  
- Plotly interactive charts  
- Break-even detection  
- Supports Black-76, Black-Scholes, Bachelier  

### 🔧 Vol Surface (coming soon)
- Yahoo SPX options  
- Fit SABR, Heston, Dupire  

### 📡 Live Monitor (coming soon)
- Bloomberg data  
- Real-time strategy PnL  
- Historical storage (SQLite)

---
""")