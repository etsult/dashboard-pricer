import streamlit as st
from market_data.providers.yahoo import YahooProvider
from surface.builders import build_vol_surface
from surface.visualization import plot_surface

st.title("Volatility Surface Viewer")

provider = YahooProvider()

ticker = st.text_input("Underlying", "^SPX")
rate = st.number_input("Risk-free rate", value=0.02, step=0.005, format="%.3f")
div_yield = st.number_input("Dividend yield", value=0.00, step=0.005, format="%.3f")

if st.button("Load Surface"):
    quotes = provider.get_option_chain(ticker)
    spot = provider.get_forward(ticker)
    surface = build_vol_surface(quotes, spot=spot, rate=rate, dividend_yield=div_yield)
    print(surface.index.min(), surface.index.max())
    fig = plot_surface(surface)
    st.plotly_chart(fig, use_container_width=True)
