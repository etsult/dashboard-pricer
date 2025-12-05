import streamlit as st
from market_data.providers.yahoo import YahooProvider
from surface.builders import build_vol_surface
from surface.visualization import plot_surface

st.title("Volatility Surface Viewer")

provider = YahooProvider()

ticker = st.text_input("Underlying", "SPX")

if st.button("Load Surface"):
    quotes = provider.get_option_chain(ticker)
    surface = build_vol_surface(quotes)
    fig = plot_surface(surface)
    st.plotly_chart(fig, use_container_width=True)
