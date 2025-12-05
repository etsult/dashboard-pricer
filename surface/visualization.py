# surface/visualization.py

import plotly.express as px
import pandas as pd

def plot_surface(surface_df: pd.DataFrame):
    df = surface_df.reset_index().melt(id_vars="strike", var_name="expiry", value_name="iv")

    fig = px.scatter_3d(
        df,
        x="strike",
        y="expiry",
        z="iv",
        color="iv",
        title="Implied Volatility Surface",
    )
    return fig
