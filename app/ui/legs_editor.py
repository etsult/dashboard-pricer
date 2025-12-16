# app/ui/legs_editor.py

from __future__ import annotations
from typing import Optional
from datetime import date
import pandas as pd
import streamlit as st


def edit_legs(default_F: float, default_sigma: float) -> pd.DataFrame:
    """
    Display and manage the option legs editor.

    Returns
    -------
    pd.DataFrame
        Columns: Type, Strike, Qty, σ, Expiry
    """

    if "legs_df" not in st.session_state:
        st.session_state.legs_df = pd.DataFrame([{
            "Type": "Call",
            "Strike": default_F * 1.02,
            "Qty": 1.0,
            "σ": default_sigma,
            "Expiry": date(2025, 12, 19),
        }])

    st.subheader("Strategy Legs Editor")

    df: pd.DataFrame = st.data_editor(
        st.session_state.legs_df,
        num_rows="dynamic",
        use_container_width=True,
    )

    st.session_state.legs_df = df
    return df
