# surface/builders.py

from typing import Dict, List
from market_data.schema import OptionQuote
import pandas as pd


def build_vol_surface(quotes: List[OptionQuote]):
    """
    Returns:
        df : DataFrame with rows = strikes, columns = expiries, values = implied vol
    """

    rows = []

    for q in quotes:
        rows.append({
            "expiry": q.expiry,
            "strike": q.strike,
            "type": q.option_type,
            "iv": q.implied_vol,
        })

    df = pd.DataFrame(rows)

    # Pivot into surface matrix
    surface = df.pivot_table(
        index="strike",
        columns="expiry",
        values="iv",
        aggfunc="mean"
    ).sort_index()

    return surface
