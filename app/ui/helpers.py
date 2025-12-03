# app/ui/helpers.py

from __future__ import annotations
from typing import Literal, Type
from datetime import datetime, date

import pandas as pd

from pricer.strategies.strategy import Strategy
from pricer.models.black76 import Black76Option
from pricer.models.black_scholes import BlackScholesOption
from pricer.models.bachelier import BachelierOption

# Type alias for allowed models
ModelName = Literal["Black-76", "Black-Scholes", "Bachelier"]


def build_strategy_from_df(
    df: pd.DataFrame,
    model: ModelName,
    F: float,
    r: float,
    q: float,
    sigma_default: float,
    valuation_date: datetime,
) -> Strategy:
    """
    Build a Strategy object from a DataFrame defining the legs.

    Parameters
    ----------
    df : pd.DataFrame
        Columns: Type, Strike, Qty, σ, Expiry
    model : Literal["Black-76", "Black-Scholes", "Bachelier"]
        Pricing model selected in the UI.
    F : float
        Forward price (for Black-76 & Bachelier)
    r : float
        Risk-free rate
    q : float
        Dividend yield (for Black-Scholes)
    sigma_default : float
        Default volatility if missing
    valuation_date : datetime
        Today date for pricing

    Returns
    -------
    Strategy
        Fully constructed option strategy.
    """

    strategy = Strategy("Custom Strategy")

    for _, row in df.iterrows():
        K: float = float(row["Strike"])
        qty: float = float(row["Qty"])
        sigma: float = float(row["σ"]) if not pd.isna(row["σ"]) else sigma_default

        expiry_raw = row["Expiry"]
        if isinstance(expiry_raw, pd.Timestamp):
            expiry: datetime = expiry_raw.to_pydatetime()
        elif isinstance(expiry_raw, date):
            expiry = datetime.combine(expiry_raw, datetime.min.time())
        elif isinstance(expiry_raw, datetime):
            expiry = expiry_raw
        else:
            raise TypeError(f"Unsupported expiry type: {type(expiry_raw)}")

        opt_type: str = row["Type"].lower()

        # -------------------------
        # Instantiate correct model
        # -------------------------
        if model == "Black-76":
            opt = Black76Option(F, K, r, sigma, expiry, option_type=opt_type, valuation_date=valuation_date)

        elif model == "Black-Scholes":
            opt = BlackScholesOption(
                S=F,  # Using F as proxy for S unless user provides spot explicitly
                K=K,
                r=r,
                q=q,
                sigma=sigma,
                expiry=expiry,
                option_type=opt_type,
                valuation_date=valuation_date,
            )

        elif model == "Bachelier":
            opt = BachelierOption(F, K, r, sigma, expiry, option_type=opt_type, valuation_date=valuation_date)

        else:
            raise ValueError(f"Unknown model '{model}'")

        strategy.add_leg(opt, qty=qty, label=f"{row['Type']} K={K}")

    return strategy
