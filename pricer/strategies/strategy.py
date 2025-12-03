# pricer/strategies/strategy.py

from __future__ import annotations
from typing import List, Tuple, Dict, Any
from datetime import datetime
import numpy as np
import pandas as pd

from pricer.models.base import OptionModel


class Strategy:
    """
    A multi-leg option strategy composed of (option, qty, label) tuples.
    The option object must implement the OptionModel API.
    """

    def __init__(self, name: str = "Custom Strategy") -> None:
        self.name: str = name
        self.legs: List[Tuple[OptionModel, float, str]] = []

    # ------------------------------------------------------------
    # LEG MANAGEMENT
    # ------------------------------------------------------------
    def add_leg(self, option: OptionModel, qty: float = 1.0, label: str | None = None) -> None:
        label = label or f"{option.option_type.capitalize()} K={option.K}"
        self.legs.append((option, qty, label))

    # ------------------------------------------------------------
    # STRATEGY GREEKS (aggregated)
    # ------------------------------------------------------------
    def price(self) -> float:
        return float(sum(q * opt.price() for opt, q, _ in self.legs))

    def delta(self) -> float:
        return float(sum(q * opt.delta() for opt, q, _ in self.legs))

    def gamma(self) -> float:
        return float(sum(q * opt.gamma() for opt, q, _ in self.legs))

    def vega(self) -> float:
        return float(sum(q * opt.vega() for opt, q, _ in self.legs))

    def theta(self) -> float:
        return float(sum(q * opt.theta() for opt, q, _ in self.legs))

    def rho(self) -> float:
        return float(sum(q * opt.rho() for opt, q, _ in self.legs))

    # ------------------------------------------------------------
    # PRICE VS FORWARD
    # ------------------------------------------------------------
    def price_vs_forward(self, F_values: np.ndarray) -> np.ndarray:
        """
        Reprice the entire strategy as a function of the forward/spot.
        Works for Black76 (F), Bachelier (F), and Black-Scholes (S).
        """
        prices: List[float] = []

        for F_val in F_values:
            total = 0.0
            for opt, qty, _ in self.legs:
                # Rebuild option with *same class*, but updated F or S
                kwargs = dict(
                    K=opt.K,
                    r=getattr(opt, "r", None),
                    sigma=opt.sigma,
                    expiry=opt.expiry,
                    option_type=opt.option_type,
                    valuation_date=opt.valuation_date,
                )

                if hasattr(opt, "F"):
                    new_opt = opt.__class__(F=F_val, **kwargs)
                elif hasattr(opt, "S"):
                    new_opt = opt.__class__(S=F_val, q=getattr(opt, "q", 0.0), **kwargs)
                else:
                    raise ValueError("Unknown model type: option has neither F nor S attribute")

                total += qty * new_opt.price()

            prices.append(total)

        return np.array(prices)

    # ------------------------------------------------------------
    # PAYOFF AT EXPIRY VS FORWARD
    # ------------------------------------------------------------
    def payoff_at_expiry_vs_forward(self, F_values: np.ndarray) -> np.ndarray:
        payoffs: List[float] = []

        for F_val in F_values:
            total = 0.0
            for opt, qty, _ in self.legs:
                if opt.option_type == "call":
                    total += qty * max(F_val - opt.K, 0.0)
                else:
                    total += qty * max(opt.K - F_val, 0.0)
            payoffs.append(total)

        return np.array(payoffs)

    # ------------------------------------------------------------
    # GREKS VS FORWARD
    # ------------------------------------------------------------
    def greeks_vs_forward(self, F_values: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Computes the Greeks for the whole strategy across a range of forward prices.
        """

        results = {g: [] for g in ["price", "delta", "gamma", "vega", "theta", "rho"]}

        for F_val in F_values:
            totals = {g: 0.0 for g in results}

            for opt, qty, _ in self.legs:
                # Rebuild option with updated forward/spot
                kwargs = dict(
                    K=opt.K,
                    r=getattr(opt, "r", None),
                    sigma=opt.sigma,
                    expiry=opt.expiry,
                    option_type=opt.option_type,
                    valuation_date=opt.valuation_date,
                )

                if hasattr(opt, "F"):
                    new_opt = opt.__class__(F=F_val, **kwargs)
                elif hasattr(opt, "S"):
                    new_opt = opt.__class__(S=F_val, q=getattr(opt, "q", 0.0), **kwargs)
                else:
                    raise ValueError("Unknown model type")

                totals["price"] += qty * new_opt.price()
                totals["delta"] += qty * new_opt.delta()
                totals["gamma"] += qty * new_opt.gamma()
                totals["vega"] += qty * new_opt.vega()
                totals["theta"] += qty * new_opt.theta()
                totals["rho"] += qty * new_opt.rho()

            for g in results:
                results[g].append(totals[g])

        return {k: np.array(v) for k, v in results.items()}

    # ------------------------------------------------------------
    # SERIALIZATION (save strategy to DataFrame)
    # ------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        rows = []

        for opt, qty, label in self.legs:
            rows.append({
                "Type": opt.option_type.capitalize(),
                "Strike": opt.K,
                "Expiry": opt.expiry.strftime("%Y-%m-%d"),
                "Qty": qty,
                "σ": opt.sigma,
                "r": getattr(opt, "r", None),
                "F_or_S": getattr(opt, "F", getattr(opt, "S", None)),
                "Valuation Date": opt.valuation_date.strftime("%Y-%m-%d"),
                "Label": label,
            })

        return pd.DataFrame(rows)
