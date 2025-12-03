# pricer/models/black76.py

from __future__ import annotations
from typing import Tuple, Union
from datetime import datetime, date

import numpy as np
from scipy.stats import norm

from pricer.models.base import OptionModel


class Black76Option(OptionModel):
    """
    Black-76 model for options on forwards/futures.
    Implements the standard OptionModel API.
    """

    def __init__(
        self,
        F: float,
        K: float,
        r: float,
        sigma: float,
        expiry: Union[datetime, date],
        option_type: str = "call",
        valuation_date: Union[datetime, None] = None,
    ) -> None:

        super().__init__(K, expiry, option_type, valuation_date)

        self.F: float = float(F)
        self.r: float = float(r)
        self.sigma: float = float(sigma)

    # ------------------------------------------------------------
    # Internal: compute d1/d2
    # ------------------------------------------------------------
    def _d1_d2(self) -> Tuple[float, float]:
        tau = self.tau
        if tau == 0 or self.sigma == 0:
            return 0.0, 0.0

        d1 = (np.log(self.F / self.K) + 0.5 * self.sigma**2 * tau) / (self.sigma * np.sqrt(tau))
        d2 = d1 - self.sigma * np.sqrt(tau)

        return float(d1), float(d2)

    # ------------------------------------------------------------
    # Price
    # ------------------------------------------------------------
    def price(self) -> float:
        d1, d2 = self._d1_d2()
        df = np.exp(-self.r * self.tau)

        if self.option_type == "call":
            return float(df * (self.F * norm.cdf(d1) - self.K * norm.cdf(d2)))

        return float(df * (self.K * norm.cdf(-d2) - self.F * norm.cdf(-d1)))

    # ------------------------------------------------------------
    # Greeks
    # ------------------------------------------------------------
    def delta(self) -> float:
        d1, _ = self._d1_d2()
        df = np.exp(-self.r * self.tau)
        if self.option_type == "call":
            return float(df * norm.cdf(d1))
        return float(-df * norm.cdf(-d1))

    def gamma(self) -> float:
        d1, _ = self._d1_d2()
        tau = self.tau
        df = np.exp(-self.r * tau)
        return float(df * norm.pdf(d1) / (self.F * self.sigma * np.sqrt(tau)))

    def vega(self) -> float:
        d1, _ = self._d1_d2()
        tau = self.tau
        df = np.exp(-self.r * tau)
        return float(df * self.F * norm.pdf(d1) * np.sqrt(tau))

    def theta(self) -> float:
        d1, _ = self._d1_d2()
        tau = self.tau
        df = np.exp(-self.r * tau)

        term1 = -(self.F * self.sigma * df * norm.pdf(d1)) / (2 * np.sqrt(tau))
        term2 = -self.r * self.price()

        return float(term1 + term2)

    def rho(self) -> float:
        return float(-self.tau * self.price())
