import numpy as np
from scipy.stats import norm
from datetime import datetime, date
from typing import Tuple, Union
import matplotlib.pyplot as plt


class ForwardOption:
    """Black 76 option on a forward contract"""

    def __init__(
        self,
        F: float,
        K: float,
        r: float,
        sigma: float,
        expiry: Union[datetime, date],
        option_type: str = "call",
        valuation_date: Union[datetime, None] = None
    ) -> None:
        self.F: float = F
        self.K: float = K
        self.r: float = r
        self.sigma: float = sigma
        self.expiry: Union[datetime, date] = expiry
        self.option_type: str = option_type.lower()
        self.valuation_date: datetime = valuation_date if valuation_date else datetime.today()

    @property
    def tau(self) -> float:
        # Ensure both values are datetime objects
        exp = self.expiry
        val = self.valuation_date

        if isinstance(exp, date) and not isinstance(exp, datetime):
            exp = datetime.combine(exp, datetime.min.time())

        if isinstance(val, date) and not isinstance(val, datetime):
            val = datetime.combine(val, datetime.min.time())

        dt = (exp - val).days
        return max(dt, 0) / 365.0


    def _d1_d2(self) -> Tuple[float, float]:
        if self.tau == 0 or self.sigma == 0:
            return 0, 0
        d1: float = (np.log(self.F / self.K) + 0.5 * self.sigma**2 * self.tau) / (self.sigma * np.sqrt(self.tau))
        d2: float = d1 - self.sigma * np.sqrt(self.tau)
        return d1, d2

    def price(self) -> float:
        d1, d2 = self._d1_d2()
        df: float = np.exp(-self.r * self.tau)
        if self.option_type == "call":
            return df * (self.F * norm.cdf(d1) - self.K * norm.cdf(d2))
        elif self.option_type == "put":
            return df * (self.K * norm.cdf(-d2) - self.F * norm.cdf(-d1))
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    def delta(self) -> float:
        d1, _ = self._d1_d2()
        df: float = np.exp(-self.r * self.tau)
        return df * (norm.cdf(d1) if self.option_type == "call" else -norm.cdf(-d1))

    def gamma(self) -> float:
        d1, _ = self._d1_d2()
        df: float = np.exp(-self.r * self.tau)
        return df * norm.pdf(d1) / (self.F * self.sigma * np.sqrt(self.tau))

    def vega(self) -> float:
        d1, _ = self._d1_d2()
        df: float = np.exp(-self.r * self.tau)
        return df * self.F * norm.pdf(d1) * np.sqrt(self.tau)

    def theta(self) -> float:
        d1, _ = self._d1_d2()
        df: float = np.exp(-self.r * self.tau)
        first: float = -(self.F * self.sigma * df * norm.pdf(d1)) / (2 * np.sqrt(self.tau))
        return first - self.r * self.price()

    def rho(self) -> float:
        return -self.tau * self.price()
