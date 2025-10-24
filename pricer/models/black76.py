import numpy as np
from scipy.stats import norm
from datetime import datetime
import matplotlib.pyplot as plt


class ForwardOption:
    """Black 76 option on a forward contract"""

    def __init__(self, F, K, r, sigma, expiry, option_type="call", valuation_date=None):
        self.F = F
        self.K = K
        self.r = r
        self.sigma = sigma
        self.expiry = expiry
        self.option_type = option_type.lower()
        self.valuation_date = valuation_date if valuation_date else datetime.today()

    @property
    def tau(self):
        dt = (self.expiry - self.valuation_date).days
        return max(dt, 0) / 365.0

    def _d1_d2(self):
        if self.tau == 0 or self.sigma == 0:
            return 0, 0
        d1 = (np.log(self.F / self.K) + 0.5 * self.sigma**2 * self.tau) / (self.sigma * np.sqrt(self.tau))
        d2 = d1 - self.sigma * np.sqrt(self.tau)
        return d1, d2

    def price(self):
        d1, d2 = self._d1_d2()
        df = np.exp(-self.r * self.tau)
        if self.option_type == "call":
            return df * (self.F * norm.cdf(d1) - self.K * norm.cdf(d2))
        elif self.option_type == "put":
            return df * (self.K * norm.cdf(-d2) - self.F * norm.cdf(-d1))
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    def delta(self):
        d1, _ = self._d1_d2()
        df = np.exp(-self.r * self.tau)
        return df * (norm.cdf(d1) if self.option_type == "call" else -norm.cdf(-d1))

    def gamma(self):
        d1, _ = self._d1_d2()
        df = np.exp(-self.r * self.tau)
        return df * norm.pdf(d1) / (self.F * self.sigma * np.sqrt(self.tau))

    def vega(self):
        d1, _ = self._d1_d2()
        df = np.exp(-self.r * self.tau)
        return df * self.F * norm.pdf(d1) * np.sqrt(self.tau)

    def theta(self):
        d1, _ = self._d1_d2()
        df = np.exp(-self.r * self.tau)
        first = -(self.F * self.sigma * df * norm.pdf(d1)) / (2 * np.sqrt(self.tau))
        return first - self.r * self.price()

    def rho(self):
        return -self.tau * self.price()
