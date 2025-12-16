# pricer/models/black_scholes.py

import numpy as np
from scipy.stats import norm
from pricer.models.base import OptionModel

class BlackScholesOption(OptionModel):
    def __init__(self, S, K, r, q, sigma, expiry, option_type="call", valuation_date=None):
        super().__init__(K, expiry, option_type, valuation_date)
        self.S = S
        self.r = r
        self.q = q
        self.sigma = sigma

    def _d1_d2(self):
        if self.tau == 0 or self.sigma == 0:
            return 0, 0
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5*self.sigma**2)*self.tau) / (self.sigma*np.sqrt(self.tau))
        d2 = d1 - self.sigma * np.sqrt(self.tau)
        return d1, d2

    def price(self):
        d1, d2 = self._d1_d2()
        df = np.exp(-self.r * self.tau)
        dq = np.exp(-self.q * self.tau)
        if self.option_type == "call":
            return dq * self.S * norm.cdf(d1) - df * self.K * norm.cdf(d2)
        else:
            return df * self.K * norm.cdf(-d2) - dq * self.S * norm.cdf(-d1)

    def delta(self):
        d1, _ = self._d1_d2()
        dq = np.exp(-self.q * self.tau)
        return dq * (norm.cdf(d1) if self.option_type == "call" else -norm.cdf(-d1))

    def gamma(self):
        d1, _ = self._d1_d2()
        dq = np.exp(-self.q * self.tau)
        return dq * norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.tau))

    def vega(self):
        d1, _ = self._d1_d2()
        dq = np.exp(-self.q * self.tau)
        return dq * self.S * norm.pdf(d1) * np.sqrt(self.tau)

    def theta(self):
        d1, d2 = self._d1_d2()
        dq = np.exp(-self.q * self.tau)
        df = np.exp(-self.r * self.tau)
        term1 = -(dq * self.S * self.sigma * norm.pdf(d1)) / (2*np.sqrt(self.tau))
        if self.option_type == "call":
            return term1 - self.r * df * self.K * norm.cdf(d2) + self.q * dq * self.S * norm.cdf(d1)
        else:
            return term1 + self.r * df * self.K * norm.cdf(-d2) - self.q * dq * self.S * norm.cdf(-d1)

    def rho(self):
        _, d2 = self._d1_d2()
        df = np.exp(-self.r*self.tau)
        if self.option_type == "call":
            return self.K * self.tau * df * norm.cdf(d2)
        else:
            return -self.K * self.tau * df * norm.cdf(-d2)
