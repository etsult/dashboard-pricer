# pricer/models/bachelier.py

import numpy as np
from scipy.stats import norm
from pricer.models.base import OptionModel

class BachelierOption(OptionModel):
    def __init__(self, F, K, r, sigma, expiry, option_type="call", valuation_date=None):
        super().__init__(K, expiry, option_type, valuation_date)
        self.F = F
        self.r = r
        self.sigma = sigma

    def price(self):
        tau = self.tau
        df = np.exp(-self.r * tau)
        vol = self.sigma * np.sqrt(tau)
        d = (self.F - self.K) / vol if vol > 0 else 0

        if self.option_type == "call":
            return df * ((self.F - self.K)*norm.cdf(d) + vol*norm.pdf(d))
        else:
            return df * ((self.K - self.F)*norm.cdf(-d) + vol*norm.pdf(d))

    def delta(self):
        tau = self.tau
        vol = self.sigma * np.sqrt(tau)
        d = (self.F - self.K) / vol if vol > 0 else 0
        df = np.exp(-self.r * tau)
        return df * norm.cdf(d) if self.option_type == "call" else -df * norm.cdf(-d)

    def gamma(self):
        tau = self.tau
        vol = self.sigma * np.sqrt(tau)
        d = (self.F - self.K) / vol if vol > 0 else 0
        df = np.exp(-self.r * tau)
        return df * norm.pdf(d) / vol

    def vega(self):
        tau = self.tau
        df = np.exp(-self.r * tau)
        return df * np.sqrt(tau) * norm.pdf((self.F - self.K) / (self.sigma*np.sqrt(tau)))

    def theta(self):
        tau = self.tau
        df = np.exp(-self.r * tau)
        return -self.r * self.price()

    def rho(self):
        return -self.tau * self.price()
