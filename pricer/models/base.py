# pricer/models/base.py

from abc import ABC, abstractmethod
from datetime import datetime, date

class OptionModel(ABC):
    """
    Abstract base class for option pricing models.
    Every model (Black-76, BS, Bachelier...) must implement this API.
    """

    def __init__(self, K, expiry, option_type="call", valuation_date=None):
        self.K = K
        self.expiry = expiry
        self.option_type = option_type.lower()

        if valuation_date is None:
            valuation_date = datetime.today()

        # Normalize to datetime
        if isinstance(expiry, date) and not isinstance(expiry, datetime):
            expiry = datetime.combine(expiry, datetime.min.time())

        if isinstance(valuation_date, date) and not isinstance(valuation_date, datetime):
            valuation_date = datetime.combine(valuation_date, datetime.min.time())

        self.expiry = expiry
        self.valuation_date = valuation_date

    @property
    def tau(self):
        dt = (self.expiry - self.valuation_date).days
        return max(dt, 0) / 365.0

    # ---- abstract methods ----
    @abstractmethod
    def price(self): ...
    @abstractmethod
    def delta(self): ...
    @abstractmethod
    def gamma(self): ...
    @abstractmethod
    def vega(self): ...
    @abstractmethod
    def theta(self): ...
    @abstractmethod
    def rho(self): ...
