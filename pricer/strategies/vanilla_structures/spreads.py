# pricer/strategies/vanilla_structures/spreads.py

from __future__ import annotations
from datetime import datetime
from pricer.strategies.strategy import Strategy
from pricer.models.factory import OptionFactory


class CallSpread(Strategy):
    def __init__(self, F, K1, K2, r, sigma, expiry, qty=1.0, model="black76"):
        super().__init__("Call Spread")

        long_call = OptionFactory.create(model, F=F, K=K1, r=r, sigma=sigma,
                                         expiry=expiry, option_type="call",
                                         valuation_date=datetime.today())

        short_call = OptionFactory.create(model, F=F, K=K2, r=r, sigma=sigma,
                                          expiry=expiry, option_type="call",
                                          valuation_date=datetime.today())

        self.add_leg(long_call, qty, f"Long Call K={K1}")
        self.add_leg(short_call, -qty, f"Short Call K={K2}")


class PutSpread(Strategy):
    def __init__(self, F, K1, K2, r, sigma, expiry, qty=1.0, model="black76"):
        super().__init__("Put Spread")

        long_put = OptionFactory.create(model, F=F, K=K1, r=r, sigma=sigma,
                                        expiry=expiry, option_type="put",
                                        valuation_date=datetime.today())

        short_put = OptionFactory.create(model, F=F, K=K2, r=r, sigma=sigma,
                                         expiry=expiry, option_type="put",
                                         valuation_date=datetime.today())

        self.add_leg(long_put, qty, f"Long Put K={K1}")
        self.add_leg(short_put, -qty, f"Short Put K={K2}")
