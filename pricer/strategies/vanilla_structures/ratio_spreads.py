from __future__ import annotations
from datetime import datetime
from pricer.strategies.strategy import Strategy
from pricer.models.factory import OptionFactory


class CallRatioSpread(Strategy):
    """Long 1 call, short 2 calls."""
    def __init__(self, F, K1, K2, r, sigma, expiry, model="black76"):
        super().__init__("Call Ratio Spread 1x2")

        long = OptionFactory.create(model, F, K1, r, sigma, expiry, "call", datetime.today())
        short = OptionFactory.create(model, F, K2, r, sigma, expiry, "call", datetime.today())

        self.add_leg(long, 1, f"Long Call K={K1}")
        self.add_leg(short, -2, f"Short 2 Calls K={K2}")


class PutRatioSpread(Strategy):
    """Long 1 put, short 2 puts."""
    def __init__(self, F, K1, K2, r, sigma, expiry, model="black76"):
        super().__init__("Put Ratio Spread 1x2")

        long = OptionFactory.create(model, F, K1, r, sigma, expiry, "put", datetime.today())
        short = OptionFactory.create(model, F, K2, r, sigma, expiry, "put", datetime.today())

        self.add_leg(long, 1, f"Long Put K={K1}")
        self.add_leg(short, -2, f"Short 2 Puts K={K2}")
