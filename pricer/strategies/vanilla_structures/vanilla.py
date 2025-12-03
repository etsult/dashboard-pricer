# pricer/strategies/vanilla_structures/call_put.py

from __future__ import annotations
from datetime import datetime
from pricer.strategies.strategy import Strategy
from pricer.models.factory import OptionFactory


class Call(Strategy):
    def __init__(self, F: float, K: float, r: float, sigma: float, expiry, qty: float = 1.0, model: str = "black76"):
        super().__init__("Naked Call")

        call = OptionFactory.create(
            model,
            F=F, K=K, r=r, sigma=sigma,
            expiry=expiry, option_type="call",
            valuation_date=datetime.today(),
        )

        self.add_leg(call, qty, f"Call K={K}")


class Put(Strategy):
    def __init__(self, F: float, K: float, r: float, sigma: float, expiry, qty: float = 1.0, model: str = "black76"):
        super().__init__("Naked Put")

        put = OptionFactory.create(
            model,
            F=F, K=K, r=r, sigma=sigma,
            expiry=expiry, option_type="put",
            valuation_date=datetime.today(),
        )

        self.add_leg(put, qty, f"Put K={K}")
