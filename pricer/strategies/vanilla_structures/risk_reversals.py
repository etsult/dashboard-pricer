# pricer/strategies/vanilla_structures/risk_reversals.py

from __future__ import annotations
from datetime import datetime
from pricer.strategies.strategy import Strategy
from pricer.models.factory import OptionFactory


class RiskReversal(Strategy):
    def __init__(self, F, K_call, K_put, r, sigma, expiry, qty=1.0, model="black76"):
        super().__init__("Risk Reversal")

        long_call = OptionFactory.create(model, F=F, K=K_call, r=r, sigma=sigma,
                                         expiry=expiry, option_type="call",
                                         valuation_date=datetime.today())

        short_put = OptionFactory.create(model, F=F, K=K_put, r=r, sigma=sigma,
                                         expiry=expiry, option_type="put",
                                         valuation_date=datetime.today())

        self.add_leg(long_call, qty, f"Long Call K={K_call}")
        self.add_leg(short_put, -qty, f"Short Put K={K_put}")
