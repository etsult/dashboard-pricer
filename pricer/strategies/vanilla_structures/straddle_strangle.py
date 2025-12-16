# pricer/strategies/vanilla_structures/straddle_strangle.py

from __future__ import annotations
from datetime import datetime
from pricer.strategies.strategy import Strategy
from pricer.models.factory import OptionFactory


class Straddle(Strategy):
    def __init__(self, F, K, r, sigma, expiry, qty=1.0, model="black76"):
        super().__init__("Straddle")

        call = OptionFactory.create(model, F=F, K=K, r=r, sigma=sigma,
                                    expiry=expiry, option_type="call",
                                    valuation_date=datetime.today())

        put = OptionFactory.create(model, F=F, K=K, r=r, sigma=sigma,
                                   expiry=expiry, option_type="put",
                                   valuation_date=datetime.today())

        self.add_leg(call, qty, f"Long Call K={K}")
        self.add_leg(put, qty, f"Long Put K={K}")


class Strangle(Strategy):
    def __init__(self, F, K_center, width, r, sigma, expiry, qty=1.0, model="black76"):
        super().__init__("Strangle")

        Kp = K_center - width
        Kc = K_center + width

        long_put = OptionFactory.create(model, F=F, K=Kp, r=r, sigma=sigma,
                                        expiry=expiry, option_type="put",
                                        valuation_date=datetime.today())

        long_call = OptionFactory.create(model, F=F, K=Kc, r=r, sigma=sigma,
                                         expiry=expiry, option_type="call",
                                         valuation_date=datetime.today())

        self.add_leg(long_put, qty, f"Long Put K={Kp}")
        self.add_leg(long_call, qty, f"Long Call K={Kc}")
