# pricer/strategies/vanilla_structures/butterflies.py

from __future__ import annotations
from datetime import datetime
from pricer.strategies.strategy import Strategy
from pricer.models.factory import OptionFactory


class CallButterfly(Strategy):
    def __init__(self, F, K_center, width, r, sigma, expiry, qty=1.0, model="black76"):
        super().__init__("Call Butterfly")

        K1 = K_center - width
        K2 = K_center
        K3 = K_center + width

        call1 = OptionFactory.create(model, F=F, K=K1, r=r, sigma=sigma,
                                     expiry=expiry, option_type="call",
                                     valuation_date=datetime.today())

        call2 = OptionFactory.create(model, F=F, K=K2, r=r, sigma=sigma,
                                     expiry=expiry, option_type="call",
                                     valuation_date=datetime.today())

        call3 = OptionFactory.create(model, F=F, K=K3, r=r, sigma=sigma,
                                     expiry=expiry, option_type="call",
                                     valuation_date=datetime.today())

        self.add_leg(call1, qty, f"Long Call K={K1}")
        self.add_leg(call2, -2 * qty, f"Short 2x Call K={K2}")
        self.add_leg(call3, qty, f"Long Call K={K3}")


class PutButterfly(Strategy):
    def __init__(self, F, K_center, width, r, sigma, expiry, qty=1.0, model="black76"):
        super().__init__("Put Butterfly")

        K1 = K_center - width
        K2 = K_center
        K3 = K_center + width

        put1 = OptionFactory.create(model, F=F, K=K1, r=r, sigma=sigma,
                                    expiry=expiry, option_type="put",
                                    valuation_date=datetime.today())

        put2 = OptionFactory.create(model, F=F, K=K2, r=r, sigma=sigma,
                                    expiry=expiry, option_type="put",
                                    valuation_date=datetime.today())

        put3 = OptionFactory.create(model, F=F, K=K3, r=r, sigma=sigma,
                                    expiry=expiry, option_type="put",
                                    valuation_date=datetime.today())

        self.add_leg(put1, qty, f"Long Put K={K1}")
        self.add_leg(put2, -2 * qty, f"Short 2x Put K={K2}")
        self.add_leg(put3, qty, f"Long Put K={K3}")
