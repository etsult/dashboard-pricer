# pricer/strategies/vanilla_structures/calendars.py

from __future__ import annotations
from datetime import datetime
from pricer.strategies.strategy import Strategy
from pricer.models.factory import OptionFactory


class CallCalendar(Strategy):
    def __init__(self, F, K, r, sigma1, sigma2, expiry1, expiry2, qty=1.0, model="black76"):
        super().__init__("Call Calendar")

        short = OptionFactory.create(model, F=F, K=K, r=r, sigma=sigma1,
                                     expiry=expiry1, option_type="call",
                                     valuation_date=datetime.today())

        long  = OptionFactory.create(model, F=F, K=K, r=r, sigma=sigma2,
                                     expiry=expiry2, option_type="call",
                                     valuation_date=datetime.today())

        self.add_leg(short, -qty, f"Short Call {expiry1}")
        self.add_leg(long, qty, f"Long Call {expiry2}")


class PutCalendar(Strategy):
    def __init__(self, F, K, r, sigma1, sigma2, expiry1, expiry2, qty=1.0, model="black76"):
        super().__init__("Put Calendar")

        short = OptionFactory.create(model, F=F, K=K, r=r, sigma=sigma1,
                                     expiry=expiry1, option_type="put",
                                     valuation_date=datetime.today())

        long  = OptionFactory.create(model, F=F, K=K, r=r, sigma=sigma2,
                                     expiry=expiry2, option_type="put",
                                     valuation_date=datetime.today())

        self.add_leg(short, -qty, f"Short Put {expiry1}")
        self.add_leg(long, qty, f"Long Put {expiry2}")
