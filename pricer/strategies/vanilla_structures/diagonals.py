from __future__ import annotations
from datetime import datetime
from pricer.strategies.strategy import Strategy
from pricer.models.factory import OptionFactory


class CallDiagonal(Strategy):
    def __init__(self, F, K1, K2, r, sigma1, sigma2, expiry1, expiry2, model="black76"):
        super().__init__("Call Diagonal Spread")

        short = OptionFactory.create(model, F, K1, r, sigma1, expiry1, "call", datetime.today())
        long  = OptionFactory.create(model, F, K2, r, sigma2, expiry2, "call", datetime.today())

        self.add_leg(short, -1, f"Short Call K={K1}, Exp {expiry1}")
        self.add_leg(long,  1, f"Long Call K={K2},  Exp {expiry2}")


class PutDiagonal(Strategy):
    def __init__(self, F, K1, K2, r, sigma1, sigma2, expiry1, expiry2, model="black76"):
        super().__init__("Put Diagonal Spread")

        short = OptionFactory.create(model, F, K1, r, sigma1, expiry1, "put", datetime.today())
        long  = OptionFactory.create(model, F, K2, r, sigma2, expiry2, "put", datetime.today())

        self.add_leg(short, -1, f"Short Put K={K1}, Exp {expiry1}")
        self.add_leg(long,  1, f"Long Put K={K2},  Exp {expiry2}")
