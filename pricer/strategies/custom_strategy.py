from pricer.strategies.strategy import Strategy
from pricer.models.black76 import ForwardOption


class CustomStrategy(Strategy):
    def __init__(self, name="Custom Strategy"):
        super().__init__(name)

    def add_option(self, F, K, r, sigma, expiry, option_type="call", qty=1, valuation_date=None, label=None):
        opt = ForwardOption(F, K, r, sigma, expiry, option_type, valuation_date)
        label = label if label else f"{option_type.capitalize()} K={K}, Exp {expiry.date()}"
        self.add_leg(opt, qty=qty, label=label)
