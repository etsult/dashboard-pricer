import pandas as pd
from datetime import datetime
from pricer.models.black76 import ForwardOption


class Strategy:
    def __init__(self, name="Custom Strategy"):
        self.name = name
        self.legs = []  # list of tuples: (ForwardOption, qty, label)

    def add_leg(self, option, qty=1, label=None):
        label = label or f"{option.option_type.capitalize()} K={option.K}"
        self.legs.append((option, qty, label))

    # --- total greeks ---
    def price(self): return sum(q * o.price() for o, q, _ in self.legs)
    def delta(self): return sum(q * o.delta() for o, q, _ in self.legs)
    def gamma(self): return sum(q * o.gamma() for o, q, _ in self.legs)
    def vega(self):  return sum(q * o.vega()  for o, q, _ in self.legs)
    def theta(self): return sum(q * o.theta() for o, q, _ in self.legs)
    def rho(self):   return sum(q * o.rho()   for o, q, _ in self.legs)

    # --- Greeks vs F ---
    def greeks_vs_forward(self, F_values):
        results = {"F": F_values}
        greeks = ["price", "delta", "gamma", "vega", "theta", "rho"]
        for greek in greeks:
            results[greek] = []
            for F in F_values:
                total = sum(
                    q * getattr(ForwardOption(
                        F, o.K, o.r, o.sigma, o.expiry, o.option_type, o.valuation_date
                    ), greek)()
                    for o, q, _ in self.legs
                )
                results[greek].append(total)
        return results

    # --- serialization ---
    def to_dataframe(self):
        rows = []
        for opt, qty, label in self.legs:
            rows.append({
                "Type": opt.option_type.capitalize(),
                "Strike": opt.K,
                "Expiry": opt.expiry.strftime("%Y-%m-%d"),
                "Qty": qty,
                "σ": opt.sigma,
                "r": opt.r,
                "F": opt.F,
                "Valuation Date": opt.valuation_date.strftime("%Y-%m-%d"),
            })
        return pd.DataFrame(rows)

    @classmethod
    def from_dataframe(cls, df, name="Custom Strategy"):
        strat = cls(name)
        for _, row in df.iterrows():
            expiry = datetime.strptime(str(row["Expiry"]), "%Y-%m-%d")
            valuation = datetime.strptime(str(row["Valuation Date"]), "%Y-%m-%d")
            opt = ForwardOption(
                F=row["F"],
                K=row["Strike"],
                r=row["r"],
                sigma=row["σ"],
                expiry=expiry,
                option_type=row["Type"].lower(),
                valuation_date=valuation,
            )
            strat.add_leg(opt, qty=row["Qty"])
        return strat
