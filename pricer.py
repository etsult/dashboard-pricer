import numpy as np
from scipy.stats import norm
from datetime import datetime
import matplotlib.pyplot as plt

class ForwardOption:
    def __init__(self, F, K, r, sigma, expiry, option_type="call", valuation_date=None):
        """
        Black 76 option on a forward/futures contract
        F : forward price
        K : strike
        r : risk-free rate
        sigma : volatility
        expiry : datetime of expiry
        option_type : "call" or "put"
        valuation_date : current date (default = today)
        """
        self.F = F
        self.K = K
        self.r = r
        self.sigma = sigma
        self.expiry = expiry
        self.option_type = option_type.lower()
        self.valuation_date = valuation_date if valuation_date else datetime.today()

    @property
    def tau(self):
        """time to maturity in years"""
        dt = (self.expiry - self.valuation_date).days
        return max(dt,0)/365.0   # act/365 convention (you can change)

    def _d1_d2(self):
        d1 = (np.log(self.F/self.K) + 0.5*self.sigma**2*self.tau) / (self.sigma*np.sqrt(self.tau))
        d2 = d1 - self.sigma*np.sqrt(self.tau)
        return d1, d2

    def price(self):
        d1, d2 = self._d1_d2()
        df = np.exp(-self.r*self.tau)
        if self.option_type == "call":
            return df*(self.F*norm.cdf(d1) - self.K*norm.cdf(d2))
        elif self.option_type == "put":
            return df*(self.K*norm.cdf(-d2) - self.F*norm.cdf(-d1))
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    def delta(self):
        d1, _ = self._d1_d2()
        df = np.exp(-self.r*self.tau)
        return df*(norm.cdf(d1) if self.option_type=="call" else -norm.cdf(-d1))

    def gamma(self):
        d1, _ = self._d1_d2()
        df = np.exp(-self.r*self.tau)
        return df*norm.pdf(d1)/(self.F*self.sigma*np.sqrt(self.tau))

    def vega(self):
        d1, _ = self._d1_d2()
        df = np.exp(-self.r*self.tau)
        return df*self.F*norm.pdf(d1)*np.sqrt(self.tau)

    def theta(self):
        d1, _ = self._d1_d2()
        df = np.exp(-self.r*self.tau)
        first = -(self.F*self.sigma*df*norm.pdf(d1))/(2*np.sqrt(self.tau))
        return first - self.r*self.price()

    def rho(self):
        return -self.tau * self.price()


class Strategy:
    def __init__(self, name="Custom Strategy"):
        self.name = name
        self.legs = []  # liste de tuples (option, qty, label)

    def add_leg(self, option, qty=1, label=None):
        label = label if label else f"{option.option_type.capitalize()} K={option.K} T={option.tau}"
        self.legs.append((option, qty, label))

    # --- compute total greeks ---
    def price(self): return sum(qty * leg.price() for leg, qty, _ in self.legs)
    def delta(self): return sum(qty * leg.delta() for leg, qty, _ in self.legs)
    def gamma(self): return sum(qty * leg.gamma() for leg, qty, _ in self.legs)
    def vega(self):  return sum(qty * leg.vega()  for leg, qty, _ in self.legs)
    def theta(self): return sum(qty * leg.theta() for leg, qty, _ in self.legs)
    def rho(self):   return sum(qty * leg.rho()   for leg, qty, _ in self.legs)

    # --- plot greek by varying forward ---
    def plot_greek_vs_forward(self, greek="delta", Fmin=None, Fmax=None, n=200):
    # auto-scale based on legs if not given
        if Fmin is None:
            Fmin = 0
        if Fmax is None:
            Fmax = 2 * max(opt.F for opt, _, _ in self.legs)

        F_values = np.linspace(Fmin, Fmax, n)

        plt.figure(figsize=(8,5))

        # per-leg curves
        for opt, qty, label in self.legs:
            vals = []
            for F in F_values:
                o = ForwardOption(F, opt.K, opt.r, opt.sigma, opt.expiry, opt.option_type, opt.valuation_date)
                vals.append(qty * getattr(o, greek)())
            plt.plot(F_values, vals, '--', label=f"{label} ({greek})")

        # total curve
        total_vals = []
        for F in F_values:
            val = 0
            for opt, qty, _ in self.legs:
                o = ForwardOption(F, opt.K, opt.r, opt.sigma, opt.expiry, opt.option_type, opt.valuation_date)
                val += qty * getattr(o, greek)()
            total_vals.append(val)
        plt.plot(F_values, total_vals, 'k', lw=2, label=f"Total {greek.upper()}")

        plt.title(f"{self.name} - {greek.upper()} vs Forward")
        plt.xlabel("Forward F")
        plt.ylabel(greek.capitalize())
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_all_vs_forward(self, Fmin=None, Fmax=None, n=200):
        if Fmin is None:
            Fmin = 1e-6
        if Fmax is None:
            Fmax = 2 * max(opt.F for opt, _, _ in self.legs)

        F_values = np.linspace(Fmin, Fmax, n)
        greeks = ["price", "delta", "gamma", "vega", "theta", "rho"]

        fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
        axes = axes.flatten()

        for ax, greek in zip(axes, greeks):
            # total curve
            total_vals = []
            for F in F_values:
                val = 0
                for opt, qty, _ in self.legs:
                    o = ForwardOption(F, opt.K, opt.r, opt.sigma, opt.expiry, opt.option_type, opt.valuation_date)
                    val += qty * getattr(o, greek)()
                total_vals.append(val)
            ax.plot(F_values, total_vals, 'k', lw=2, label=f"Total {greek.capitalize()}")

            # per-leg curves
            for opt, qty, label in self.legs:
                vals = []
                for F in F_values:
                    o = ForwardOption(F, opt.K, opt.r, opt.sigma, opt.expiry, opt.option_type, opt.valuation_date)
                    vals.append(qty * getattr(o, greek)())
                ax.plot(F_values, vals, '--', label=f"{label} ({greek})")

            ax.set_ylabel(greek.capitalize())
            ax.legend(fontsize=8)
            ax.grid(True)

        for ax in axes[-2:]:
            ax.set_xlabel("Forward F")

        plt.suptitle(f"{self.name} - Greeks and Price vs Forward", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()



class PutCalendar(Strategy):
    def __init__(self, F, K, r, sigma, expiry_short, expiry_long, valuation_date=None):
        super().__init__("Put Calendar Spread")

        # long maturity put
        self.add_leg(
            ForwardOption(F, K, r, sigma, expiry_long, "put", valuation_date),
            qty=1,
            label=f"Long Put {expiry_long.date()}"
        )

        # short maturity put
        self.add_leg(
            ForwardOption(F, K, r, sigma, expiry_short, "put", valuation_date),
            qty=-1,
            label=f"Short Put {expiry_short.date()}"
        )


class CallRatio(Strategy):
    def __init__(self, F, K, r, sigma, expiry, spread=10, ratio=2, valuation_date=None):
        super().__init__("Call Ratio Spread")

        # Long call at strike K
        self.add_leg(
            ForwardOption(F, K, r, sigma, expiry, "call", valuation_date),
            qty=1,
            label=f"Long Call K={K}, Exp {expiry.date()}"
        )

        # Short ratio calls at strike K+spread
        self.add_leg(
            ForwardOption(F, K+spread, r, sigma, expiry, "call", valuation_date),
            qty=-ratio,
            label=f"Short {ratio} Calls K={K+spread}, Exp {expiry.date()}"
        )

class CustomStrategy(Strategy):
    def __init__(self, name="Custom Strategy"):
        super().__init__(name)

    def add_option(self, F, K, r, sigma, expiry, option_type="call", qty=1, valuation_date=None, label=None):
        opt = ForwardOption(F, K, r, sigma, expiry, option_type, valuation_date)
        label = label if label else f"{option_type.capitalize()} K={K}, Exp {expiry.date()}"
        self.add_leg(opt, qty=qty, label=label)


from datetime import datetime

today = datetime.now()
exp1 = datetime(2025,11,19)
exp2 = datetime(2025,11,19)

# Custom Strategy
custom = CustomStrategy("Custom")
ref = 12630
custom.add_option(F=ref, K=12500, r=0.02, sigma=0.12, expiry=exp1, option_type="put", qty=1, valuation_date=today)
custom.add_option(F=ref, K=11000, r=0.02, sigma=0.12, expiry=exp2, option_type="put", qty=1, valuation_date=today)

print("Custom Strategy Price:", custom.price())
print("Custom Strategy Delta:", custom.delta())

# Plot all Greeks and Price on one figure
custom.plot_all_vs_forward()