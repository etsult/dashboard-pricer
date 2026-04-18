"""
Vectorized IR option book engine.

Prices and Greeks for a Book of IRPosition objects.
Uses Bachelier (normal vol) + ZABR smile + two-curve discounting.

All computation is pure numpy — no QuantLib at pricing time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr

from market_data.curves.rate_curve import RateCurve
from .indexes import INDEX_CATALOG, index_forward_rate
from .instruments import IRPosition, Book
from .zabr import zabr_normal_vol


# ── Bachelier price + Greeks (vectorized) ─────────────────────────────────────

def _bachelier(
    F: np.ndarray,
    K: np.ndarray,
    sigma_n: np.ndarray,
    tau: np.ndarray,
    is_call: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (price, delta, gamma, vega, theta) per unit annuity.
    price is the undiscounted option value.
    """
    tau = np.clip(tau, 1e-8, None)
    vol_t = sigma_n * np.sqrt(tau)
    vol_t = np.clip(vol_t, 1e-12, None)
    d = (F - K) / vol_t

    pdf = np.exp(-0.5 * d**2) / np.sqrt(2 * np.pi)
    cdf = ndtr(d)

    price = np.where(
        is_call,
        (F - K) * cdf + vol_t * pdf,
        (K - F) * ndtr(-d) + vol_t * pdf,
    )
    delta = np.where(is_call, cdf, cdf - 1.0)
    gamma = pdf / vol_t
    vega  = np.sqrt(tau) * pdf          # ∂price/∂sigma_n
    theta = -0.5 * sigma_n * pdf / np.sqrt(tau)

    return price, delta, gamma, vega, theta


# ── Per-position pricer ───────────────────────────────────────────────────────

class BookEngine:
    """
    Price and compute Greeks for a Book of IR option positions.

    Parameters
    ----------
    curve      : RateCurve built from the discount curve (OIS)
    book       : Book of IRPosition objects
    use_zabr   : if True, use ZABR vol smile per caplet (slower but realistic)
    """

    def __init__(self, curve: RateCurve, book: Book, use_zabr: bool = False):
        self.curve    = curve
        self.book     = book
        self.use_zabr = use_zabr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def price_book(self) -> pd.DataFrame:
        """Price all positions. Returns a DataFrame with one row per position."""
        rows = []
        for pos in self.book.positions:
            row = self._price_position(pos)
            rows.append(row)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def greeks_book(self, bump_bp: float = 1.0) -> pd.DataFrame:
        """
        Key-rate DV01 + vega per position.

        bump_bp : parallel shift in bps used to compute DV01
        """
        rows = []
        bump = bump_bp / 10_000.0
        curve_up   = self.curve.shifted(+bump_bp)
        curve_dn   = self.curve.shifted(-bump_bp)

        for pos in self.book.positions:
            pv    = self._price_position(pos, self.curve)["pv"]
            pv_up = self._price_position(pos, curve_up)["pv"]
            pv_dn = self._price_position(pos, curve_dn)["pv"]
            dv01  = (pv_up - pv_dn) / 2.0  # DV01 = ΔPV per 1bp

            # Vega: bump sigma_n by 1bp and reprice
            pos_vega = IRPosition(
                instrument=pos.instrument,
                index_key=pos.index_key,
                notional=pos.notional,
                strike=pos.strike,
                expiry_y=pos.expiry_y,
                tenor_y=pos.tenor_y,
                sigma_n=pos.sigma_n + 1e-4,  # +1bp normal vol
                direction=pos.direction,
                start_y=pos.start_y,
                reset_lag_y=pos.reset_lag_y,
                pay_lag_y=pos.pay_lag_y,
            )
            pv_vega = self._price_position(pos_vega, self.curve)["pv"]
            vega_1bp = pv_vega - pv

            rows.append({
                "label":    pos.label,
                "PV ($)":   round(pv, 0),
                "DV01 ($)": round(dv01, 0),
                "Vega/bp ($)": round(vega_1bp, 0),
                "direction": pos.direction,
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def scenario_pnl(self, shifts_bp: list[float]) -> pd.DataFrame:
        """
        P&L per position for a list of parallel shifts (in bps).
        Returns a DataFrame: rows=positions, cols=shifts.
        """
        base_pvs = [self._price_position(p)["pv"] for p in self.book.positions]
        labels   = [p.label for p in self.book.positions]

        results = {"Position": labels}
        for bp in shifts_bp:
            c = self.curve.shifted(bp)
            pvs = [self._price_position(p, c)["pv"] for p in self.book.positions]
            results[f"{bp:+.0f}bp"] = [round(v - b, 0) for v, b in zip(pvs, base_pvs)]
        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _price_position(self, pos: IRPosition, curve: RateCurve | None = None) -> dict:
        curve = curve or self.curve
        if pos.instrument in ("cap", "floor"):
            return self._price_cap_floor(pos, curve)
        return self._price_swaption(pos, curve)

    def _price_cap_floor(self, pos: IRPosition, curve: RateCurve) -> dict:
        """Price a cap or floor as a strip of caplets/floorlets."""
        idx    = INDEX_CATALOG[pos.index_key]
        freq   = 1.0 / idx["reset_freq"]          # period in years
        is_cap = pos.instrument == "cap"

        T_start = pos.start_y
        T_end   = T_start + pos.tenor_y
        pay_dates = np.arange(T_start + freq, T_end + 1e-9, freq)

        total_pv = 0.0
        for T_pay in pay_dates:
            T_reset = T_pay - freq + pos.reset_lag_y
            tau_opt = max(T_reset - 0.0, 1e-6)   # time to fixing
            T_pay_adj = T_pay + pos.pay_lag_y

            # Forward rate on index curve (OIS + basis)
            ois_z = curve.zero_rate(T_reset)
            f_ois = curve.zero_rate(T_pay_adj)
            basis = idx["basis_bps"] / 10_000.0
            df1   = np.exp(-(ois_z + basis) * T_reset)
            df2   = np.exp(-(f_ois  + basis) * T_pay_adj)
            F_fwd = (df1 / df2 - 1.0) / freq

            df_pay = curve.discount_factor(T_pay_adj)

            # Vol: flat or ZABR
            sigma = pos.sigma_n
            if self.use_zabr:
                from .zabr import zabr_atm_vol
                ccy = idx["ccy"]
                alpha, nu, rho = zabr_atm_vol(tau_opt, freq, ccy)
                sigma = float(zabr_normal_vol(
                    np.array([F_fwd]), np.array([pos.strike]),
                    np.array([tau_opt]), np.array([alpha]),
                    np.array([nu]), np.array([rho]),
                )[0])

            price, *_ = _bachelier(
                np.array([F_fwd]), np.array([pos.strike]),
                np.array([sigma]),  np.array([tau_opt]),
                np.array([is_cap]),
            )
            caplet_pv = float(price[0]) * freq * df_pay * pos.notional
            total_pv += caplet_pv

        pv = total_pv * pos.direction
        atm_fwd = curve.forward_rate(T_start + 1e-4, T_start + freq)
        return {
            "label":      pos.label,
            "instrument": pos.instrument,
            "index":      idx["label"],
            "expiry_y":   pos.expiry_y,
            "tenor_y":    pos.tenor_y,
            "strike_pct": pos.strike * 100,
            "atm_fwd_pct": atm_fwd * 100,
            "sigma_bps":  pos.sigma_n * 10_000,
            "pv":         pv,
            "direction":  pos.direction,
        }

    def _price_swaption(self, pos: IRPosition, curve: RateCurve) -> dict:
        """Price a payer or receiver swaption (Bachelier on par swap rate)."""
        idx    = INDEX_CATALOG[pos.index_key]
        freq   = 1.0 / idx["reset_freq"]
        is_payer = pos.instrument == "payer_swaption"

        T_exp = pos.expiry_y + pos.start_y
        T_end = T_exp + pos.tenor_y

        # OIS discount + index forward for two-curve par rate
        basis = idx["basis_bps"] / 10_000.0
        pay_dates = np.arange(T_exp + freq, T_end + 1e-9, freq)
        ann = freq * sum(curve.discount_factor(t) for t in pay_dates)

        # Forward par swap rate via index curve
        z_exp = curve.zero_rate(T_exp)
        z_end = curve.zero_rate(T_end)
        df_exp_idx = np.exp(-(z_exp + basis) * T_exp)
        df_end_idx = np.exp(-(z_end + basis) * T_end)
        S0 = (df_exp_idx - df_end_idx) / ann if ann > 1e-12 else 0.0

        tau = max(T_exp, 1e-6)
        sigma = pos.sigma_n
        if self.use_zabr:
            from .zabr import zabr_atm_vol
            ccy = idx["ccy"]
            alpha, nu, rho = zabr_atm_vol(tau, pos.tenor_y, ccy)
            sigma = float(zabr_normal_vol(
                np.array([S0]), np.array([pos.strike]),
                np.array([tau]), np.array([alpha]),
                np.array([nu]), np.array([rho]),
            )[0])

        price, delta, gamma, vega, theta = _bachelier(
            np.array([S0]), np.array([pos.strike]),
            np.array([sigma]), np.array([tau]),
            np.array([is_payer]),
        )

        pv = float(price[0]) * ann * pos.notional * pos.direction

        return {
            "label":       pos.label,
            "instrument":  pos.instrument,
            "index":       idx["label"],
            "expiry_y":    pos.expiry_y,
            "tenor_y":     pos.tenor_y,
            "strike_pct":  pos.strike * 100,
            "atm_fwd_pct": S0 * 100,
            "sigma_bps":   sigma * 10_000,
            "annuity":     ann,
            "pv":          pv,
            "direction":   pos.direction,
        }
