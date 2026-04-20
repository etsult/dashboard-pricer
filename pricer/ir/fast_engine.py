"""
Fast vectorized IR book pricer.

Prices N positions simultaneously using pure numpy — no Python loops over
positions at pricing time.  Suitable for books of 10k–500k positions.

Approximations
──────────────
  Swaption annuity : exact vectorized coupon sum  Σ df_OIS(tᵢ) × Δtᵢ
  Cap/floor strip  : exact vectorized 2-D caplet grid
  Greeks (delta, vega) : Bachelier analytical
  DV01 : delta × annuity × notional × 1e-4 (1bp linear approx)

Multi-curve
───────────
Accepts either a RateCurve (auto-wrapped as CurveSet with flat-basis fallback)
or a CurveSet with per-index projection curves.
  discounting : OIS curve for all df(tᵢ) in annuity / caplet PV
  forwarding  : projection curve for F_swap and F_caplet
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr

from market_data.curves.rate_curve import RateCurve
from market_data.curves.curve_set import CurveSet
from .indexes import INDEX_CATALOG
from .instruments import Book

_SQRT_2PI = np.sqrt(2 * np.pi)

# ── Dense pre-computed grid for O(1) vectorized lookups ──────────────────────

class _CurveSetGrid:
    """
    Pre-computed dense grids for both OIS and per-index projection curves.
    Vectorized lookups via np.interp — no Python loops at pricing time.
    """

    def __init__(self, cs: CurveSet, n: int = 20_000):
        self._T = np.linspace(1e-4, 31.0, n)

        # OIS grids
        z_ois        = np.interp(self._T, cs.ois._tenors, cs.ois._zero_rates)
        self._z_ois  = z_ois
        self._df_ois = np.exp(-z_ois * self._T)

        # Per-index projection grids (only for indexes with real projection curves)
        self._proj: dict[str, np.ndarray] = {}
        for key, curve in cs.projections.items():
            z = np.interp(self._T, curve._tenors, curve._zero_rates)
            self._proj[key] = np.exp(-z * self._T)

        # Flat-basis fallback for indexes without a projection curve
        self._basis = {k: v["basis_bps"] / 10_000.0 for k, v in INDEX_CATALOG.items()}

    # ── OIS (discounting) ─────────────────────────────────────────────────────

    def df(self, T: np.ndarray) -> np.ndarray:
        """OIS discount factor — use for discounting all cash flows."""
        return np.interp(T, self._T, self._df_ois)

    def zero(self, T: np.ndarray) -> np.ndarray:
        """OIS zero rate."""
        return np.interp(T, self._T, self._z_ois)

    # ── Projection (forwarding) ───────────────────────────────────────────────

    def proj_df(self, index_key: str, T: np.ndarray) -> np.ndarray:
        """Projection curve DF for a single index."""
        T = np.asarray(T, dtype=float)
        if index_key in self._proj:
            return np.interp(T, self._T, self._proj[index_key])
        basis = self._basis.get(index_key, 0.0)
        return self.df(T) * np.exp(-basis * T)

    def proj_df_grouped(self, index_keys: list[str], T: np.ndarray) -> np.ndarray:
        """
        Vectorized projection DFs for a mixed list of index keys.
        Groups by unique key → one np.interp call per unique index (not per position).
        """
        result = np.empty_like(T, dtype=float)
        for key in set(index_keys):
            mask = np.array([k == key for k in index_keys])
            result[mask] = self.proj_df(key, T[mask])
        return result

    def proj_zero_2d(self, index_keys: list[str], T_2d: np.ndarray) -> np.ndarray:
        """
        Projection zero rates for a 2-D time array [n_positions × max_caplets].
        index_keys has length n_positions; one key per row of T_2d.
        """
        result = np.empty_like(T_2d, dtype=float)
        for key in set(index_keys):
            m = np.array([k == key for k in index_keys])
            if key in self._proj:
                df = np.interp(T_2d[m, :], self._T, self._proj[key])
                result[m, :] = (
                    -np.log(np.clip(df, 1e-12, None))
                    / np.clip(T_2d[m, :], 1e-9, None)
                )
            else:
                result[m, :] = self.zero(T_2d[m, :]) + self._basis.get(key, 0.0)
        return result


# ── Bachelier (fully vectorized) ──────────────────────────────────────────────

def _bach_price_greeks(F, K, sigma_n, tau, is_call):
    tau     = np.clip(tau, 1e-8, None)
    vol_t   = np.clip(sigma_n * np.sqrt(tau), 1e-12, None)
    d       = (F - K) / vol_t
    pdf     = np.exp(-0.5 * d ** 2) / _SQRT_2PI
    cdf     = ndtr(d)
    price   = np.where(is_call,
                       (F - K) * cdf + vol_t * pdf,
                       (K - F) * ndtr(-d) + vol_t * pdf)
    delta   = np.where(is_call, cdf, cdf - 1.0)
    vega    = np.sqrt(tau) * pdf
    gamma   = pdf / vol_t
    return price, delta, vega, gamma


# ── Main fast engine ──────────────────────────────────────────────────────────

class FastBookEngine:
    """
    Vectorized pricer for large IR books (options + swaps).

    Usage
    -----
    eng = FastBookEngine(curve_or_set, book)
    pv_df   = eng.price_book()      # DataFrame with PV per position
    risk_df = eng.risk_book()       # DV01, vega, delta per position
    agg     = eng.aggregate_risk()  # book-level aggregated risk tables

    Parameters
    ----------
    curve_or_set : RateCurve | CurveSet
        RateCurve is auto-wrapped as CurveSet (flat basis fallback).
        Pass a CurveSet with per-index projection curves for true dual-curve.
    book : Book
    """

    def __init__(self, curve_or_set: "RateCurve | CurveSet", book: Book):
        self.book = book
        if isinstance(curve_or_set, RateCurve):
            cs = CurveSet.from_single_curve(curve_or_set)
        else:
            cs = curve_or_set
        self._cs   = cs
        self._grid = _CurveSetGrid(cs)

    # ── Array extraction ──────────────────────────────────────────────────────

    def _extract_arrays(self):
        pos = self.book.positions
        n   = len(pos)

        instr    = np.array([p.instrument for p in pos])
        idx_keys = [p.index_key  for p in pos]
        notional = np.array([p.notional   for p in pos], dtype=float)
        strike   = np.array([p.strike     for p in pos], dtype=float)
        expiry   = np.array([p.expiry_y   for p in pos], dtype=float)
        tenor    = np.array([p.tenor_y    for p in pos], dtype=float)
        sigma_n  = np.array([p.sigma_n    for p in pos], dtype=float)
        direction= np.array([p.direction  for p in pos], dtype=float)

        freq = np.array([1.0 / INDEX_CATALOG[k]["reset_freq"] for k in idx_keys])

        is_swaption  = np.isin(instr, ["payer_swaption", "receiver_swaption"])
        is_payer_sw  = instr == "payer_swaption"
        is_call_cap  = instr == "cap"
        is_cap_floor = np.isin(instr, ["cap", "floor"])
        is_irs       = np.isin(instr, ["payer_irs", "receiver_irs"])
        is_payer_irs = instr == "payer_irs"

        return {
            "n": n, "instr": instr, "idx_keys": idx_keys,
            "notional": notional, "strike": strike,
            "expiry": expiry, "tenor": tenor,
            "sigma_n": sigma_n, "direction": direction, "freq": freq,
            "is_swaption": is_swaption, "is_payer_sw": is_payer_sw,
            "is_call_cap": is_call_cap, "is_cap_floor": is_cap_floor,
            "is_irs": is_irs, "is_payer_irs": is_payer_irs,
        }

    # ── Shared: exact annuity (OIS-discounted, vectorized 2-D coupon sum) ─────

    @staticmethod
    def _exact_annuity(g: _CurveSetGrid,
                       T_start: np.ndarray,
                       tenor:   np.ndarray,
                       freq:    np.ndarray) -> np.ndarray:
        """
        Exact annuity for a subbook: Σ df_OIS(T_start + j×freq) × freq, j=1..n_c
        Returns shape (len(T_start),).
        """
        n_c    = np.maximum(np.round(tenor / freq).astype(int), 1)
        max_nc = int(n_c.max())
        cl_idx = np.arange(1, max_nc + 1)
        T_pay  = T_start[:, None] + cl_idx[None, :] * freq[:, None]  # [m, max_nc]
        valid  = cl_idx[None, :] <= n_c[:, None]
        ann    = (g.df(T_pay) * freq[:, None] * valid).sum(axis=1)
        return np.clip(ann, 1e-10, None)

    # ── price_book ────────────────────────────────────────────────────────────

    def price_book(self) -> pd.DataFrame:
        """Return per-position PV. O(N) vectorized."""
        if self.book.is_empty():
            return pd.DataFrame()

        a = self._extract_arrays()
        g = self._grid
        n = a["n"]

        pv      = np.zeros(n)
        F_fwd   = np.zeros(n)
        annuity = np.zeros(n)

        # ── Swaptions ─────────────────────────────────────────────────────────
        sw = a["is_swaption"]
        if sw.any():
            T_exp    = a["expiry"][sw]
            T_end    = T_exp + a["tenor"][sw]
            freq_sw  = a["freq"][sw]

            # Exact annuity: OIS-discounted coupon sum
            ann_sw  = self._exact_annuity(g, T_exp, a["tenor"][sw], freq_sw)

            # Forward swap rate: projection curve for DFs, OIS for annuity
            keys_sw  = [a["idx_keys"][i] for i in np.where(sw)[0]]
            df_exp_p = g.proj_df_grouped(keys_sw, T_exp)
            df_end_p = g.proj_df_grouped(keys_sw, T_end)
            F_sw     = (df_exp_p - df_end_p) / ann_sw

            price_sw, _, _, _ = _bach_price_greeks(
                F_sw, a["strike"][sw], a["sigma_n"][sw],
                T_exp, a["is_payer_sw"][sw],
            )
            pv[sw]      = price_sw * ann_sw * a["notional"][sw] * a["direction"][sw]
            F_fwd[sw]   = F_sw
            annuity[sw] = ann_sw

        # ── Caps / Floors — exact 2-D caplet strip ────────────────────────────
        cf = a["is_cap_floor"]
        if cf.any():
            mat      = a["tenor"][cf]
            frq      = a["freq"][cf]
            keys_cf  = [a["idx_keys"][i] for i in np.where(cf)[0]]

            n_cl     = np.maximum(np.round(mat / frq).astype(int), 1)
            max_cl   = int(n_cl.max())
            cl_idx   = np.arange(1, max_cl + 1)

            T_r2d    = cl_idx[None, :] * frq[:, None]     # reset times
            T_p2d    = T_r2d + frq[:, None]               # pay   times
            valid    = cl_idx[None, :] <= n_cl[:, None]

            # Forward rate on projection curve: F = (df_proj(T_r) / df_proj(T_p) - 1) / freq
            z1_2d  = g.proj_zero_2d(keys_cf, T_r2d)
            z2_2d  = g.proj_zero_2d(keys_cf, T_p2d)
            df1_2d = np.exp(-z1_2d * T_r2d)
            df2_2d = np.exp(-z2_2d * T_p2d)
            F_2d   = np.clip(
                (df1_2d / np.clip(df2_2d, 1e-12, None) - 1.0) / frq[:, None],
                1e-5, None,
            )

            df_pay2d = g.df(T_p2d)                        # OIS discounting
            tau_2d   = np.clip(T_r2d, 1e-6, None)

            K_2d     = a["strike"][cf][:, None] + np.zeros(max_cl)
            sig_2d   = a["sigma_n"][cf][:, None] + np.zeros(max_cl)
            call_2d  = a["is_call_cap"][cf][:, None] + np.zeros(max_cl, dtype=bool)
            price_2d, _, _, _ = _bach_price_greeks(F_2d, K_2d, sig_2d, tau_2d, call_2d)

            pv_strip = (price_2d * frq[:, None] * df_pay2d * valid).sum(axis=1)
            pv[cf]    = pv_strip * a["notional"][cf] * a["direction"][cf]
            F_fwd[cf] = F_2d[:, 0]

        # ── Vanilla IRS (linear, no optionality) ─────────────────────────────
        irs = a["is_irs"]
        if irs.any():
            T_start  = a["expiry"][irs]     # swap start (0 = spot)
            ten_irs  = a["tenor"][irs]
            freq_irs = a["freq"][irs]

            ann_irs  = self._exact_annuity(g, T_start, ten_irs, freq_irs)

            T_end_irs = T_start + ten_irs
            keys_irs  = [a["idx_keys"][i] for i in np.where(irs)[0]]
            df_s_proj = g.proj_df_grouped(keys_irs, T_start)
            df_e_proj = g.proj_df_grouped(keys_irs, T_end_irs)
            F_irs     = (df_s_proj - df_e_proj) / ann_irs

            sign_irs = np.where(a["is_payer_irs"][irs], 1.0, -1.0)
            pv[irs]      = sign_irs * (F_irs - a["strike"][irs]) * ann_irs * a["notional"][irs] * a["direction"][irs]
            F_fwd[irs]   = F_irs
            annuity[irs] = ann_irs

        return pd.DataFrame({
            "label":       [p.label     for p in self.book.positions],
            "instrument":  a["instr"],
            "index_key":   a["idx_keys"],
            "ccy":         [INDEX_CATALOG[k]["ccy"] for k in a["idx_keys"]],
            "expiry_y":    a["expiry"],
            "tenor_y":     a["tenor"],
            "strike_pct":  a["strike"] * 100,
            "atm_pct":     F_fwd * 100,
            "sigma_bps":   a["sigma_n"] * 10_000,
            "notional":    a["notional"],
            "direction":   a["direction"],
            "pv":          pv,
        })

    # ── risk_book ─────────────────────────────────────────────────────────────

    def risk_book(self, bump_bp: float = 1.0) -> pd.DataFrame:
        """
        Per-position DV01, vega, delta using analytical Greeks.
        Much faster than bump-and-reprice for large books.
        """
        if self.book.is_empty():
            return pd.DataFrame()

        a    = self._extract_arrays()
        g    = self._grid
        n    = a["n"]
        bump = bump_bp / 10_000.0

        dv01  = np.zeros(n)
        vega  = np.zeros(n)
        delta = np.zeros(n)
        pv    = np.zeros(n)

        # ── Swaptions ─────────────────────────────────────────────────────────
        sw = a["is_swaption"]
        if sw.any():
            T_exp   = a["expiry"][sw]
            T_end   = T_exp + a["tenor"][sw]
            freq_sw = a["freq"][sw]

            ann_sw  = self._exact_annuity(g, T_exp, a["tenor"][sw], freq_sw)

            keys_sw  = [a["idx_keys"][i] for i in np.where(sw)[0]]
            df_exp_p = g.proj_df_grouped(keys_sw, T_exp)
            df_end_p = g.proj_df_grouped(keys_sw, T_end)
            F_sw     = (df_exp_p - df_end_p) / ann_sw

            price_sw, delta_sw, vega_sw, _ = _bach_price_greeks(
                F_sw, a["strike"][sw], a["sigma_n"][sw],
                T_exp, a["is_payer_sw"][sw],
            )
            scale_sw  = ann_sw * a["notional"][sw] * a["direction"][sw]
            pv[sw]    = price_sw * scale_sw
            delta[sw] = delta_sw * scale_sw
            vega[sw]  = vega_sw  * scale_sw * 1e-4
            dv01[sw]  = delta_sw * scale_sw * bump

        # ── Caps / Floors — exact 2-D caplet strip ────────────────────────────
        cf = a["is_cap_floor"]
        if cf.any():
            mat      = a["tenor"][cf]
            frq      = a["freq"][cf]
            keys_cf  = [a["idx_keys"][i] for i in np.where(cf)[0]]

            n_cl   = np.maximum(np.round(mat / frq).astype(int), 1)
            max_cl = int(n_cl.max())
            cl_idx = np.arange(1, max_cl + 1)

            T_r2d  = cl_idx[None, :] * frq[:, None]
            T_p2d  = T_r2d + frq[:, None]
            valid  = cl_idx[None, :] <= n_cl[:, None]

            z1_2d  = g.proj_zero_2d(keys_cf, T_r2d)
            z2_2d  = g.proj_zero_2d(keys_cf, T_p2d)
            df1_2d = np.exp(-z1_2d * T_r2d)
            df2_2d = np.exp(-z2_2d * T_p2d)
            F_2d   = np.clip(
                (df1_2d / np.clip(df2_2d, 1e-12, None) - 1.0) / frq[:, None],
                1e-5, None,
            )
            df_pay2d = g.df(T_p2d)
            tau_2d   = np.clip(T_r2d, 1e-6, None)

            K_2d    = a["strike"][cf][:, None] + np.zeros(max_cl)
            sig_2d  = a["sigma_n"][cf][:, None] + np.zeros(max_cl)
            call_2d = a["is_call_cap"][cf][:, None] + np.zeros(max_cl, dtype=bool)
            price_2d, delta_2d, vega_2d, _ = _bach_price_greeks(
                F_2d, K_2d, sig_2d, tau_2d, call_2d,
            )

            scale_2d  = frq[:, None] * df_pay2d * valid
            not_dir   = a["notional"][cf] * a["direction"][cf]
            pv[cf]    = (price_2d * scale_2d).sum(axis=1) * not_dir
            delta[cf] = (delta_2d * scale_2d).sum(axis=1) * not_dir
            vega[cf]  = (vega_2d  * scale_2d).sum(axis=1) * not_dir * 1e-4
            dv01[cf]  = (delta_2d * scale_2d).sum(axis=1) * not_dir * bump

        # ── Vanilla IRS ───────────────────────────────────────────────────────
        irs = a["is_irs"]
        if irs.any():
            T_start  = a["expiry"][irs]
            ten_irs  = a["tenor"][irs]
            freq_irs = a["freq"][irs]

            ann_irs   = self._exact_annuity(g, T_start, ten_irs, freq_irs)

            keys_irs  = [a["idx_keys"][i] for i in np.where(irs)[0]]
            df_s_proj = g.proj_df_grouped(keys_irs, T_start)
            df_e_proj = g.proj_df_grouped(keys_irs, T_start + ten_irs)
            F_irs     = (df_s_proj - df_e_proj) / ann_irs

            sign_irs  = np.where(a["is_payer_irs"][irs], 1.0, -1.0)
            scale_irs = sign_irs * ann_irs * a["notional"][irs] * a["direction"][irs]

            pv[irs]    = (F_irs - a["strike"][irs]) * scale_irs
            delta[irs] = scale_irs          # Δ = 1 per rate unit for payer, -1 for receiver
            vega[irs]  = 0.0               # no vol sensitivity
            dv01[irs]  = scale_irs * bump  # = delta × 1bp

        return pd.DataFrame({
            "label":      [p.label for p in self.book.positions],
            "instrument": a["instr"],
            "index_key":  a["idx_keys"],
            "ccy":        [INDEX_CATALOG[k]["ccy"] for k in a["idx_keys"]],
            "expiry_y":   a["expiry"],
            "tenor_y":    a["tenor"],
            "notional":   a["notional"],
            "direction":  a["direction"],
            "pv":         pv,
            "delta":      delta,
            "dv01":       dv01,
            "vega":       vega,
        })

    # ── aggregate_risk ────────────────────────────────────────────────────────

    def aggregate_risk(self) -> dict[str, pd.DataFrame]:
        """Aggregate risk by instrument, index, CCY, and maturity bucket."""
        df = self.risk_book()
        if df.empty:
            return {}

        def _agg(by):
            return (
                df.groupby(by, observed=True)[["pv", "dv01", "vega"]]
                .sum()
                .rename(columns={"pv": "PV ($)", "dv01": "DV01 ($)", "vega": "Vega/bp ($)"})
                .sort_values("PV ($)", ascending=False)
            )

        bins = [0, 1/12, 3/12, 1, 2, 5, 10, 30]
        lbls = ["<1M", "1-3M", "3M-1Y", "1-2Y", "2-5Y", "5-10Y", ">10Y"]
        df["exp_bucket"] = pd.cut(df["expiry_y"], bins=bins, labels=lbls)

        return {
            "by_instrument": _agg("instrument"),
            "by_index":      _agg("index_key"),
            "by_ccy":        _agg("ccy"),
            "by_expiry":     _agg("exp_bucket"),
        }
