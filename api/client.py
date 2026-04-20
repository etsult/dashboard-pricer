"""
Dual-mode pricer client.

Usage
-----
from api.client import get_client

client = get_client()          # reads PRICER_MODE env var ("local" | "http")
book   = client.generate_book(n=10_000)
df_pv  = client.price_book(book, curve)
df_risk= client.risk_book(book, curve, dv01_bp=1, gamma_bp=5)
cube   = client.vol_cube(seed=42)

Modes
-----
local  (default) — calls the engine directly in-process. Zero latency.
                   Use this for Streamlit running alongside the engine.
http             — makes HTTP calls to the FastAPI backend.
                   Set PRICER_API_URL=http://host:8000 (default localhost:8000).
                   Use this when frontend and backend run on separate processes/hosts.

Both modes expose the exact same interface so Streamlit pages (or any other
frontend) never import the engine directly — they only import this client.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# Protocol — the shared interface both clients must satisfy
# ══════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class PricerClient(Protocol):

    def generate_book(
        self,
        n: int = 10_000,
        seed: int = 42,
        usd_weight: float = 0.60,
        add_hedges: bool = True,
    ):
        """Return a Book object (or equivalent dict representation)."""
        ...

    def price_book(self, book, curve, shift_bp: float = 0.0) -> pd.DataFrame:
        """Return position-level PV DataFrame."""
        ...

    def risk_book(
        self,
        book,
        curve,
        shift_bp: float = 0.0,
        dv01_bp: float = 1.0,
        gamma_bp: float = 5.0,
    ) -> pd.DataFrame:
        """Return position-level DV01 / Γ+ / Γ− DataFrame."""
        ...

    def fast_risk(self, book, curve) -> tuple[pd.DataFrame, dict]:
        """Analytical Greeks (DV01, vega) + aggregated risk tables. Fast, no bumping."""
        ...

    def vol_cube(self, seed: int = 0):
        """Return a VolCube object."""
        ...

    def get_indexes(self) -> list[dict]:
        """Return index catalog."""
        ...

    def benchmark_book(self, book, curve) -> pd.DataFrame:
        """
        Price book with QuantLib and FastBookEngine; return comparison DataFrame.
        Requires QuantLib (pip install QuantLib). Slow — use only for small books.
        """
        ...

    def nn_price_book(self, book, curve) -> pd.DataFrame:
        """
        Price swaptions with SwaptionNetV2 (research/swaption_nn_v2_best.pt).
        Non-swaption instruments return NaN for pv/delta/vega/dv01.
        Requires PyTorch (pip install torch).
        """
        ...


# ══════════════════════════════════════════════════════════════════════════════
# NN inference helper (shared by LocalClient and HttpClient)
# ══════════════════════════════════════════════════════════════════════════════

def _nn_price_book(book, curve) -> pd.DataFrame:
    """
    Price swaptions with SwaptionNetV2 (research/swaption_nn_v2_best.pt).

    Non-swaption positions get NaN for pv/delta/vega/dv01.
    Runs one batched forward pass for all swaptions — O(1) NN calls regardless
    of book size.

    Requires: pip install torch
    """
    import pathlib, sys

    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch required for NN pricing: pip install torch")

    root = pathlib.Path(__file__).parent.parent
    research_dir = root / "research"
    if str(research_dir) not in sys.path:
        sys.path.insert(0, str(research_dir))

    from swaption_nn_v2 import SwaptionNetV2
    from pricer.ir.fast_engine import FastBookEngine, _CurveSetGrid
    from pricer.ir.indexes import INDEX_CATALOG
    from pricer.ir.zabr import zabr_atm_vol
    from market_data.curves.curve_set import CurveSet

    model_path = research_dir / "swaption_nn_v2_best.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"NN model checkpoint not found: {model_path}\n"
            "Train the model first by running research/swaption_nn_v2.py"
        )

    model = SwaptionNetV2()
    ckpt  = torch.load(str(model_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    # Forward rates from fast engine; grid for exact annuity
    cs      = CurveSet.from_single_curve(curve)
    g       = _CurveSetGrid(cs)
    fast_df = FastBookEngine(curve, book).price_book()

    CCY_IDX = {"USD": 0, "EUR": 1, "GBP": 2}

    # Build base row list (NaN for non-swaptions)
    rows = []
    for i, p in enumerate(book.positions):
        meta = INDEX_CATALOG[p.index_key]
        rows.append({
            "label": p.label, "instrument": p.instrument,
            "index_key": p.index_key, "ccy": meta["ccy"],
            "expiry_y": p.expiry_y, "tenor_y": p.tenor_y,
            "strike_pct": p.strike * 100, "sigma_bps": p.sigma_n * 10_000,
            "notional": p.notional, "direction": p.direction,
            "atm_pct": np.nan, "pv": np.nan,
            "delta": np.nan, "vega": np.nan, "dv01": np.nan,
        })

    sw_idxs = [i for i, p in enumerate(book.positions)
               if p.instrument in ("payer_swaption", "receiver_swaption")]
    if not sw_idxs:
        return pd.DataFrame(rows)

    # Build batched NN input tensors for all swaptions at once
    cont_list, conv_list, strata_list = [], [], []
    ann_list, sigma_n_list, sqrt_T_list, scale_list, F_list = [], [], [], [], []

    for i in sw_idxs:
        p    = book.positions[i]
        meta = INDEX_CATALOG[p.index_key]
        ccy  = meta["ccy"]

        F       = fast_df.iloc[i]["atm_pct"] / 100.0
        K       = p.strike
        sigma_n = max(p.sigma_n, 1e-4)
        tau     = max(p.expiry_y, 1e-6)
        sqrt_T  = np.sqrt(tau)

        alpha, nu, rho = zabr_atm_vol(tau, p.tenor_y, ccy)
        sigma_atm      = float(alpha)
        moneyness      = (F - K) / (sigma_n * sqrt_T)
        log_mk         = float(np.log(np.clip(K / max(F, 1e-6), 1e-4, 10.0)))
        strata_val     = 2 if abs(moneyness) >= 3 else (1 if abs(moneyness) >= 1 else 0)

        freq    = 1.0 / meta["reset_freq"]
        ann_val = float(FastBookEngine._exact_annuity(
            g, np.array([p.expiry_y]), np.array([p.tenor_y]), np.array([freq])
        )[0])

        cont_list.append([moneyness, log_mk, tau, p.tenor_y,
                          sigma_atm, nu, rho,
                          1.0 if p.instrument == "payer_swaption" else 0.0])
        conv_list.append(CCY_IDX.get(ccy, 0))
        strata_list.append(strata_val)
        ann_list.append(ann_val)
        sigma_n_list.append(sigma_n)
        sqrt_T_list.append(sqrt_T)
        scale_list.append(ann_val * p.notional * p.direction)
        F_list.append(F)

    cont_t  = torch.tensor(cont_list,   dtype=torch.float32)
    conv_t  = torch.tensor(conv_list,   dtype=torch.long)
    strat_t = torch.tensor(strata_list, dtype=torch.long)

    with torch.no_grad():
        out = model(cont_t, conv_t, strat_t).numpy()   # [n_sw, 6]

    sigma_n_arr = np.array(sigma_n_list)
    sqrt_T_arr  = np.array(sqrt_T_list)
    scale_arr   = np.array(scale_list)

    # Denormalise outputs:  norm_price = price/(σ√T),  norm_vega = vega/√T
    price_units = out[:, 0] * sigma_n_arr * sqrt_T_arr
    delta_units = out[:, 1]
    vega_units  = out[:, 2] * sqrt_T_arr

    for j, i in enumerate(sw_idxs):
        sc = scale_arr[j]
        rows[i].update({
            "atm_pct": F_list[j] * 100,
            "pv":    float(price_units[j]) * sc,
            "delta": float(delta_units[j]) * sc,
            "vega":  float(vega_units[j])  * sc * 1e-4,
            "dv01":  float(delta_units[j]) * sc * 1e-4,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# LocalClient — direct engine calls, no HTTP overhead
# ══════════════════════════════════════════════════════════════════════════════

class LocalClient:
    """
    Calls the pricer engine directly.
    Same thread, no serialisation — fastest possible path.
    """

    # ── Book generation ───────────────────────────────────────────────────────

    def generate_book(self, n=10_000, seed=42, usd_weight=0.60, add_hedges=True):
        from pricer.ir.book_generator import generate_book
        return generate_book(n=n, seed=seed,
                             usd_weight=usd_weight, add_hedges=add_hedges)

    # ── Pricing ───────────────────────────────────────────────────────────────

    def price_book(self, book, curve, shift_bp: float = 0.0) -> pd.DataFrame:
        from pricer.ir.fast_engine import FastBookEngine
        return FastBookEngine(curve.shifted(shift_bp), book).price_book()

    # ── Risk (bump-and-reprice) ───────────────────────────────────────────────

    def risk_book(
        self,
        book,
        curve,
        shift_bp: float = 0.0,
        dv01_bp: float = 1.0,
        gamma_bp: float = 5.0,
    ) -> pd.DataFrame:
        from pricer.ir.fast_engine import FastBookEngine
        c0   = curve.shifted(shift_bp)
        cols = ["label", "instrument", "index_key", "ccy",
                "expiry_y", "tenor_y", "notional", "direction", "pv"]
        df   = FastBookEngine(c0,                       book).price_book()[cols].copy()
        pu1  = FastBookEngine(c0.shifted(+dv01_bp),     book).price_book()["pv"].values
        pu5  = FastBookEngine(c0.shifted(+gamma_bp),    book).price_book()["pv"].values
        pd5  = FastBookEngine(c0.shifted(-gamma_bp),    book).price_book()["pv"].values
        df["dv01"]     = pu1 - df["pv"].values
        df["gamma_up"] = pu5 - df["pv"].values
        df["gamma_dn"] = pd5 - df["pv"].values
        return df

    # ── Fast analytical risk ──────────────────────────────────────────────────

    def fast_risk(self, book, curve) -> tuple[pd.DataFrame, dict]:
        from pricer.ir.fast_engine import FastBookEngine
        eng = FastBookEngine(curve, book)
        return eng.risk_book(), eng.aggregate_risk()

    # ── Vol cube ──────────────────────────────────────────────────────────────

    def vol_cube(self, seed: int = 0):
        from pricer.ir.vol_cube import VolCube
        return VolCube(seed=seed)

    # ── Indexes ───────────────────────────────────────────────────────────────

    def get_indexes(self) -> list[dict]:
        from pricer.ir.indexes import INDEX_CATALOG
        return [
            {"key": k, "label": v["label"], "ccy": v["ccy"],
             "reset_freq": v["reset_freq"], "basis_bps": v["basis_bps"]}
            for k, v in INDEX_CATALOG.items()
        ]

    # ── QuantLib benchmark ────────────────────────────────────────────────────

    def benchmark_book(self, book, curve) -> pd.DataFrame:
        from pricer.ir.fast_engine import FastBookEngine
        from pricer.ir.ql_engine import QLBookEngine
        fast_df = FastBookEngine(curve, book).price_book()
        return QLBookEngine(curve, book).compare(fast_df)

    # ── Neural Network (SwaptionNetV2) ────────────────────────────────────────

    def nn_price_book(self, book, curve) -> pd.DataFrame:
        return _nn_price_book(book, curve)


# ══════════════════════════════════════════════════════════════════════════════
# HttpClient — calls the FastAPI backend over HTTP
# ══════════════════════════════════════════════════════════════════════════════

class HttpClient:
    """
    Thin HTTP wrapper around the FastAPI endpoints.
    Deserialises responses back into the same types LocalClient returns
    so callers see no difference.

    Requires: pip install httpx
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required for HttpClient: pip install httpx")
        self._http = httpx.Client(base_url=base_url, timeout=120)
        self._base = base_url

    def _post(self, path: str, body: dict) -> dict:
        r = self._http.post(path, json=body)
        r.raise_for_status()
        return r.json()

    def _get(self, path: str, **params) -> dict | list:
        r = self._http.get(path, params=params)
        r.raise_for_status()
        return r.json()

    # ── Serialise curve → CurveSpec ───────────────────────────────────────────

    @staticmethod
    def _curve_spec(curve, shift_bp: float = 0.0) -> dict:
        """Convert a RateCurve to the CurveSpec wire format."""
        tenors = curve._tenors.tolist()
        rates  = curve._zero_rates.tolist()
        return {"points": list(zip(tenors, rates)), "shift_bp": shift_bp}

    # ── Serialise Book → list[dict] ───────────────────────────────────────────

    @staticmethod
    def _positions(book) -> list[dict]:
        return [
            {
                "instrument": p.instrument,
                "index_key":  p.index_key,
                "notional":   p.notional,
                "strike":     p.strike,
                "expiry_y":   p.expiry_y,
                "tenor_y":    p.tenor_y,
                "sigma_n":    p.sigma_n,
                "direction":  p.direction,
                "label":      getattr(p, "label", ""),
            }
            for p in book.positions
        ]

    # ── Deserialise response → Book ───────────────────────────────────────────

    @staticmethod
    def _to_book(data: dict):
        from pricer.ir.instruments import IRPosition, Book
        positions = [
            IRPosition(
                instrument=p["instrument"],
                index_key=p["index_key"],
                notional=p["notional"],
                strike=p["strike"],
                expiry_y=p["expiry_y"],
                tenor_y=p["tenor_y"],
                sigma_n=p["sigma_n"],
                direction=p["direction"],
                label=p.get("label", ""),
            )
            for p in data["positions"]
        ]
        return Book(positions=positions)

    # ── Public interface ──────────────────────────────────────────────────────

    def generate_book(self, n=10_000, seed=42, usd_weight=0.60, add_hedges=True):
        data = self._post("/api/ir/books/generate", {
            "n": n, "seed": seed,
            "usd_weight": usd_weight, "add_hedges": add_hedges,
        })
        return self._to_book(data)

    def price_book(self, book, curve, shift_bp: float = 0.0) -> pd.DataFrame:
        data = self._post("/api/ir/books/price", {
            "positions": self._positions(book),
            "curve": self._curve_spec(curve, shift_bp),
        })
        return pd.DataFrame(data["positions"])

    def risk_book(
        self, book, curve,
        shift_bp: float = 0.0,
        dv01_bp:  float = 1.0,
        gamma_bp: float = 5.0,
    ) -> pd.DataFrame:
        data = self._post("/api/ir/books/risk", {
            "positions": self._positions(book),
            "curve":     self._curve_spec(curve, shift_bp),
            "dv01_bp":   dv01_bp,
            "gamma_bp":  gamma_bp,
        })
        return pd.DataFrame(data["positions"])

    def fast_risk(self, book, curve) -> tuple[pd.DataFrame, dict]:
        # Delegate to risk_book; build aggregates locally (avoids extra endpoint).
        df = self.risk_book(book, curve)
        import pandas as pd
        _BINS = [0, 1/12, 3/12, 1, 2, 5, 10, 30]
        _LBLS = ["<1M", "1-3M", "3M-1Y", "1-2Y", "2-5Y", "5-10Y", ">10Y"]

        def _agg(by):
            return (
                df.groupby(by)[["pv", "dv01"]]
                .sum()
                .rename(columns={"pv": "PV ($)", "dv01": "DV01 ($)"})
            )

        df["exp_bucket"] = pd.cut(df["expiry_y"], bins=_BINS, labels=_LBLS)
        agg = {
            "by_instrument": _agg("instrument"),
            "by_index":      _agg("index_key"),
            "by_ccy":        _agg("ccy"),
            "by_expiry":     _agg("exp_bucket"),
        }
        return df, agg

    def vol_cube(self, seed: int = 0):
        # VolCube is built locally even in HTTP mode (it's pure computation,
        # no market data fetch needed) — avoids serialising a large surface grid.
        from pricer.ir.vol_cube import VolCube
        return VolCube(seed=seed)

    def get_indexes(self) -> list[dict]:
        return self._get("/api/ir/indexes")

    def benchmark_book(self, book, curve) -> pd.DataFrame:
        from pricer.ir.fast_engine import FastBookEngine
        from pricer.ir.ql_engine import QLBookEngine
        fast_df = FastBookEngine(curve, book).price_book()
        return QLBookEngine(curve, book).compare(fast_df)

    def nn_price_book(self, book, curve) -> pd.DataFrame:
        # NN inference always runs locally even in HTTP mode (model weights are local)
        return _nn_price_book(book, curve)


# ══════════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════════

def get_client() -> LocalClient | HttpClient:
    """
    Return the right client based on the PRICER_MODE environment variable.

      PRICER_MODE=local   → LocalClient (default)
      PRICER_MODE=http    → HttpClient using PRICER_API_URL (default: localhost:8000)

    Example .env:
      PRICER_MODE=http
      PRICER_API_URL=http://pricer-backend:8000
    """
    mode = os.getenv("PRICER_MODE", "local").lower()
    if mode == "http":
        url = os.getenv("PRICER_API_URL", "http://localhost:8000")
        return HttpClient(base_url=url)
    return LocalClient()
