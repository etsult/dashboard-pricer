"""
Synthetic IR option book generator.

Generates realistic books of caps, floors, payer / receiver swaptions
across all USD and EUR indexes in O(N) time — no Python loops needed
for parameter generation.

Calibrated to a realistic 2025 market environment:
  USD: SOFR OIS 4.80% · typical ATM 4.40-4.90%
  EUR: €STR OIS 2.20% · typical ATM 2.25-2.50%
"""

from __future__ import annotations

import numpy as np
from .indexes import INDEX_CATALOG
from .instruments import IRPosition, Book
from .vol_cube import VolCube

# ── Index universe ─────────────────────────────────────────────────────────────

USD_INDEXES = [k for k, v in INDEX_CATALOG.items() if v["ccy"] == "USD"]
EUR_INDEXES = [k for k, v in INDEX_CATALOG.items() if v["ccy"] == "EUR"]
ALL_INDEXES = USD_INDEXES + EUR_INDEXES

# Flat ATM forward rates per CCY (realistic 2025)
_ATM_RATE = {"USD": 0.0450, "EUR": 0.0235, "GBP": 0.0455}

# Expiry distribution weights (shorter tenors dominate trading activity)
_EXPIRIES = np.array([1/12, 3/12, 6/12, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
_EXP_W    = np.array([0.06, 0.12, 0.16, 0.22, 0.16, 0.10, 0.10, 0.05, 0.03])
_EXP_W   /= _EXP_W.sum()

# Swap / cap tenor distribution
_TENORS   = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0])
_TEN_W    = np.array([0.08, 0.14, 0.14, 0.26, 0.12, 0.16, 0.05, 0.03, 0.02])
_TEN_W   /= _TEN_W.sum()

# Notional distribution (log-uniform, rounded to 1M)
_NOT_LO = np.log(5e6)
_NOT_HI = np.log(300e6)

# Instrument mix (payer heavy — typical dealer book)
_INSTR   = ["payer_swaption", "receiver_swaption", "cap", "floor"]
_INSTR_W = np.array([0.32, 0.28, 0.22, 0.18])


# ── Generator ─────────────────────────────────────────────────────────────────

def generate_book(
    n: int = 10_000,
    seed: int = 42,
    usd_weight: float = 0.60,    # fraction of positions on USD indexes
    add_hedges: bool = True,     # add ~15% offsetting positions (delta hedges)
) -> Book:
    """
    Generate a synthetic IR option book with n positions.

    Parameters
    ----------
    n           : number of option positions to generate
    seed        : random seed for reproducibility
    usd_weight  : fraction on USD indexes (rest is EUR)
    add_hedges  : add offsetting short positions (~15% of book)

    Returns
    -------
    Book object with IRPosition list (created in ~0.2s for 100k positions)
    """
    rng = np.random.default_rng(seed)

    # ── 1. Instrument types ───────────────────────────────────────────────────
    instr_idx = rng.choice(len(_INSTR), size=n, p=_INSTR_W)
    instruments = np.array(_INSTR)[instr_idx]

    # ── 2. Index (USD vs EUR split, then pick within CCY) ─────────────────────
    is_usd = rng.random(n) < usd_weight
    indexes = np.where(
        is_usd,
        np.array(USD_INDEXES)[rng.choice(len(USD_INDEXES), n)],
        np.array(EUR_INDEXES)[rng.choice(len(EUR_INDEXES), n)],
    )
    ccys = np.array([INDEX_CATALOG[k]["ccy"] for k in indexes])

    # ── 3. Expiries & tenors ─────────────────────────────────────────────────
    expiry_idx = rng.choice(len(_EXPIRIES), size=n, p=_EXP_W)
    expiries   = _EXPIRIES[expiry_idx]

    tenor_idx  = rng.choice(len(_TENORS), size=n, p=_TEN_W)
    tenors     = _TENORS[tenor_idx]

    # ── 4. Notionals ──────────────────────────────────────────────────────────
    notionals = np.round(np.exp(rng.uniform(_NOT_LO, _NOT_HI, n)) / 1e6) * 1e6

    # ── 5. ATM forward rates & vols ───────────────────────────────────────────
    F_atm = np.array([_ATM_RATE[c] for c in ccys])
    F_atm += rng.normal(0, 0.001, n)
    F_atm  = np.clip(F_atm, 0.001, 0.15)

    # Vol cube: one surface per CCY (swaptions) and per index (caps/floors).
    # Vectorised batch lookup — no Python loop over positions.
    cube = VolCube(seed=int(seed))
    is_sw = np.isin(instruments, ["payer_swaption", "receiver_swaption"])
    is_cf = ~is_sw

    sigma_n = np.zeros(n)
    if is_sw.any():
        sigma_n[is_sw] = cube.batch_swaption_vol(ccys[is_sw], expiries[is_sw], tenors[is_sw])
    if is_cf.any():
        # For caps/floors the "tenor" is the strip maturity
        sigma_n[is_cf] = cube.batch_capfloor_vol(indexes[is_cf], tenors[is_cf])
    sigma_n = np.clip(sigma_n, 0.0010, 0.0300)

    # ── 6. Strikes: ATM ± vol-scaled moneyness spread ────────────────────────
    atm_move = sigma_n * np.sqrt(expiries)                   # 1-sigma move in rate space
    z        = rng.normal(0, 0.6, n)                         # moneyness z-score
    strikes  = np.clip(F_atm + z * atm_move, 0.0005, 0.12)

    # ── 7. Direction: slight long bias (typical client-facing dealer book) ────
    directions = rng.choice([-1, 1], size=n, p=[0.43, 0.57])

    # ── 8. Build IRPosition objects ───────────────────────────────────────────
    positions = [
        IRPosition(
            instrument=str(instruments[i]),
            index_key=str(indexes[i]),
            notional=float(notionals[i]),
            strike=float(strikes[i]),
            expiry_y=float(expiries[i]),
            tenor_y=float(tenors[i]),
            sigma_n=float(sigma_n[i]),
            direction=int(directions[i]),
        )
        for i in range(n)
    ]

    # ── 9. Add delta hedges (short swaptions at ATM to neutralise delta) ──────
    if add_hedges:
        n_hedges = max(1, int(n * 0.15))
        h_instr  = rng.choice(["payer_swaption", "receiver_swaption"], n_hedges)
        h_idx    = np.array(USD_INDEXES + EUR_INDEXES)[rng.choice(len(ALL_INDEXES), n_hedges)]
        h_ccys   = [INDEX_CATALOG[k]["ccy"] for k in h_idx]
        h_exp    = _EXPIRIES[rng.choice(len(_EXPIRIES), n_hedges, p=_EXP_W)]
        h_ten    = _TENORS[rng.choice(len(_TENORS), n_hedges, p=_TEN_W)]
        h_not    = np.round(np.exp(rng.uniform(_NOT_LO, _NOT_HI, n_hedges)) / 1e6) * 1e6
        h_F      = np.array([_ATM_RATE[c] for c in h_ccys])

        h_sigma = cube.batch_swaption_vol(
            np.array(h_ccys), h_exp, h_ten,
        )
        h_sigma = np.clip(h_sigma, 0.0010, 0.0300)

        for i in range(n_hedges):
            positions.append(IRPosition(
                instrument=str(h_instr[i]),
                index_key=str(h_idx[i]),
                notional=float(h_not[i]),
                strike=float(h_F[i]),
                expiry_y=float(h_exp[i]),
                tenor_y=float(h_ten[i]),
                sigma_n=float(h_sigma[i]),
                direction=-1,
            ))

    return Book(positions)


def book_summary(book: Book) -> dict:
    """
    Fast O(N) book statistics — no pricing, just parameter statistics.
    Returns a dict of DataFrames and scalars for display.
    """
    import pandas as pd

    if book.is_empty():
        return {}

    pos = book.positions
    n   = len(pos)

    df = pd.DataFrame({
        "instrument": [p.instrument for p in pos],
        "index":      [p.index_key  for p in pos],
        "ccy":        [INDEX_CATALOG[p.index_key]["ccy"] for p in pos],
        "notional":   [p.notional   for p in pos],
        "expiry_y":   [p.expiry_y   for p in pos],
        "tenor_y":    [p.tenor_y    for p in pos],
        "strike":     [p.strike     for p in pos],
        "sigma_n":    [p.sigma_n    for p in pos],
        "direction":  [p.direction  for p in pos],
    })

    total_notional = df["notional"].sum()

    by_instr = (
        df.groupby("instrument")["notional"]
        .agg(["count", "sum"])
        .rename(columns={"count": "# Positions", "sum": "Notional ($)"})
        .sort_values("Notional ($)", ascending=False)
    )

    by_index = (
        df.groupby("index")["notional"]
        .agg(["count", "sum"])
        .rename(columns={"count": "# Positions", "sum": "Notional ($)"})
        .sort_values("Notional ($)", ascending=False)
    )

    by_ccy = (
        df.groupby("ccy")["notional"]
        .agg(["count", "sum"])
        .rename(columns={"count": "# Positions", "sum": "Notional ($)"})
    )

    # Maturity ladder: bucket by expiry
    bins  = [0, 1/12, 3/12, 1, 2, 5, 10, 30]
    lbls  = ["<1M", "1-3M", "3M-1Y", "1-2Y", "2-5Y", "5-10Y", ">10Y"]
    df["exp_bucket"] = pd.cut(df["expiry_y"], bins=bins, labels=lbls)
    by_exp = (
        df.groupby("exp_bucket", observed=True)["notional"]
        .agg(["count", "sum"])
        .rename(columns={"count": "# Positions", "sum": "Notional ($)"})
    )

    long_notional  = df[df["direction"] > 0]["notional"].sum()
    short_notional = df[df["direction"] < 0]["notional"].sum()

    return {
        "n":               n,
        "total_notional":  total_notional,
        "long_notional":   long_notional,
        "short_notional":  short_notional,
        "by_instrument":   by_instr,
        "by_index":        by_index,
        "by_ccy":          by_ccy,
        "by_expiry":       by_exp,
        "df":              df,
    }
