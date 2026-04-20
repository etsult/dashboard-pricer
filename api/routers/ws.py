"""
WebSocket router — real-time streaming endpoints.

Usage from the React frontend:
  const ws = new WebSocket('ws://localhost:5173/ws/live-monitor?currency=BTC&interval=30')
  ws.onmessage = e => console.log(JSON.parse(e.data))

The Vite dev server proxies /ws/* → ws://localhost:8000/ws/*, so the frontend
never needs to know the backend port.
"""

from __future__ import annotations

import asyncio
import datetime
from typing import Literal

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import time

import numpy as np

from api.schemas.market_data import VolTermStructureResponse, VolTermStructurePoint
from market_data.providers.deribit import DeribitProvider
from market_data.curves.vol_term_structure import build_term_structure
from market_data.curves.rate_curve import RateCurve
from pricer.ir.book_generator import generate_book as _gen_book
from pricer.ir.fast_engine import FastBookEngine
from pricer.ir.instruments import IRPosition, Book

router = APIRouter(prefix="/ws", tags=["WebSocket"])


def _fetch_term_structure(
    currency: str, rate: float
) -> VolTermStructureResponse:
    """Synchronous fetch — called inside asyncio.to_thread to avoid blocking the event loop."""
    provider = DeribitProvider()
    spot = provider.get_forward(currency)
    quotes = provider.get_option_chain(currency)
    fetched_at = datetime.datetime.utcnow().strftime("%H:%M:%S UTC")

    ts = build_term_structure(quotes, spot=spot, rate=rate)

    points = [
        VolTermStructurePoint(
            tenor_label=row["tenor_label"],
            days=round(row["days"], 1),
            atm_iv_pct=round(row["atm_iv"] * 100, 2),
            fwd_vol_pct=round(row["fwd_vol"] * 100, 2)
            if row["fwd_vol"] is not None and str(row["fwd_vol"]) != "nan"
            else None,
            fwd_label=row.get("fwd_label"),
            total_var=round(row["total_var"], 6),
            is_calendar_arb=bool(row["is_calendar_arb"]),
        )
        for _, row in ts.iterrows()
    ]

    return VolTermStructureResponse(
        currency=currency,
        spot=round(spot, 2),
        fetched_at=fetched_at,
        n_quotes=len(quotes),
        term_structure=points,
        n_arb_violations=int(ts["is_calendar_arb"].sum()),
    )


@router.websocket("/live-monitor")
async def live_monitor(
    websocket: WebSocket,
    currency: Literal["BTC", "ETH", "SOL"] = "BTC",
    rate: float = 0.05,
    interval: int = 30,
) -> None:
    """
    Stream vol term structure updates every `interval` seconds.

    Query params:
      currency  — BTC | ETH | SOL  (default BTC)
      rate      — funding rate      (default 0.05)
      interval  — seconds between   (default 30, min 5)

    Each message is a JSON object matching VolTermStructureResponse.
    On error, sends {"error": "<message>"} instead.
    """
    await websocket.accept()

    # Clamp interval: never faster than 5 s to respect Deribit rate limits
    interval = max(5, interval)

    try:
        while True:
            try:
                data = await asyncio.to_thread(_fetch_term_structure, currency, rate)
                await websocket.send_text(data.model_dump_json())
            except Exception as exc:
                await websocket.send_json({"error": str(exc)})

            await asyncio.sleep(interval)

    except WebSocketDisconnect:
        pass  # client closed — clean exit


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for intraday Greeks stream
# ─────────────────────────────────────────────────────────────────────────────

_BASE_CURVE = {0.25: 0.043, 0.5: 0.044, 1: 0.042, 2: 0.041, 5: 0.042, 10: 0.044, 30: 0.048}
_BASE_SHORT_RATE = 0.043  # 3M rate

# Bucketing grids
_EXP_BINS = [0, 1/12, 3/12, 1, 2, 5, 10, 30]
_EXP_LBLS = ["<1M", "1-3M", "3M-1Y", "1-2Y", "2-5Y", "5-10Y", ">10Y"]
_TEN_BINS = [0, 1, 2, 5, 10, 30]
_TEN_LBLS = ["≤1Y", "1-2Y", "2-5Y", "5-10Y", ">10Y"]


def _vol_bumped_book(book: Book, bump_bps: float = 1.0) -> Book:
    bump = bump_bps / 10_000
    return Book(positions=[
        IRPosition(
            instrument=p.instrument, index_key=p.index_key,
            notional=p.notional, strike=p.strike,
            expiry_y=p.expiry_y, tenor_y=p.tenor_y,
            sigma_n=p.sigma_n + bump,
            direction=p.direction, label=p.label,
        )
        for p in book.positions
    ])


def _bucket_greeks(
    book: Book,
    pv0: np.ndarray,
    dv01_arr: np.ndarray,
    gu_arr: np.ndarray,
    gd_arr: np.ndarray,
    vega_arr: np.ndarray,
) -> tuple[dict, dict]:
    """
    Returns (by_expiry, matrix_dv01) where:
      by_expiry   : expiry-bucket → {count, pv, dv01, gamma_up, gamma_dn, vega}
      matrix_dv01 : expiry-bucket → tenor-bucket → dv01  (vol-matrix style)
    """
    expiries = np.array([p.expiry_y for p in book.positions])
    tenors   = np.array([p.tenor_y  for p in book.positions])

    exp_idx = np.searchsorted(_EXP_BINS[1:], expiries, side="left")
    exp_idx = np.clip(exp_idx, 0, len(_EXP_LBLS) - 1)
    ten_idx = np.searchsorted(_TEN_BINS[1:], tenors, side="left")
    ten_idx = np.clip(ten_idx, 0, len(_TEN_LBLS) - 1)

    by_expiry: dict = {}
    for ei, lbl in enumerate(_EXP_LBLS):
        m = exp_idx == ei
        if not m.any():
            continue
        by_expiry[lbl] = {
            "count":    int(m.sum()),
            "pv":       round(float(pv0[m].sum()), 0),
            "dv01":     round(float(dv01_arr[m].sum()), 0),
            "gamma_up": round(float(gu_arr[m].sum()), 0),
            "gamma_dn": round(float(gd_arr[m].sum()), 0),
            "vega":     round(float(vega_arr[m].sum()), 0),
        }

    # Matrix: expiry × tenor  (DV01 as the primary Greek — most intuitive for rates)
    matrix_dv01: dict = {}
    for ei, exp_lbl in enumerate(_EXP_LBLS):
        row: dict = {}
        em = exp_idx == ei
        if not em.any():
            continue
        for ti, ten_lbl in enumerate(_TEN_LBLS):
            mask = em & (ten_idx == ti)
            if not mask.any():
                continue
            row[ten_lbl] = round(float(dv01_arr[mask].sum()), 0)
        if row:
            matrix_dv01[exp_lbl] = row

    return by_expiry, matrix_dv01


def _compute_step(curve: RateCurve, book: Book, use_ql: bool) -> dict:
    """
    Compute PV + Greeks for one path step.

    Vol note: sigma_n is FROZEN at book-generation values.
    Only the yield curve shifts intraday (rate path simulation).
    Proper intraday vol simulation would require:
      dσ_N = ρ_σr × σ_vov × dW_rates + √(1−ρ²) × σ_vov × dW_⊥
    with empirical ρ_σr ≈ −0.3 to −0.5 for USD. (TODO)
    """
    t0 = time.perf_counter()

    if use_ql:
        from pricer.ir.ql_engine import QLBookEngine
        df0  = QLBookEngine(curve, book).price_book()
        pv0  = df0["pv"].values
        dv01_arr = QLBookEngine(curve.shifted(+1), book).price_book()["pv"].values - pv0
        gu_arr   = QLBookEngine(curve.shifted(+5), book).price_book()["pv"].values - pv0
        gd_arr   = QLBookEngine(curve.shifted(-5), book).price_book()["pv"].values - pv0
        vega_arr = QLBookEngine(curve, _vol_bumped_book(book)).price_book()["pv"].values - pv0
    else:
        df0  = FastBookEngine(curve, book).price_book()
        pv0  = df0["pv"].values
        dv01_arr = FastBookEngine(curve.shifted(+1), book).price_book()["pv"].values - pv0
        gu_arr   = FastBookEngine(curve.shifted(+5), book).price_book()["pv"].values - pv0
        gd_arr   = FastBookEngine(curve.shifted(-5), book).price_book()["pv"].values - pv0
        vega_arr = FastBookEngine(curve, _vol_bumped_book(book)).price_book()["pv"].values - pv0

    by_expiry, matrix_dv01 = _bucket_greeks(book, pv0, dv01_arr, gu_arr, gd_arr, vega_arr)

    return {
        "pv":         round(float(pv0.sum()), 2),
        "dv01":       round(float(dv01_arr.sum()), 2),
        "gamma_up":   round(float(gu_arr.sum()), 2),
        "gamma_dn":   round(float(gd_arr.sum()), 2),
        "vega":       round(float(vega_arr.sum()), 2),
        "by_expiry":  by_expiry,
        "matrix_dv01": matrix_dv01,
        "compute_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


@router.websocket("/intraday-greeks")
async def intraday_greeks(
    websocket: WebSocket,
    n_positions: int = 300,
    seed: int = 42,
    pricer_model: str = "fast",      # fast | quantlib
    rate_model: str = "ou",          # bm | ou  (hull-white: TODO)
    n_steps: int = 390,              # trading-day minutes
    speed_ms: int = 100,             # ms between steps
    rate_vol_bps: float = 80.0,      # annualised rate vol
    mean_reversion: float = 0.15,    # κ for OU (ignored for bm)
) -> None:
    """
    Simulate one intraday rate path and stream portfolio Greeks at each step.

    Rate models
    -----------
    bm  : dS = σ dW  (flat Brownian motion)
    ou  : dS = -κ S dt + σ dW  (Ornstein-Uhlenbeck, H-W proxy)
    TODO: full Hull-White with term-structure calibration

    Pricer models
    -------------
    fast     : vectorised numpy / Bachelier (O(N), <5 ms per reprice)
    quantlib : QuantLib Bachelier + exact schedules (slower, for validation)
    TODO: NN pricer, ORE, Strata
    """
    await websocket.accept()

    n_steps      = int(np.clip(n_steps, 10, 780))
    speed_ms     = max(speed_ms, 20)
    n_positions  = int(np.clip(n_positions, 50, 5_000))
    use_ql       = pricer_model == "quantlib"
    if use_ql:
        n_positions = min(n_positions, 300)   # QL is ~100 ms/reprice

    try:
        # ── Generate book ────────────────────────────────────────────────────
        book = await asyncio.to_thread(
            _gen_book, n=n_positions, seed=seed, usd_weight=0.6, add_hedges=False
        )
        await websocket.send_json({
            "status": "init",
            "n_positions": len(book.positions),
            "pricer_model": pricer_model,
            "rate_model": rate_model,
            "n_steps": n_steps,
        })

        # ── Rate path setup ──────────────────────────────────────────────────
        rng   = np.random.default_rng(seed)
        dt    = 1.0 / max(n_steps, 1)
        sigma = (rate_vol_bps / 10_000) * np.sqrt(dt)   # per-step σ in rate units
        kappa = mean_reversion * dt                       # per-step mean reversion
        shift_rate = 0.0                                  # cumulative shift (in rate units)
        start_min  = 9 * 60 + 30                         # 09:30

        # ── Streaming loop ───────────────────────────────────────────────────
        for step in range(n_steps):
            dW = float(rng.standard_normal())
            if rate_model == "ou":
                shift_rate += -kappa * shift_rate + sigma * dW
            else:  # bm
                shift_rate += sigma * dW

            shift_bp = shift_rate * 10_000
            curve    = RateCurve(_BASE_CURVE).shifted(shift_bp)

            greeks = await asyncio.to_thread(_compute_step, curve, book, use_ql)

            minute  = start_min + step
            t_label = f"{minute // 60:02d}:{minute % 60:02d}"

            await websocket.send_json({
                "status": "running",
                "step": step,
                "n_steps": n_steps,
                "t_label": t_label,
                "shift_bp": round(shift_bp, 3),
                "short_rate_pct": round((_BASE_SHORT_RATE + shift_rate) * 100, 4),
                **greeks,
            })

            await asyncio.sleep(speed_ms / 1_000)

        await websocket.send_json({"status": "done", "step": n_steps})

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"status": "error", "message": str(exc)})
        except Exception:
            pass
