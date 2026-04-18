"""
Router: /ir/books  and  /ir/risk

Endpoints
---------
POST /ir/books/generate          Generate a synthetic book (returns full position list)
POST /ir/books/price             Price a book with FastBookEngine (O(N) vectorised)
POST /ir/books/risk              Bump-and-reprice Greeks (DV01, Γ+, Γ−)
GET  /ir/indexes                 List all available indexes with metadata
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException

from api.schemas.books import (
    GenerateBookRequest,
    BookSchema, IRPositionSchema,
    PriceBookRequest, PriceBookResponse, PriceRow,
    RiskBookRequest, RiskBookResponse, RiskRow, AggregateRisk,
    CurveSpec,
)
from pricer.ir.instruments import IRPosition, Book
from pricer.ir.fast_engine import FastBookEngine
from pricer.ir.book_generator import generate_book as _gen_book
from pricer.ir.indexes import INDEX_CATALOG
from market_data.curves.rate_curve import RateCurve

router = APIRouter(prefix="/ir", tags=["IR Books & Risk"])

_EXP_BINS = [0, 1/12, 3/12, 1, 2, 5, 10, 30]
_EXP_LBLS = ["<1M", "1-3M", "3M-1Y", "1-2Y", "2-5Y", "5-10Y", ">10Y"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _curve_from_spec(spec: CurveSpec) -> RateCurve:
    par = {t: r for t, r in spec.points}
    return RateCurve(par).shifted(spec.shift_bp)


def _book_from_schema(positions: list[IRPositionSchema]) -> Book:
    return Book(positions=[
        IRPosition(
            instrument=p.instrument,
            index_key=p.index_key,
            notional=p.notional,
            strike=p.strike,
            expiry_y=p.expiry_y,
            tenor_y=p.tenor_y,
            sigma_n=p.sigma_n,
            direction=p.direction,
            label=p.label or f"{p.instrument[:3].upper()} {p.expiry_y:.1f}Y",
        )
        for p in positions
    ])


def _expiry_bucket(t: float) -> str:
    import pandas as pd
    cat = pd.cut([t], bins=_EXP_BINS, labels=_EXP_LBLS)
    return str(cat[0])


# ── GET /ir/indexes ───────────────────────────────────────────────────────────

@router.get("/indexes")
def list_indexes() -> list[dict]:
    """Return all available IR indexes with metadata."""
    return [
        {
            "key":        k,
            "label":      v["label"],
            "ccy":        v["ccy"],
            "reset_freq": v["reset_freq"],
            "daycount":   v["daycount"],
            "basis_bps":  v["basis_bps"],
        }
        for k, v in INDEX_CATALOG.items()
    ]


# ── POST /ir/books/generate ───────────────────────────────────────────────────

@router.post("/books/generate", response_model=BookSchema)
def generate_book(req: GenerateBookRequest) -> BookSchema:
    """
    Generate a synthetic IR option book.
    Returns up to 500k positions with realistic ZABR vols from the vol cube.
    """
    book = _gen_book(
        n=req.n,
        seed=req.seed,
        usd_weight=req.usd_weight,
        add_hedges=req.add_hedges,
    )
    pos_schemas = [
        IRPositionSchema(
            instrument=p.instrument,
            index_key=p.index_key,
            notional=p.notional,
            strike=p.strike,
            expiry_y=p.expiry_y,
            tenor_y=p.tenor_y,
            sigma_n=p.sigma_n,
            direction=p.direction,
            label=p.label,
        )
        for p in book.positions
    ]
    return BookSchema(
        book_id=str(uuid.uuid4()),
        n_positions=len(pos_schemas),
        positions=pos_schemas,
    )


# ── POST /ir/books/price ──────────────────────────────────────────────────────

@router.post("/books/price", response_model=PriceBookResponse)
def price_book(req: PriceBookRequest) -> PriceBookResponse:
    """
    Price a book of IR options using the fast vectorised engine.
    Accepts an inline curve spec (zero rates) and returns PV per position.
    """
    if not req.positions:
        raise HTTPException(422, "positions list is empty")

    curve = _curve_from_spec(req.curve)
    book  = _book_from_schema(req.positions)
    df    = FastBookEngine(curve, book).price_book()

    if df.empty:
        raise HTTPException(500, "Pricing returned empty result")

    rows = [
        PriceRow(
            label=str(r["label"]),
            instrument=str(r["instrument"]),
            index_key=str(r["index_key"]),
            ccy=str(r["ccy"]),
            expiry_y=float(r["expiry_y"]),
            tenor_y=float(r["tenor_y"]),
            strike_pct=float(r["strike_pct"]),
            atm_pct=float(r["atm_pct"]),
            sigma_bps=float(r["sigma_bps"]),
            notional=float(r["notional"]),
            direction=int(r["direction"]),
            pv=float(r["pv"]),
        )
        for _, r in df.iterrows()
    ]
    return PriceBookResponse(total_pv=float(df["pv"].sum()), positions=rows)


# ── POST /ir/books/risk ───────────────────────────────────────────────────────

@router.post("/books/risk", response_model=RiskBookResponse)
def risk_book(req: RiskBookRequest) -> RiskBookResponse:
    """
    Bump-and-reprice Greeks for an IR option book.

    Returns per-position DV01 (+dv01_bp bump), Γ+ (+gamma_bp) and Γ− (−gamma_bp)
    plus aggregated risk by index and expiry bucket.

    Four FastBookEngine calls — O(N) vectorised, <0.5s for 50k positions.
    """
    if not req.positions:
        raise HTTPException(422, "positions list is empty")

    curve = _curve_from_spec(req.curve)
    book  = _book_from_schema(req.positions)

    base_cols = ["label", "instrument", "index_key", "ccy",
                 "expiry_y", "tenor_y", "notional", "direction", "pv"]

    df0  = FastBookEngine(curve,                          book).price_book()[base_cols].copy()
    pv_u1 = FastBookEngine(curve.shifted(+req.dv01_bp),  book).price_book()["pv"].values
    pv_u5 = FastBookEngine(curve.shifted(+req.gamma_bp), book).price_book()["pv"].values
    pv_d5 = FastBookEngine(curve.shifted(-req.gamma_bp), book).price_book()["pv"].values

    df0["dv01"]     = pv_u1 - df0["pv"].values
    df0["gamma_up"] = pv_u5 - df0["pv"].values
    df0["gamma_dn"] = pv_d5 - df0["pv"].values

    # Aggregate by index
    by_idx: dict[str, dict[str, float]] = {}
    for idx, grp in df0.groupby("index_key"):
        by_idx[str(idx)] = {
            "pv":       float(grp["pv"].sum()),
            "dv01":     float(grp["dv01"].sum()),
            "gamma_up": float(grp["gamma_up"].sum()),
            "gamma_dn": float(grp["gamma_dn"].sum()),
        }

    # Aggregate by expiry bucket
    import pandas as pd
    df0["bucket"] = pd.cut(df0["expiry_y"], bins=_EXP_BINS, labels=_EXP_LBLS).astype(str)
    by_exp: dict[str, dict[str, float]] = {}
    for bkt, grp in df0.groupby("bucket"):
        by_exp[str(bkt)] = {
            "pv":       float(grp["pv"].sum()),
            "dv01":     float(grp["dv01"].sum()),
            "gamma_up": float(grp["gamma_up"].sum()),
            "gamma_dn": float(grp["gamma_dn"].sum()),
        }

    agg = AggregateRisk(
        total_pv=       float(df0["pv"].sum()),
        total_dv01=     float(df0["dv01"].sum()),
        total_gamma_up= float(df0["gamma_up"].sum()),
        total_gamma_dn= float(df0["gamma_dn"].sum()),
        by_index=by_idx,
        by_expiry=by_exp,
    )

    rows = [
        RiskRow(
            label=str(r["label"]),
            instrument=str(r["instrument"]),
            index_key=str(r["index_key"]),
            ccy=str(r["ccy"]),
            expiry_y=float(r["expiry_y"]),
            tenor_y=float(r["tenor_y"]),
            notional=float(r["notional"]),
            direction=int(r["direction"]),
            pv=float(r["pv"]),
            dv01=float(r["dv01"]),
            gamma_up=float(r["gamma_up"]),
            gamma_dn=float(r["gamma_dn"]),
        )
        for _, r in df0.iterrows()
    ]
    return RiskBookResponse(aggregate=agg, positions=rows)
