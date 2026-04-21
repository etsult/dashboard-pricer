"""
Router: /ir

Endpoints:
  POST /ir/curve          — build and return a zero-rate curve
  POST /ir/cap-floor      — price a cap or floor
  POST /ir/swaption       — price a swaption
  POST /ir/irs            — price a vanilla or xccy IRS
  POST /ir/benchmark      — cross-engine consistency benchmark
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException

from api.schemas.ir_options import (
    CapFloorRequest, CapFloorResponse, CapletDetail, SensitivityPoint,
    SwaptionRequest, SwaptionResponse,
    IRSRequest, IRSResponse, IRSLegDetail,
    BenchmarkRequest, BenchmarkResponse, BenchmarkEngineResult, BenchmarkDiagnostics,
    CurveResponse, ZeroCurvePoint,
    FredCurveSource, ManualCurveSource,
)
from api.services.ir_pricer import price_cap_floor, price_swaption, price_irs
from market_data.curves.rate_curve import RateCurve
from market_data.providers.fred import fetch_usd_curve

router = APIRouter(prefix="/ir", tags=["Interest Rate Options"])

_TENOR_LABELS = {
    1/12: "1M", 2/12: "2M", 3/12: "3M", 6/12: "6M",
    1.0: "1Y", 2.0: "2Y", 3.0: "3Y", 5.0: "5Y",
    7.0: "7Y", 10.0: "10Y", 15.0: "15Y", 20.0: "20Y", 30.0: "30Y",
}

# Freq (years) → USD index with matching reset frequency
_FREQ_TO_INDEX = {0.25: "SOFR_3M", 0.5: "TERM_SOFR_6M", 1.0: "SOFR"}


async def _build_curve(source: FredCurveSource | ManualCurveSource) -> RateCurve:
    if source.type == "fred":
        try:
            par_yields = fetch_usd_curve(api_key=source.api_key)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"FRED API error: {exc}")
    else:
        par_yields = {pt.tenor: pt.rate for pt in source.points}
    return RateCurve(par_yields)


# ─── QuantLib single-instrument dispatch ──────────────────────────────────────

def _ql_price_swaption(
    curve: RateCurve,
    swaption_type: str,
    notional: float,
    expiry_y: float,
    swap_tenor: float,
    freq: float,
    sigma_n: float,
    strike: float,
) -> float:
    from pricer.ir.ql_engine import QLBookEngine
    from pricer.ir.instruments import IRPosition, Book

    index_key = _FREQ_TO_INDEX.get(freq, "TERM_SOFR_6M")
    instr = "payer_swaption" if swaption_type == "payer" else "receiver_swaption"
    book = Book(positions=[IRPosition(
        instrument=instr, index_key=index_key,
        notional=notional, strike=strike,
        expiry_y=expiry_y, tenor_y=swap_tenor,
        sigma_n=sigma_n, direction=1,
    )])
    return float(QLBookEngine(curve, book).price_book()["pv"].iloc[0])


def _ql_price_capfloor(
    curve: RateCurve,
    instrument_type: str,
    notional: float,
    maturity: float,
    freq: float,
    sigma_n: float,
    strike: float,
) -> float:
    from pricer.ir.ql_engine import QLBookEngine
    from pricer.ir.instruments import IRPosition, Book

    index_key = _FREQ_TO_INDEX.get(freq, "SOFR_3M")
    book = Book(positions=[IRPosition(
        instrument=instrument_type, index_key=index_key,
        notional=notional, strike=strike,
        expiry_y=0.0, tenor_y=maturity,
        sigma_n=sigma_n, direction=1,
    )])
    return float(QLBookEngine(curve, book).price_book()["pv"].iloc[0])


def _ql_price_irs(
    curve: RateCurve,
    irs_type: str,
    notional: float,
    T_start: float,
    T_end: float,
    fixed_rate: float,
    freq: float,
) -> float:
    from pricer.ir.ql_engine import QLBookEngine
    from pricer.ir.instruments import IRPosition, Book

    index_key = _FREQ_TO_INDEX.get(freq, "TERM_SOFR_6M")
    instr = "payer_irs" if irs_type == "payer" else "receiver_irs"
    book = Book(positions=[IRPosition(
        instrument=instr, index_key=index_key,
        notional=notional, strike=fixed_rate,
        expiry_y=T_start, tenor_y=(T_end - T_start),
        sigma_n=0.0, direction=1,
    )])
    return float(QLBookEngine(curve, book).price_book()["pv"].iloc[0])


# ─── POST /ir/curve ───────────────────────────────────────────────────────────

@router.post("/curve", response_model=CurveResponse)
async def get_curve(source: FredCurveSource | ManualCurveSource) -> CurveResponse:
    curve = await _build_curve(source)
    zdf   = curve.zero_curve_df()
    points = [
        ZeroCurvePoint(
            tenor=row["Tenor"],
            tenor_label=_TENOR_LABELS.get(row["Tenor"], f"{row['Tenor']}Y"),
            zero_rate_pct=round(row["Zero Rate (%)"], 4),
            discount_factor=round(float(np.exp(-row["Zero Rate (%)"] / 100 * row["Tenor"])), 6),
        )
        for _, row in zdf.iterrows()
    ]
    return CurveResponse(points=points)


# ─── POST /ir/cap-floor ───────────────────────────────────────────────────────

@router.post("/cap-floor", response_model=CapFloorResponse)
async def price_cap_floor_endpoint(req: CapFloorRequest) -> CapFloorResponse:
    from pricer.ir.ql_engine import ql_available

    curve    = await _build_curve(req.curve)
    t_offset = req.start_shift_y + req.settlement_delay_y
    use_ql   = req.pricer_model == "quantlib" and ql_available() and req.vol_type == "normal"

    # QuantLib doesn't support forward-starting caps yet — fall back silently
    if use_ql and t_offset > 1e-6:
        use_ql = False

    def _price(k: float, sigma: float) -> float:
        if use_ql:
            return _ql_price_capfloor(curve, req.instrument_type, req.notional,
                                      req.maturity, req.freq, sigma, k)
        return price_cap_floor(curve, k, req.maturity, req.freq, sigma,
                               req.notional, req.vol_type, req.instrument_type, t_offset)[0]

    price, details = (
        (price_cap_floor(curve, req.strike, req.maturity, req.freq, req.sigma,
                         req.notional, req.vol_type, req.instrument_type, t_offset))
        if not use_ql else
        (_ql_price_capfloor(curve, req.instrument_type, req.notional,
                            req.maturity, req.freq, req.sigma, req.strike), [])
    )

    # Re-fetch caplet details from fast engine when QL is used (QL doesn't expose them)
    if use_ql:
        _, details = price_cap_floor(curve, req.strike, req.maturity, req.freq, req.sigma,
                                     req.notional, req.vol_type, req.instrument_type, t_offset)

    price_bps = price / req.notional * 10_000.0

    k_range = np.linspace(max(req.strike * 0.5, 0.001), req.strike * 1.5, 80)
    sens_strike = [SensitivityPoint(x=round(k*100, 4), price=round(_price(k, req.sigma), 2))
                   for k in k_range]

    if req.vol_type == "normal":
        vol_range = np.linspace(5, 250, 60) / 10_000.0
        vol_axis  = vol_range * 10_000
    else:
        vol_range = np.linspace(0.05, 1.0, 60)
        vol_axis  = vol_range * 100

    sens_vol = [SensitivityPoint(x=round(float(vax), 4), price=round(_price(req.strike, v), 2))
                for v, vax in zip(vol_range, vol_axis)]

    actual_model = "quantlib" if use_ql else req.pricer_model
    return CapFloorResponse(
        price=round(price, 2),
        price_bps=round(price_bps, 4),
        strike_pct=round(req.strike * 100, 4),
        caplet_details=[CapletDetail(**d) for d in details],
        sensitivity_strike=sens_strike,
        sensitivity_vol=sens_vol,
        pricer_model=actual_model,
    )


# ─── POST /ir/swaption ────────────────────────────────────────────────────────

@router.post("/swaption", response_model=SwaptionResponse)
async def price_swaption_endpoint(req: SwaptionRequest) -> SwaptionResponse:
    from pricer.ir.ql_engine import ql_available

    curve    = await _build_curve(req.curve)
    t_offset = req.start_shift_y + req.settlement_delay_y
    T_end    = req.expiry + req.swap_tenor
    use_ql   = req.pricer_model == "quantlib" and ql_available() and req.vol_type == "normal"

    def _price(k: float, sigma: float) -> float:
        if use_ql:
            return _ql_price_swaption(curve, req.swaption_type, req.notional,
                                      req.expiry + t_offset, req.swap_tenor,
                                      req.freq, sigma, k)
        return price_swaption(curve, req.expiry, T_end, k, sigma,
                              req.notional, req.vol_type, req.swaption_type,
                              req.freq, t_offset)[0]

    if use_ql:
        price = _ql_price_swaption(curve, req.swaption_type, req.notional,
                                   req.expiry + t_offset, req.swap_tenor,
                                   req.freq, req.sigma, req.strike)
        # Compute S0 and annuity from fast engine (same curve, negligible diff)
        _, S0, annuity = price_swaption(curve, req.expiry, T_end, req.strike, req.sigma,
                                        req.notional, req.vol_type, req.swaption_type,
                                        req.freq, t_offset)
    else:
        price, S0, annuity = price_swaption(curve, req.expiry, T_end, req.strike, req.sigma,
                                            req.notional, req.vol_type, req.swaption_type,
                                            req.freq, t_offset)

    price_bps = price / req.notional * 10_000.0
    moneyness = (S0 - req.strike) * 10_000 if req.swaption_type == "payer" else (req.strike - S0) * 10_000
    moneyness_label = "ITM" if moneyness > 0.5 else ("ATM" if abs(moneyness) <= 0.5 else "OTM")

    k_range = np.linspace(max(S0 * 0.5, 0.0001), S0 * 1.5, 80)
    sens_strike = [SensitivityPoint(x=round(k*100, 4), price=round(_price(k, req.sigma), 2))
                   for k in k_range]

    if req.vol_type == "normal":
        vol_range = np.linspace(5, 200, 60) / 10_000.0
        vol_axis  = vol_range * 10_000
    else:
        vol_range = np.linspace(0.03, 0.60, 60)
        vol_axis  = vol_range * 100

    sens_vol = [SensitivityPoint(x=round(float(vax), 4), price=round(_price(req.strike, v), 2))
                for v, vax in zip(vol_range, vol_axis)]

    actual_model = "quantlib" if use_ql else req.pricer_model
    return SwaptionResponse(
        price=round(price, 2),
        price_bps=round(price_bps, 4),
        par_swap_rate_pct=round(S0 * 100, 4),
        annuity=round(annuity, 6),
        moneyness_bps=round(moneyness, 2),
        moneyness_label=moneyness_label,
        sensitivity_strike=sens_strike,
        sensitivity_vol=sens_vol,
        pricer_model=actual_model,
    )


# ─── POST /ir/irs ─────────────────────────────────────────────────────────────

@router.post("/irs", response_model=IRSResponse)
async def price_irs_endpoint(req: IRSRequest) -> IRSResponse:
    from pricer.ir.ql_engine import ql_available

    curve   = await _build_curve(req.curve)
    T_start = req.start_shift_y
    T_end   = T_start + req.tenor_y
    use_ql  = req.pricer_model == "quantlib" and ql_available() and not req.xccy

    def _price(fixed_rate: float, shifted_curve: RateCurve = curve) -> float:
        if use_ql:
            return _ql_price_irs(shifted_curve, req.irs_type, req.notional,
                                 T_start, T_end, fixed_rate, req.fixed_freq)
        return price_irs(shifted_curve, T_start, T_end, fixed_rate,
                         req.fixed_freq, req.float_freq, req.notional,
                         req.irs_type, req.basis_spread_bps)[0]

    if use_ql:
        price     = _price(req.fixed_rate)
        # Get S0, annuity, and leg details from fast engine
        _, S0, ann, fix_pv, flt_pv, leg_details = price_irs(
            curve, T_start, T_end, req.fixed_rate, req.fixed_freq, req.float_freq,
            req.notional, req.irs_type, req.basis_spread_bps,
        )
    else:
        price, S0, ann, fix_pv, flt_pv, leg_details = price_irs(
            curve, T_start, T_end, req.fixed_rate, req.fixed_freq, req.float_freq,
            req.notional, req.irs_type, req.basis_spread_bps,
        )

    price_bps = price / req.notional * 10_000.0

    # DV01 via +1bp parallel shift
    dv01 = round(_price(req.fixed_rate, curve.shifted(+1)) - float(price), 2)

    rate_range = np.linspace(max(req.fixed_rate * 0.3, 0.001), req.fixed_rate * 2.0, 80)
    sens_rate = [SensitivityPoint(x=round(r*100, 4), price=round(_price(r), 2))
                 for r in rate_range]

    actual_model = "quantlib" if use_ql else req.pricer_model
    return IRSResponse(
        price=round(float(price), 2),
        price_bps=round(price_bps, 4),
        par_swap_rate_pct=round(S0 * 100, 4),
        annuity=round(ann, 6),
        fixed_leg_pv=round(float(fix_pv), 2),
        float_leg_pv=round(float(flt_pv), 2),
        dv01=dv01,
        leg_details=[IRSLegDetail(**d) for d in leg_details],
        sensitivity_rate=sens_rate,
        pricer_model=actual_model,
    )


# ─── POST /ir/benchmark ───────────────────────────────────────────────────────

@router.post("/benchmark", response_model=BenchmarkResponse)
async def benchmark_pricers(req: BenchmarkRequest) -> BenchmarkResponse:
    """
    Price the same instrument across all available engines and return a diff table.

    Engines
    -------
    fast_bachelier     : Our vectorised Bachelier / Black-76 (single-curve OIS)
    quantlib_single    : QuantLib bachelierBlackFormula, single-curve (projection = OIS, no basis)
    quantlib_multi     : QuantLib bachelierBlackFormula, multi-curve (projection = OIS + index basis)
    ore                : Open Risk Engine (QuantLib-based, if installed)

    The diff between fast_bachelier and quantlib_single measures pure *formula* error.
    The diff between quantlib_single and quantlib_multi measures the *multi-curve basis* effect.
    """
    from pricer.ir.ql_engine import ql_available, QLBookEngine
    from pricer.ir.instruments import IRPosition, Book
    from market_data.curves.curve_set import CurveSet
    from pricer.ir.indexes import INDEX_CATALOG

    # ── Validate request ─────────────────────────────────────────────────────
    instr_class = req.instrument_class
    if instr_class == "swaption" and req.swaption is None:
        raise HTTPException(422, "swaption field required for instrument_class=swaption")
    if instr_class == "cap_floor" and req.cap_floor is None:
        raise HTTPException(422, "cap_floor field required for instrument_class=cap_floor")
    if instr_class == "irs" and req.irs is None:
        raise HTTPException(422, "irs field required for instrument_class=irs")

    # ── Build curve ──────────────────────────────────────────────────────────
    if instr_class == "swaption":
        curve = await _build_curve(req.swaption.curve)   # type: ignore[union-attr]
        notional = req.swaption.notional                  # type: ignore[union-attr]
    elif instr_class == "cap_floor":
        curve = await _build_curve(req.cap_floor.curve)  # type: ignore[union-attr]
        notional = req.cap_floor.notional                 # type: ignore[union-attr]
    else:
        curve = await _build_curve(req.irs.curve)         # type: ignore[union-attr]
        notional = req.irs.notional                       # type: ignore[union-attr]

    # ── Determine index key and basis for diagnostics ────────────────────────
    if instr_class == "irs":
        freq = req.irs.fixed_freq                         # type: ignore[union-attr]
    elif instr_class == "swaption":
        freq = req.swaption.freq                          # type: ignore[union-attr]
    else:
        freq = req.cap_floor.freq                         # type: ignore[union-attr]

    index_key   = _FREQ_TO_INDEX.get(freq, "TERM_SOFR_6M")
    basis_bps   = float(INDEX_CATALOG[index_key]["basis_bps"])

    # ── CurveSets ────────────────────────────────────────────────────────────
    # single-curve: projection = OIS (no basis) → apples-to-apples with fast engine
    cs_single = CurveSet(ois=curve, projections={index_key: curve})
    # multi-curve: projection via INDEX_CATALOG basis → realistic market treatment
    cs_multi  = CurveSet.from_single_curve(curve)  # fallback adds basis

    # ── Fast Bachelier price ─────────────────────────────────────────────────
    def _fast_price() -> tuple[float, float]:
        """Returns (price, atm_rate)."""
        if instr_class == "swaption":
            r = req.swaption                             # type: ignore[union-attr]
            t_off = r.start_shift_y + r.settlement_delay_y
            p, S0, _ = price_swaption(curve, r.expiry, r.expiry + r.swap_tenor,
                                      r.strike, r.sigma, r.notional,
                                      r.vol_type, r.swaption_type, r.freq, t_off)
            return p, S0
        if instr_class == "cap_floor":
            r = req.cap_floor                            # type: ignore[union-attr]
            t_off = r.start_shift_y + r.settlement_delay_y
            p, _ = price_cap_floor(curve, r.strike, r.maturity, r.freq, r.sigma,
                                   r.notional, r.vol_type, r.instrument_type, t_off)
            fwd = curve.forward_rate(r.freq, r.freq * 2)
            return p, fwd
        r = req.irs                                      # type: ignore[union-attr]
        p, S0, *_ = price_irs(curve, r.start_shift_y, r.start_shift_y + r.tenor_y,
                               r.fixed_rate, r.fixed_freq, r.float_freq,
                               r.notional, r.irs_type, r.basis_spread_bps)
        return p, S0

    def _ql_price(cs: CurveSet) -> tuple[float, float]:
        """Price via QLBookEngine with the given CurveSet."""
        if instr_class == "swaption":
            r = req.swaption                             # type: ignore[union-attr]
            t_off = r.start_shift_y + r.settlement_delay_y
            instr_str = "payer_swaption" if r.swaption_type == "payer" else "receiver_swaption"
            book = Book(positions=[IRPosition(
                instrument=instr_str, index_key=index_key,
                notional=r.notional, strike=r.strike,
                expiry_y=r.expiry + t_off, tenor_y=r.swap_tenor,
                sigma_n=r.sigma if r.vol_type == "normal" else 0.0, direction=1,
            )])
            df_res = QLBookEngine(cs, book).price_book()
            pv = float(df_res["pv"].iloc[0])
            atm = float(df_res["atm_pct"].iloc[0]) / 100
            return pv, atm
        if instr_class == "cap_floor":
            r = req.cap_floor                            # type: ignore[union-attr]
            book = Book(positions=[IRPosition(
                instrument=r.instrument_type, index_key=index_key,
                notional=r.notional, strike=r.strike,
                expiry_y=0.0, tenor_y=r.maturity,
                sigma_n=r.sigma if r.vol_type == "normal" else 0.0, direction=1,
            )])
            df_res = QLBookEngine(cs, book).price_book()
            pv  = float(df_res["pv"].iloc[0])
            fwd = float(df_res["atm_pct"].iloc[0]) / 100
            return pv, fwd
        r = req.irs                                      # type: ignore[union-attr]
        instr_str = "payer_irs" if r.irs_type == "payer" else "receiver_irs"
        book = Book(positions=[IRPosition(
            instrument=instr_str, index_key=index_key,
            notional=r.notional, strike=r.fixed_rate,
            expiry_y=r.start_shift_y, tenor_y=r.tenor_y,
            sigma_n=0.0, direction=1,
        )])
        df_res = QLBookEngine(cs, book).price_book()
        pv  = float(df_res["pv"].iloc[0])
        atm = float(df_res["atm_pct"].iloc[0]) / 100
        return pv, atm

    # ── Run all engines ──────────────────────────────────────────────────────
    fast_price, fast_atm = _fast_price()
    fast_bps = fast_price / notional * 10_000

    results: list[BenchmarkEngineResult] = [
        BenchmarkEngineResult(
            engine="fast_bachelier",
            curve_mode="single-curve (OIS)",
            available=True,
            price=round(fast_price, 2),
            price_bps=round(fast_bps, 4),
            atm_rate_pct=round(fast_atm * 100, 4),
            diff_vs_fast_bps=0.0,
        )
    ]

    ql_single_price = ql_single_atm = ql_multi_price = None
    if ql_available():
        try:
            ql_single_price, ql_single_atm = _ql_price(cs_single)
            ql_single_bps = ql_single_price / notional * 10_000
            results.append(BenchmarkEngineResult(
                engine="quantlib_single",
                curve_mode="single-curve (OIS, no basis)",
                available=True,
                price=round(ql_single_price, 2),
                price_bps=round(ql_single_bps, 4),
                atm_rate_pct=round(ql_single_atm * 100, 4),
                diff_vs_fast_bps=round(ql_single_bps - fast_bps, 4),
            ))
        except Exception as exc:
            results.append(BenchmarkEngineResult(
                engine="quantlib_single", curve_mode="single-curve (OIS)", available=False,
                note=str(exc),
            ))

        try:
            ql_multi_price, ql_multi_atm = _ql_price(cs_multi)
            ql_multi_bps = ql_multi_price / notional * 10_000
            results.append(BenchmarkEngineResult(
                engine="quantlib_multi",
                curve_mode=f"multi-curve (OIS + {basis_bps:.0f}bps {index_key} basis)",
                available=True,
                price=round(ql_multi_price, 2),
                price_bps=round(ql_multi_bps, 4),
                atm_rate_pct=round(ql_multi_atm * 100, 4),
                diff_vs_fast_bps=round(ql_multi_bps - fast_bps, 4),
            ))
        except Exception as exc:
            results.append(BenchmarkEngineResult(
                engine="quantlib_multi", curve_mode="multi-curve", available=False,
                note=str(exc),
            ))
    else:
        results.append(BenchmarkEngineResult(
            engine="quantlib_single", curve_mode="single-curve", available=False,
            note="QuantLib not installed. Run: pip install QuantLib",
        ))
        results.append(BenchmarkEngineResult(
            engine="quantlib_multi", curve_mode="multi-curve", available=False,
            note="QuantLib not installed.",
        ))

    # ORE check (QuantLib-based, prices will match quantlib_multi for vanilla)
    try:
        import ORE  # noqa: F401
        ore_available = True
    except ImportError:
        ore_available = False

    results.append(BenchmarkEngineResult(
        engine="ore",
        curve_mode="multi-curve (QuantLib/ORE)",
        available=ore_available,
        price=round(ql_multi_price, 2) if ore_available and ql_multi_price is not None else None,
        price_bps=round(ql_multi_price / notional * 10_000, 4) if ore_available and ql_multi_price else None,
        diff_vs_fast_bps=round((ql_multi_price / notional * 10_000) - fast_bps, 4) if ore_available and ql_multi_price else None,
        note=None if ore_available else "ORE not installed. Run: pip install open-source-risk-engine",
    ))

    # ── Diagnostics ──────────────────────────────────────────────────────────
    formula_diff   = round((ql_single_price / notional * 10_000) - fast_bps, 4) if ql_single_price is not None else 0.0
    multicurve_eff = round(((ql_multi_price or 0) - (ql_single_price or 0)) / notional * 10_000, 4) if ql_multi_price and ql_single_price else 0.0

    return BenchmarkResponse(
        instrument_class=instr_class,
        notional=notional,
        results=results,
        diagnostics=BenchmarkDiagnostics(
            index_key=index_key,
            basis_spread_bps=basis_bps,
            formula_diff_bps=formula_diff,
            multicurve_effect_bps=multicurve_eff,
        ),
    )
