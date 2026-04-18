"""
Router: /execution

Endpoints:
  GET  /execution/dca/status       — next scheduled run + config
  POST /execution/dca/run          — manual trigger (test before month-end)
  GET  /execution/dca/history      — all past executions
  GET  /execution/dca/summary      — total BTC accumulated, avg cost, etc.
  GET  /execution/binance/ticker   — live price check
  GET  /execution/binance/balance  — account balance check
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from storage.database import get_db
from storage.models import DCAExecution

router = APIRouter(prefix="/execution", tags=["Execution"])


def _testnet_flag() -> bool:
    return os.environ.get("DCA_TESTNET", "true").lower() != "false"


def _get_executor():
    from execution.binance_dca import executor_from_env
    try:
        return executor_from_env(testnet=_testnet_flag())
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ─── Status & config ──────────────────────────────────────────────────────────

@router.get("/dca/status")
def dca_status():
    """Show current DCA config and next scheduled run."""
    from execution.scheduler import scheduler

    job = scheduler.get_job("monthly_dca")
    next_run = job.next_run_time.isoformat() if job and job.next_run_time else None

    return {
        "symbol":          os.environ.get("DCA_SYMBOL", "BTC/EUR"),
        "amount_eur":      float(os.environ.get("DCA_AMOUNT_EUR", "100")),
        "day_of_month":    int(os.environ.get("DCA_DAY_OF_MONTH", "1")),
        "hour_local":      int(os.environ.get("DCA_HOUR_LOCAL", "20")),
        "timezone":        os.environ.get("DCA_TIMEZONE", "Europe/Paris"),
        "testnet":         _testnet_flag(),
        "scheduler_running": scheduler.running,
        "next_run":        next_run,
    }


# ─── Manual trigger ───────────────────────────────────────────────────────────

@router.post("/dca/run")
async def trigger_dca_now(
    symbol:       str   = Query("BTC/EUR"),
    amount_eur:   float = Query(100.0, gt=0),
    db: Session = Depends(get_db),
):
    """
    Manually trigger a DCA buy right now.
    Use this to test your API keys before the first scheduled run.

    TESTNET is controlled by the DCA_TESTNET environment variable.
    """
    executor = _get_executor()

    row = DCAExecution(
        symbol=symbol,
        quote_amount=amount_eur,
        testnet=_testnet_flag(),
        status="pending",
    )
    db.add(row)
    db.commit()

    try:
        result = executor.buy_quote_amount(symbol=symbol, quote_amount=amount_eur)
        row.status     = "success"
        row.order_id   = result.get("order_id")
        row.filled_btc = result.get("filled_btc")
        row.cost_eur   = result.get("cost_eur")
        row.avg_price  = result.get("avg_price")
        row.fee_json   = json.dumps(result.get("fee")) if result.get("fee") else None
        db.commit()
        return {"status": "success", **result}

    except Exception as exc:
        row.status = "failed"
        row.error  = str(exc)
        db.commit()
        raise HTTPException(status_code=502, detail=str(exc))


# ─── History ──────────────────────────────────────────────────────────────────

@router.get("/dca/history")
def dca_history(
    limit: int = Query(50, le=500),
    db: Session = Depends(get_db),
):
    """All past DCA executions, most recent first."""
    rows = (
        db.query(DCAExecution)
        .order_by(DCAExecution.executed_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id":           r.id,
            "executed_at":  r.executed_at.isoformat(),
            "symbol":       r.symbol,
            "quote_amount": r.quote_amount,
            "status":       r.status,
            "order_id":     r.order_id,
            "filled_btc":   r.filled_btc,
            "cost_eur":     r.cost_eur,
            "avg_price":    r.avg_price,
            "testnet":      r.testnet,
            "error":        r.error,
        }
        for r in rows
    ]


# ─── Summary ──────────────────────────────────────────────────────────────────

@router.get("/dca/summary")
def dca_summary(db: Session = Depends(get_db)):
    """
    Aggregate stats across all successful executions.
    Shows total BTC accumulated, total EUR spent, and average cost basis.
    """
    rows = (
        db.query(DCAExecution)
        .filter(DCAExecution.status == "success")
        .all()
    )

    if not rows:
        return {"message": "No successful executions yet."}

    total_eur  = sum(r.cost_eur   or 0 for r in rows)
    total_btc  = sum(r.filled_btc or 0 for r in rows)
    avg_cost   = total_eur / total_btc if total_btc > 0 else None
    n_buys     = len(rows)
    first_buy  = min(r.executed_at for r in rows)
    last_buy   = max(r.executed_at for r in rows)

    return {
        "n_executions":      n_buys,
        "total_eur_spent":   round(total_eur, 2),
        "total_btc_bought":  round(total_btc, 8),
        "avg_cost_basis_eur": round(avg_cost, 2) if avg_cost else None,
        "first_buy":         first_buy.isoformat(),
        "last_buy":          last_buy.isoformat(),
        "testnet_only":      all(r.testnet for r in rows),
    }


# ─── Binance connectivity ─────────────────────────────────────────────────────

@router.get("/binance/balance")
def binance_balance():
    """Check account balance — useful to verify API keys are working."""
    executor = _get_executor()
    try:
        return executor.check_connection()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.get("/binance/ticker")
def binance_ticker(symbol: str = Query("BTC/EUR")):
    """Fetch current market price for a symbol."""
    executor = _get_executor()
    try:
        return executor.get_ticker(symbol)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
