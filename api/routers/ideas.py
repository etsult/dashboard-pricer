"""
Router: /ideas and /portfolios

FastAPI concepts demonstrated here:
  - Depends(get_db): injects a DB session per request, closes it automatically
  - Path parameters: /ideas/{idea_id}
  - Query parameters: /ideas?status=live&asset_class=crypto
  - 404 handling with HTTPException
  - Combining a backtest run with DB persistence in one endpoint
"""

from __future__ import annotations

import asyncio
import json
from functools import partial
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from storage.database import get_db
from storage import repository as repo
from api.schemas.ideas import (
    IdeaCreate, IdeaUpdate, IdeaOut, NoteCreate, NoteOut,
    BacktestSummary,
    PortfolioCreate, PortfolioAddIdea, PortfolioOut, PortfolioIdeaOut,
)
from api.schemas.research import DHStraddleRequest
from research.backtest import run_dh_straddle
from research.costs import CostModel


router = APIRouter(tags=["Ideas & Portfolios"])


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _idea_to_out(idea) -> IdeaOut:
    return IdeaOut(
        id=idea.id,
        created_at=idea.created_at,
        updated_at=idea.updated_at,
        name=idea.name,
        description=idea.description,
        asset_class=idea.asset_class,
        underlying=idea.underlying,
        strategy_type=idea.strategy_type,
        direction=idea.direction,
        status=idea.status,
        conviction=idea.conviction,
        parameters=idea.parameters_dict,
        tags=idea.tags_list,
        backtest_results=[
            BacktestSummary(
                id=r.id,
                run_at=r.run_at,
                sharpe=r.sharpe,
                sortino=r.sortino,
                total_pnl=r.total_pnl,
                max_drawdown=r.max_drawdown,
                win_rate=r.win_rate,
                annotation=r.annotation,
            )
            for r in idea.backtest_results
        ],
        notes=[NoteOut(id=n.id, created_at=n.created_at, body=n.body) for n in idea.notes],
    )


def _get_or_404(db: Session, idea_id: int):
    idea = repo.get_idea(db, idea_id)
    if not idea:
        raise HTTPException(status_code=404, detail=f"Trade idea {idea_id} not found")
    return idea


# ─── Trade Ideas ──────────────────────────────────────────────────────────────

@router.post("/ideas", response_model=IdeaOut, status_code=201)
def create_idea(body: IdeaCreate, db: Session = Depends(get_db)) -> IdeaOut:
    """Create a new trade idea. Status starts at 'idea'."""
    try:
        idea = repo.create_idea(db, **body.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _idea_to_out(idea)


@router.get("/ideas", response_model=list[IdeaOut])
def list_ideas(
    status: Optional[str] = Query(None, description="Filter by status"),
    asset_class: Optional[str] = Query(None),
    tag: Optional[str] = Query(None, description="Filter by tag (partial match)"),
    db: Session = Depends(get_db),
) -> list[IdeaOut]:
    """List all trade ideas with optional filters."""
    ideas = repo.list_ideas(db, status=status, asset_class=asset_class, tag=tag)
    return [_idea_to_out(i) for i in ideas]


@router.get("/ideas/{idea_id}", response_model=IdeaOut)
def get_idea(idea_id: int, db: Session = Depends(get_db)) -> IdeaOut:
    """Get a single idea with all its backtest results and notes."""
    return _idea_to_out(_get_or_404(db, idea_id))


@router.patch("/ideas/{idea_id}", response_model=IdeaOut)
def update_idea(idea_id: int, body: IdeaUpdate, db: Session = Depends(get_db)) -> IdeaOut:
    """Update an idea's status, conviction, parameters, or tags."""
    _get_or_404(db, idea_id)
    updated = repo.update_idea(db, idea_id, **body.model_dump(exclude_none=True))
    return _idea_to_out(updated)


@router.delete("/ideas/{idea_id}", status_code=204)
def archive_idea(idea_id: int, db: Session = Depends(get_db)):
    """Archive an idea (soft delete — keeps all history)."""
    _get_or_404(db, idea_id)
    repo.archive_idea(db, idea_id)


# ─── Notes ────────────────────────────────────────────────────────────────────

@router.post("/ideas/{idea_id}/notes", response_model=NoteOut, status_code=201)
def add_note(idea_id: int, body: NoteCreate, db: Session = Depends(get_db)) -> NoteOut:
    """Add a research note to an idea (observations, regime conditions, improvements)."""
    _get_or_404(db, idea_id)
    note = repo.add_note(db, idea_id, body.body)
    return NoteOut(id=note.id, created_at=note.created_at, body=note.body)


# ─── Backtest (run + persist) ─────────────────────────────────────────────────

@router.post("/ideas/{idea_id}/backtest", status_code=201)
async def run_and_attach_backtest(
    idea_id: int,
    req: DHStraddleRequest,
    annotation: Optional[str] = Query(None, description="Optional note on this run"),
    db: Session = Depends(get_db),
):
    """
    Run a backtest for this idea and persist the result.
    The idea's status auto-advances to 'backtested' on first run.

    This endpoint combines two things:
      1. POST /research/backtest/dh-straddle  (compute)
      2. Persist the result against this specific idea  (store)
    """
    _get_or_404(db, idea_id)

    costs = CostModel(
        spread_pct=req.costs.spread_pct,
        commission_pct=req.costs.commission_pct,
        slippage_pct=req.costs.slippage_pct,
        funding_rate_daily=req.costs.funding_rate_daily,
    )

    loop = asyncio.get_event_loop()
    fn = partial(
        run_dh_straddle,
        currency=req.currency,
        history_days=req.history_days,
        T_days=req.T_days,
        rebalance_freq=req.rebalance_freq,
        notional_usd=req.notional_usd,
        costs=costs,
    )

    try:
        result = await loop.run_in_executor(None, fn)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Backtest error: {exc}")

    # Persist net performance (after costs) as the canonical result
    net_metrics = result["net_performance"]
    bt = repo.attach_backtest(
        db,
        idea_id=idea_id,
        metrics=net_metrics,
        parameters_snapshot=req.model_dump(exclude={"costs"}),
        cost_model_snapshot=req.costs.model_dump(),
        daily_pnl=[
            {"date": r["date"], "pnl": r["net_total_pnl"]}
            for r in result["daily"]
            if "net_total_pnl" in r
        ],
        annotation=annotation,
    )

    return {
        "backtest_id": bt.id,
        "idea_id": idea_id,
        "gross_performance": result["gross_performance"],
        "net_performance": net_metrics,
        "cost_summary": result["cost_summary"],
    }


# ─── Portfolios ───────────────────────────────────────────────────────────────

@router.post("/portfolios", response_model=PortfolioOut, status_code=201)
def create_portfolio(body: PortfolioCreate, db: Session = Depends(get_db)) -> PortfolioOut:
    """Create a named portfolio to group and combine ideas."""
    p = repo.create_portfolio(db, **body.model_dump())
    return _portfolio_to_out(p, db)


@router.get("/portfolios", response_model=list[PortfolioOut])
def list_portfolios(db: Session = Depends(get_db)) -> list[PortfolioOut]:
    return [_portfolio_to_out(p, db) for p in repo.list_portfolios(db)]


@router.get("/portfolios/{portfolio_id}", response_model=PortfolioOut)
def get_portfolio(portfolio_id: int, db: Session = Depends(get_db)) -> PortfolioOut:
    p = repo.get_portfolio(db, portfolio_id)
    if not p:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return _portfolio_to_out(p, db)


@router.post("/portfolios/{portfolio_id}/ideas", status_code=201)
def add_idea_to_portfolio(
    portfolio_id: int, body: PortfolioAddIdea, db: Session = Depends(get_db)
):
    """Add a trade idea to a portfolio."""
    if not repo.get_portfolio(db, portfolio_id):
        raise HTTPException(status_code=404, detail="Portfolio not found")
    _get_or_404(db, body.idea_id)
    repo.add_idea_to_portfolio(db, portfolio_id, body.idea_id, body.notional_usd)
    return {"status": "added"}


@router.delete("/portfolios/{portfolio_id}/ideas/{idea_id}", status_code=204)
def remove_idea_from_portfolio(
    portfolio_id: int, idea_id: int, db: Session = Depends(get_db)
):
    removed = repo.remove_idea_from_portfolio(db, portfolio_id, idea_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Idea not in portfolio")


def _portfolio_to_out(p, db: Session) -> PortfolioOut:
    ideas_out = []
    for entry in p.entries:
        best = repo.get_best_backtest(db, entry.idea_id)
        ideas_out.append(PortfolioIdeaOut(
            idea_id=entry.idea_id,
            idea_name=entry.idea.name,
            idea_status=entry.idea.status,
            notional_usd=entry.notional_usd,
            weight=entry.weight,
            best_sharpe=best.sharpe if best else None,
        ))
    return PortfolioOut(
        id=p.id,
        name=p.name,
        description=p.description,
        construction_method=p.construction_method,
        created_at=p.created_at,
        ideas=ideas_out,
    )
