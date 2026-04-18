"""
APScheduler setup for recurring DCA jobs.

The scheduler lives inside the FastAPI process — it fires as long as
uvicorn is running. For production you'd use a separate worker process
(Celery, or a cron job calling the API endpoint), but for a personal
DCA plan this is perfectly reliable.

Schedule: 1st of every month at 09:00 UTC.
Configurable via environment variables:
  DCA_DAY_OF_MONTH   (default: 1)
  DCA_HOUR_UTC       (default: 9)
  DCA_AMOUNT_EUR     (default: 100)
  DCA_SYMBOL         (default: BTC/EUR)
  DCA_TESTNET        (default: true — set to "false" for live)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

log = logging.getLogger(__name__)

scheduler = AsyncIOScheduler(timezone="UTC")


async def _run_dca_job() -> None:
    """The actual job — called by APScheduler on schedule."""
    from execution.binance_dca import executor_from_env
    from storage.database import SessionLocal
    from storage.models import DCAExecution

    symbol       = os.environ.get("DCA_SYMBOL", "BTC/EUR")
    amount       = float(os.environ.get("DCA_AMOUNT_EUR", "100"))
    testnet      = os.environ.get("DCA_TESTNET", "true").lower() != "false"

    log.info("DCA job fired — %s %.2f EUR testnet=%s", symbol, amount, testnet)

    db = SessionLocal()
    row = DCAExecution(
        symbol=symbol,
        quote_amount=amount,
        testnet=testnet,
        status="pending",
    )
    db.add(row)
    db.commit()

    try:
        executor = executor_from_env(testnet=testnet)
        result   = executor.buy_quote_amount(symbol=symbol, quote_amount=amount)

        row.status     = "success"
        row.order_id   = result.get("order_id")
        row.filled_btc = result.get("filled_btc")
        row.cost_eur   = result.get("cost_eur")
        row.avg_price  = result.get("avg_price")
        row.fee_json   = json.dumps(result.get("fee")) if result.get("fee") else None
        log.info("DCA success: %.6f BTC @ %.2f EUR", row.filled_btc or 0, row.avg_price or 0)

    except Exception as exc:
        row.status = "failed"
        row.error  = str(exc)
        log.error("DCA failed: %s", exc)

    finally:
        db.commit()
        db.close()


def start_scheduler() -> None:
    """Register the monthly DCA job and start the scheduler."""
    day  = int(os.environ.get("DCA_DAY_OF_MONTH", "1"))
    hour = int(os.environ.get("DCA_HOUR_LOCAL", "20"))
    tz   = os.environ.get("DCA_TIMEZONE", "Europe/Paris")

    scheduler.add_job(
        _run_dca_job,
        trigger=CronTrigger(day=day, hour=hour, minute=0, timezone=tz),
        id="monthly_dca",
        name="Monthly BTC DCA",
        replace_existing=True,
        misfire_grace_time=3600,
    )

    scheduler.start()
    log.info(
        "Scheduler started — DCA fires on day %d of each month at %02d:00 %s",
        day, hour, tz,
    )


def stop_scheduler() -> None:
    if scheduler.running:
        scheduler.shutdown()
