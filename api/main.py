"""
FastAPI application entry point.

Run with:
  uvicorn api.main:app --reload --port 8000

Then open:
  http://localhost:8000/docs   ← interactive Swagger UI (try every endpoint live)
  http://localhost:8000/redoc  ← alternative ReDoc documentation

FastAPI concepts demonstrated here:
  - FastAPI(): the main application object
  - app.include_router(): composing the app from multiple routers
  - lifespan: startup/shutdown hooks (loading .env, etc.)
  - CORS middleware: required when a browser frontend calls this API
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import strategies, ir_options, market_data, research, ideas, execution
from api.routers import books, vol_cube_router, ws, amm


# ─── Lifespan (startup / shutdown) ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code before `yield` runs at startup.
    Code after `yield` runs at shutdown.

    Here we load environment variables (FRED_API_KEY, etc.) from .env.
    """
    from dotenv import load_dotenv
    load_dotenv()
    # Create all DB tables if they don't exist yet
    from storage.database import Base, engine
    from storage import models  # noqa: F401 — registers all models with Base
    Base.metadata.create_all(bind=engine)
    # Start the DCA scheduler
    from execution.scheduler import start_scheduler, stop_scheduler
    start_scheduler()
    yield
    stop_scheduler()
    # nothing to clean up for now


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Dashboard Pricer API",
    description=(
        "REST API exposing the option pricing engine.\n\n"
        "**Endpoints:**\n"
        "- `/strategies/price` — multi-leg EQD strategy pricer\n"
        "- `/ir/curve` — yield curve bootstrapping\n"
        "- `/ir/cap-floor` — cap/floor pricer\n"
        "- `/ir/swaption` — swaption pricer\n"
        "- `/market/vol-term-structure` — live crypto vol from Deribit\n"
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ─── CORS ─────────────────────────────────────────────────────────────────────
# Allows a browser frontend (React, etc.) running on a different port to call this API.
# In production you'd lock down `allow_origins` to your actual frontend URL.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React / Vite dev servers
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Routers ──────────────────────────────────────────────────────────────────
# Each router is a mini-app. Including it here mounts all its routes under /api.

app.include_router(strategies.router,       prefix="/api")
app.include_router(ir_options.router,       prefix="/api")
app.include_router(books.router,            prefix="/api")
app.include_router(vol_cube_router.router,  prefix="/api")
app.include_router(market_data.router,      prefix="/api")
app.include_router(research.router,         prefix="/api")
app.include_router(ideas.router,            prefix="/api")
app.include_router(execution.router,        prefix="/api")
app.include_router(ws.router)               # WebSocket — no /api prefix (uses /ws/...)
app.include_router(amm.router,              prefix="/api")


# ─── Root ─────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return {"message": "Dashboard Pricer API", "docs": "/docs"}
