#!/usr/bin/env python3
"""Run AS model backtest across 5 assets and report PnL metrics."""
import sys
sys.path.insert(0, '.')

from research.market_making.data_loader import load_ohlcv_synthetic
from research.market_making.backtest import BacktestConfig, run_backtest, gamma_sensitivity, kappa_sensitivity
import numpy as np

assets = [
    ("SOL/USDT",  "binance", 0.10, 3333.0, 5.0,  0.10),
    ("AVAX/USDT", "binance", 0.10, 1666.0, 5.0,  0.10),
    ("BNB/USDT",  "binance", 0.10, 2500.0, 5.0,  0.10),
    ("BTC/USDT",  "binance", 0.05, 5000.0, 0.1,  0.001),
    ("ETH/USDT",  "binance", 0.10, 4000.0, 1.0,  0.01),
]

ref_prices = {
    "SOL/USDT":  150.0,
    "AVAX/USDT": 35.0,
    "BNB/USDT":  400.0,
    "BTC/USDT":  60000.0,
    "ETH/USDT":  3000.0,
}

daily_vols = {
    "SOL/USDT":  0.045,
    "AVAX/USDT": 0.055,
    "BNB/USDT":  0.035,
    "BTC/USDT":  0.025,
    "ETH/USDT":  0.030,
}

hdr = "{:<12} {:>10} {:>10} {:>10} {:>10} {:>7} {:>8} {:>9} {:>6} {:>9}"
print(hdr.format("Asset", "Capital", "TotalPnL", "DailyPnL", "DailyStd", "Sharpe", "MaxDD", "AnnRet%", "Fills", "FillRate"))
print("-" * 100)

row_fmt = "{:<12} {:>10.0f} {:>10.2f} {:>10.4f} {:>10.4f} {:>7.2f} {:>8.2f} {:>9.1f} {:>6} {:>9.1f}"

for sym, exch, gamma, kappa, max_inv, order_sz in assets:
    price = ref_prices[sym]
    df = load_ohlcv_synthetic(sym, n_bars=43200, bar_minutes=1,
                               sigma_daily=daily_vols[sym], seed=42)
    cfg = BacktestConfig(
        symbol=sym, exchange=exch, model="AS_Gueant",
        gamma=gamma, kappa=kappa, T_bars=60.0,
        max_inventory=max_inv, order_size=order_sz,
        maker_fee=-0.0001, taker_fee=0.0004,
        flat_at_end=True, initial_cash=0.0,
        vol_window_bars=60, vol_floor_frac=1e-5,
    )
    r = run_backtest(df, cfg)
    m = r.metrics

    capital   = max_inv * price
    total_pnl = m.get("total_pnl", 0.0)
    daily_pnl = m.get("daily_pnl_mean", 0.0)
    daily_std = m.get("daily_pnl_std", 0.0)
    sharpe    = m.get("sharpe_daily", 0.0)
    max_dd    = m.get("max_drawdown", 0.0)
    n_fills   = m.get("n_fills", 0)
    fill_rate = m.get("fill_rate_pct", 0.0)
    ann_ret   = (daily_pnl * 365 / capital * 100) if capital > 0 else 0.0

    print(row_fmt.format(sym, capital, total_pnl, daily_pnl, daily_std,
                          sharpe, max_dd, ann_ret, n_fills, fill_rate))

# Gamma sweep on SOL
print("\n=== SOL/USDT gamma sensitivity ===")
df_sol = load_ohlcv_synthetic("SOL/USDT", n_bars=43200, bar_minutes=1, sigma_daily=0.045, seed=42)
cfg_sol = BacktestConfig(symbol="SOL/USDT", exchange="binance", model="AS_Gueant",
    gamma=0.10, kappa=3333.0, T_bars=60.0, max_inventory=5.0, order_size=0.10)
rows = gamma_sensitivity(df_sol, cfg_sol)
hdr2 = "{:<8} {:>10} {:>10} {:>7} {:>7} {:>9}"
print(hdr2.format("gamma", "TotalPnL", "DailyPnL", "Sharpe", "nFills", "FillRate"))
for row in rows:
    print("{:<8} {:>10.2f} {:>10.4f} {:>7.2f} {:>7} {:>9.1f}".format(
        row["gamma"], row.get("total_pnl", 0), row.get("daily_pnl_mean", 0),
        row.get("sharpe_daily", 0), row.get("n_fills", 0), row.get("fill_rate_pct", 0)))

# Kappa sweep on SOL
print("\n=== SOL/USDT kappa sensitivity ===")
rows_k = kappa_sensitivity(df_sol, cfg_sol)
hdr3 = "{:<8} {:>7} {:>10} {:>7} {:>7} {:>9}"
print(hdr3.format("kappa", "hs_bps", "TotalPnL", "Sharpe", "nFills", "FillRate"))
for row in rows_k:
    print("{:<8} {:>7.1f} {:>10.2f} {:>7.2f} {:>7} {:>9.1f}".format(
        row["kappa"], row.get("target_hs_bps", 0), row.get("total_pnl", 0),
        row.get("sharpe_daily", 0), row.get("n_fills", 0), row.get("fill_rate_pct", 0)))
