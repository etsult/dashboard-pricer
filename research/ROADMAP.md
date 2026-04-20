# Quant Research Lab — Roadmap

## 1. Neural Network Pricers

### 1.1 Swaption / Cap-Floor NN Pricer  *(in progress)*
- **Architecture**: feed-forward network trained on (F, K, σ_N, τ, ann) → PV
- **Training data**: ~10M synthetic Bachelier prices from the fast engine
- **Status**: Tier-1 model built; training pipeline exists at `pricer/nn/`

### 1.2 Differential Machine Learning (Savine / arXiv 2005.02347)
- **Learning from payoffs** (not just prices): train on individual MC paths with pathwise AAD derivatives
  - Each path gives N+1 training signals: one price + N sensitivities
  - Regularisation: `λ × ‖∂NN/∂x − ∂true/∂x‖²` term in the loss
  - Benefit: dramatically fewer paths needed for a smooth, accurate model
- **Learning from samples** (simpler baseline): pre-compute (price, greeks) pairs, train on them directly
- **Target instruments**: swaptions first, then Bermudan swaptions (much harder — requires LSM or AAD-PDE)
- **Key challenge**: business day conventions, reset calendars, gap risk — the NN learns from the *closed-form* pricer not the actual schedule, so edge-case dates (weekend resets, stub periods) are not captured unless the training includes them as features

### 1.3 Exotic NN Pricers
- CMS spread options (SABR CMS replication benchmark)
- Bermudan swaptions (LSM + differential ML)
- Callable bonds / range accruals (longer horizon)

---

## 2. Rate Models

### 2.1 Hull-White (1-factor) — *next priority for Intraday Greeks*
- Calibrate `κ` (mean reversion) and `σ` (vol) to swaption market quotes
- Replace current BM/OU simulation in `/ws/intraday-greeks`
- Deliverable: `pricer/ir/hull_white.py` + WebSocket param `rate_model=hw`

### 2.2 2-Factor Hull-White / G2++
- Second factor adds curve twist dynamics
- Required for multi-bucket DV01 simulation to be realistic
- Calibrate jointly to short-end and long-end swaptions

### 2.3 SABR / ZABR Surface
- ZABR (β=0, normal backbone) per expiry slice already implemented in `ql_greeks_benchmark.py`
- Wire into the live pricer: replace flat σ_N with ZABR smile-interpolated vol
- Vol-rate correlation `ρ_σr ≈ −0.3 to −0.5` for USD: implement stochastic vol-rate in intraday simulation

---

## 3. Data Infrastructure

### 3.1 Real-Time Rate Data
- Connect to a free provider (options: FRED streaming, ECB SDW, Quandl/Nasdaq Data Link free tier)
- Target: live USD/EUR SOFR + swaption ATM vol every 5 minutes
- TODO: evaluate `refinitiv_data` SDK (free research tier)

### 3.2 Deribit Crypto Options (already partially connected)
- Live BTC/ETH option chains via `market_data/providers/deribit.py`
- Extend: SOL options, funding rate curve, perpetual basis

### 3.3 Vol Cube Bootstrapping
- Build a full 3D vol cube (expiry × tenor × strike) from market data
- Store as `market_data/curves/vol_cube.py`
- Feed into ZABR calibration per slice

---

## 4. Portfolio Analytics

### 4.1 60/40 Multi-Country Portfolio Study
- Instruments: equity ETFs + government bond ETFs across US, EU, JP, EM
- Metrics: Sharpe, Sortino, Calmar, max drawdown, CVaR, rolling beta, tracking error
- Stress scenarios: 2008, 2020, 2022 rate shock, EM crisis

### 4.2 Rolling Risk Metrics (Strategy Compare page)
- Rolling Sharpe (21d, 63d, 252d windows)
- Rolling Sortino, rolling Calmar
- Rolling max drawdown and time underwater
- Rolling realised vol and beta vs SPX
- Correlation matrix heatmap (assets × assets)

### 4.3 Risk Factor Attribution
- PCA on returns → explain portfolio variance by factor
- Macro factor loading: rates, credit, equity, FX

---

## 5. Execution & Risk Infrastructure

### 5.1 Intraday Greeks — Full Real-Time Loop
- Replace OU rate simulation with calibrated Hull-White
- Add vol-rate correlation `dσ_N = ρ × σ_vov × dW_rates + √(1−ρ²) × σ_vov × dW_⊥`
- Live DV01 matrix update every tick with colour-coded P&L attribution

### 5.2 Delta Hedging Module (`pages/6_DeltaHedge.py`)
- Discrete delta hedge simulation: hedge at fixed intervals or delta threshold
- Track hedging P&L, residual gamma, slippage cost
- Compare: daily hedge vs weekly hedge vs threshold-trigger

### 5.3 ORE (Open Risk Engine) Integration
- Counterparty credit risk: CVA, DVA, FVA for IR swap portfolios
- Initial margin via ISDA SIMM
- Long-horizon scenario generation (Monte Carlo over 10Y horizon)

### 5.4 Strata / QuantLib Pricing Consistency
- Validate fast Bachelier engine vs QuantLib Bachelier on 50k instrument sample
- Build a nightly regression suite: `tests/regression/test_pricer_consistency.py`

---

## 6. Frontend / UX

### 6.1 Curve Editor
- Interactive yield curve editor (drag pillars) → live reprice of entire book
- Show zero, forward, and par curves simultaneously

### 6.2 Vol Surface Explorer
- Interactive 3D surface (expiry × strike × vol) with smile animations
- SABR smile overlay vs raw market quotes

### 6.3 Scenario Manager
- Save named curve/vol scenarios
- Overlay P&L across scenarios on one chart
- Stress test button: shock rates ±100bp, flatten/steepen, vol crush

---

## Priority Order (April 2026)

| Priority | Item | Est. effort |
|----------|------|-------------|
| 1 | Hull-White rate model for intraday simulation | 2–3 days |
| 2 | Real-time data connection (FRED streaming / Deribit vol) | 1 day |
| 3 | Differential ML swaption pricer (Savine) | 1–2 weeks |
| 4 | Rolling risk metrics in Strategy Compare | 1 day |
| 5 | 60/40 multi-country portfolio study | 2–3 days |
| 6 | ZABR smile wired into live pricer | 2 days |
| 7 | ORE CVA/DVA integration | 1–2 weeks |
