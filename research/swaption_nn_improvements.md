# SwaptionNet — Improvements Roadmap & Research Bibliography

---

## How to Improve the Model

### Priority 1 — Training Signal

#### 1.1 Differential Machine Learning (biggest impact)
The single most important improvement. Instead of supervising only on price,
add **differential labels** — supervise the NN's own gradients against analytical Greeks.

```python
# During training: compute NN gradient w.r.t. inputs
cont.requires_grad_(True)
pred = model(cont, conv, strata)
grads = torch.autograd.grad(pred[:,0].sum(), cont, create_graph=True)[0]

# Supervise gradient[:,0] (moneyness) against analytical delta
# Supervise gradient[:,4] (sigma_atm)  against analytical vega
greek_loss = mse(grads[:,0], target_delta) + mse(grads[:,4], target_vega)
total = price_loss + lambda_greek * greek_loss
```

**Why**: Greeks are free information — the Bachelier formula gives exact derivatives.
Forcing the NN's derivatives to match reduces price error by ~5-10× at same network size.
The model learns the *shape* of the pricing function, not just point values.
See: Huge & Savine (2020).

#### 1.2 Importance Sampling for Tails
Current deep-OTM P95 error is ~4 bps. The W-shaped error curve (visible in plots)
shows the model undersamples |d| > 3. Fix:

```python
# Oversample wings: weight samples by inverse density
weights = 1.0 / (norm.pdf(moneyness) + eps)
weights /= weights.mean()
sampler = WeightedRandomSampler(weights, len(weights))
loader  = DataLoader(ds, sampler=sampler, batch_size=4096)
```

Or use a **stratified sampler** with 4 bins on |d|, guaranteed count per bin per batch.

#### 1.3 Quantile-Calibrated Loss
Current pinball loss bounds P10/P90 but not extreme quantiles.
Add P2 and P98 terms to cap worst-case errors:

```python
loss = (mse_price
      + 0.15 * (pinball(pred, true, 0.02) + pinball(pred, true, 0.98))
      + 0.15 * (pinball(pred, true, 0.10) + pinball(pred, true, 0.90)))
```

---

### Priority 2 — Architecture

#### 2.1 Separate Short-Expiry Head
The heatmap shows T < 1Y is hardest. Add a **mixture of experts**:
- Gate network outputs weight between `general_head` and `short_expiry_head`
- Short-expiry expert is a deeper MLP trained only on T < 1Y samples
- Soft routing: `price = gate * short_expert(x) + (1-gate) * general_expert(x)`

#### 2.2 Transformer / Self-Attention on Book
When pricing a full book (not single instruments), attention across instruments
lets the model learn cross-instrument correlations (curve shape, spread term structure).
Useful for aggregated risk, not single-instrument inference.

#### 2.3 Normalising Flows for Uncertainty
Replace point estimate with a conditional normalising flow:
output = distribution over price given inputs.
Gives calibrated confidence intervals — useful for model risk reporting.

#### 2.4 Input Feature Engineering
Add features the model currently cannot derive:
- `d² = moneyness²` — explicit quadratic curvature for the wings
- `sqrt_T` — separate from T, helps with theta/time-decay shape
- `tenor_T_ratio = tenor / T` — how long the swap is relative to expiry
- `atm_skew = -rho * nu` — SABR skew parameter, directly linked to delta smile
- `atm_curvature = nu²` — SABR curvature, linked to vanna/volga

---

### Priority 3 — Data Generation

#### 3.1 Full ZABR Surface Calibration
Replace the smooth parametric surface with **historically calibrated surfaces**:
1. Build ZABR surface from swaption vol cube (ICAP/Bloomberg data)
2. Fit alpha/nu/rho per (expiry, tenor) bucket using Levenberg-Marquardt
3. Use calibrated params as base, perturb for training diversity

This makes training data distribution match real market conditions exactly.

#### 3.2 Multi-Factor Curve Shapes
Current two-curve model uses flat rates. Add:
- Nelson-Siegel factor curves (level, slope, curvature)
- Forward curve inversion (currently impossible with flat model)
- Basis volatility between OIS and index curve

#### 3.3 Convexity Adjustments
For CMS (Constant Maturity Swap) swaptions, add:
- CMS convexity adjustment as an additional feature
- Replication-based adjustment (Hagan 2003) as analytical target

---

### Priority 4 — Production Hardening

#### 4.1 Model Validation Framework
```python
# Automated daily checks:
# 1. Put-call parity: |call - put - (F-K)*annuity*df| < 0.01bps
# 2. Monotonicity: call price monotone in F, monotone in sigma
# 3. No-arbitrage: butterfly spread > 0 for all strikes
# 4. Greek bounds: 0 < delta_call < 1, gamma > 0, vega > 0
```

#### 4.2 Conformal Prediction Wrapper
Wrap the NN with conformal prediction for distribution-free coverage guarantees:
```python
# Calibration set residuals → coverage threshold
# At inference: return (point_estimate ± coverage_interval)
# Guaranteed 95% coverage with no distributional assumptions
```

#### 4.3 Continuous Retraining
When ZABR surface is recalibrated (daily), fine-tune the NN on new surface
rather than retraining from scratch. Use EWC (Elastic Weight Consolidation)
to prevent catastrophic forgetting of old regimes.

---

## Research Papers

### Foundational — Neural Networks for Derivatives

| Paper | Authors | Year | Key Contribution |
|---|---|---|---|
| *A Nonparametric Approach to Pricing and Hedging Derivative Securities Via Learning Networks* | Hutchinson, Lo, Poggio | 1994 | First NN option pricer; proved NNs can learn Black-Scholes |
| *Managing Smile Risk* | Hagan, Kumar, Lesniewski, Woodward | 2002 | SABR model — the foundation of vol surface parameterisation |
| *ZABR — Expansions for the Masses* | Antonov, Spector, Piterbarg | 2011 | ZABR generalisation of SABR; normal vol approximation used here |
| *Model Calibration with Neural Networks* | Hernandez | 2017 | First production use of NN for model calibration on a rates desk |

---

### Core Papers — Directly Relevant to This Pipeline

#### Differential Machine Learning ⭐ (most important)
**Huge, B. & Savine, A. (2020)**
*Differential Machine Learning*
arXiv: [2005.02347](https://arxiv.org/abs/2005.02347)

> Trains NNs on both prices AND their derivatives w.r.t. inputs (differential labels).
> Analytical Greeks are used as free supervision signal — convergence is 10-100× faster.
> Direct implementation blueprint for Priority 1.1 above.
> **Read this first.**

---

#### Deep Learning Volatility ⭐
**Horvath, B., Muguruza, A. & Tomas, M. (2021)**
*Deep Learning Volatility*
arXiv: [1901.09647](https://arxiv.org/abs/1901.09647)
Published in: *Quantitative Finance*

> NN calibration of implied vol surfaces in milliseconds, including rough vol models.
> Treats the vol surface as an image and trains a CNN on it.
> Directly applicable to ZABR surface calibration acceleration.

---

#### Deep Hedging ⭐
**Bühler, H., Gonon, L., Teichmann, J. & Wood, B. (2018)**
*Deep Hedging*
arXiv: [1802.03042](https://arxiv.org/abs/1802.03042)
Published in: *Quantitative Finance* (2019)

> Reinforcement learning for hedging under transaction costs and liquidity constraints.
> Replaces delta-hedging with a learned policy that handles real-world frictions.
> Relevant when extending this pricer to a hedging engine.

---

#### Physics-Informed Neural Networks
**Raissi, M., Perdikaris, P. & Karniadakis, G.E. (2017)**
*Physics Informed Deep Learning: Data-driven Solutions of Nonlinear PDEs*
arXiv: [1711.10561](https://arxiv.org/abs/1711.10561)
Published in: *Journal of Computational Physics* (2019)

> Encode PDE constraints (Black-Scholes PDE, no-arbitrage conditions) directly in loss.
> Directly applicable: add `∂V/∂T + 0.5σ²∂²V/∂F² = 0` as a loss penalty.
> Enforces arbitrage-free pricing without labelled data.

---

#### Deep Hedging — Market Simulation
**Wiese, M., Bai, L., Wood, B. & Buehler, H. (2019)**
*Deep Hedging: Learning to Simulate Equity Option Markets*
arXiv: [1911.01700](https://arxiv.org/abs/1911.01700)
NeurIPS 2019 Workshop

> GAN-based market simulator for option price paths.
> Useful for generating more realistic training data than parametric ZABR surfaces.

---

#### Deep Hedging — Risk-Neutral Dynamics
**Buehler, H., Murray, P., Pakkanen, M.S. & Wood, B. (2021)**
*Deep Hedging: Learning Risk-Neutral Implied Volatility Dynamics*
arXiv: [2103.11948](https://arxiv.org/abs/2103.11948)

> Learns risk-neutral measure from simulated paths, handles transaction costs.
> Extension of Deep Hedging to vol dynamics — relevant for multi-period swaption books.

---

### Volatility Models & Smile

| Paper | Authors | Year | Where to Find |
|---|---|---|---|
| *Arbitrage-Free SABR* | Hagan, Lesniewski, Woodward | 2014 | Wilmott Magazine |
| *Effective Parameters for SABR* | Obloj | 2008 | arXiv:0708.2339 |
| *A New Simple Approach for Constructing Implied Volatility Surfaces* | Le Floc'h, Kennedy | 2014 | SSRN 2266396 |
| *Rough Volatility* | El Euch, Rosenbaum | 2019 | *Mathematical Finance* / arXiv:1609.02108 |
| *Volatility is (Mostly) Path-Dependent* | Guyon, Lekeufack | 2023 | *Quantitative Finance* / arXiv:2305.00556 |

---

### Calibration & Surrogate Models

| Paper | Authors | Year | Key Idea |
|---|---|---|---|
| *An Artificial Neural Network Representation of the SABR Model* | McGhee | 2018 | SSRN 3288882 — NN as SABR surrogate, direct predecessor to this work |
| *Machine Learning for Quantitative Finance* | Ruf, Wang | 2020 | arXiv:2012.04180 — survey, hedging with NNs |
| *Chebyshev Tensors for Derivative Pricing* | Gauthier, Rivain | 2021 | Polynomial surrogates — faster than NN for low-dimensional problems |
| *Neural Networks for Option Pricing and Hedging* | Liu, Oosterlee, Bohte | 2019 | arXiv:1901.08943 — systematic comparison of NN architectures |

---

### Automatic Differentiation (AAD) — for Greek computation

| Paper | Authors | Year | Note |
|---|---|---|---|
| *Smoking Adjoints: Fast Monte Carlo Greeks* | Giles, Glasserman | 2006 | *Risk Magazine* — original AAD for finance |
| *Adjoint Methods in Computational Finance* | Homescu | 2011 | arXiv:1107.1892 — survey of AAD for Greeks |
| *Modern Computational Finance: AAD and Parallel Simulations* | Savine | 2018 | Book (Wiley) — best practical reference for AAD in C++ |

---

### Uncertainty Quantification

| Paper | Authors | Year | Key Idea |
|---|---|---|---|
| *Conformal Prediction: a Unified Review* | Angelopoulos, Bates | 2021 | arXiv:2107.07511 — distribution-free coverage guarantees |
| *Deep Ensembles* | Lakshminarayanan et al | 2017 | arXiv:1612.01474 — simple, effective uncertainty via ensemble |
| *MC Dropout as Bayesian Approximation* | Gal, Ghahramani | 2016 | arXiv:1506.02142 — dropout at inference for uncertainty |

---

### Reinforcement Learning for IR Derivatives

| Paper | Authors | Year | Key Idea |
|---|---|---|---|
| *Deep Reinforcement Learning for Option Pricing* | Cao et al | 2021 | arXiv:2101.02079 — RL for American options |
| *Hedging with Neural Networks* | Ruf, Wang | 2022 | *Quantitative Finance* — when do NNs beat Black-Scholes for hedging? |

---

## Reading Order

```
If you want to implement improvements immediately:
  1. Huge & Savine (2020) — differential ML   [2005.02347]
  2. Raissi et al (2017) — PINNs              [1711.10561]
  3. McGhee (2018) — SABR NN surrogate        [SSRN 3288882]

If you want to understand the theoretical foundations:
  1. Hagan et al (2002) — SABR               [Wilmott]
  2. Antonov et al (2011) — ZABR             [SSRN]
  3. Horvath et al (2021) — DL Volatility    [1901.09647]

If you want to extend to hedging:
  1. Buehler et al (2018) — Deep Hedging     [1802.03042]
  2. Ruf & Wang (2022) — Hedging with NNs    [Quant Finance]
  3. Buehler et al (2021) — Risk-neutral dyn [2103.11948]
```

---

*Generated: 2026-04-17 | Model: SwaptionNetV2 | Pipeline: research/swaption_nn_v2.py*
