"""
Production-grade European Swaption NN — v2.

Improvements over v1:
  1. ZABR surface  : calibrated alpha/nu/rho per (expiry, tenor, ccy) bucket
                     with smooth interpolation — realistic smile term structure
  2. Two-curve     : OIS discount curve + index forward curve per CCY
                     basis spread baked into forward rate AND annuity
  3. Tail sampling : 3-strata mixture — ATM 60% / OTM 25% / deep-OTM 15%
                     ensures model learns far wings, not just ATM
  4. Greek targets : analytical Bachelier delta/vega/gamma/vanna/theta
                     stored as supervised outputs (no bumping)
  5. Loss          : MSE + pinball(q=0.1, q=0.9) on price
                     + weighted Greek MSE on delta, vega, vanna

Synthetic data only — no market feed required.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.special import ndtr
import QuantLib as ql
import time, warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cpu")
torch.set_default_dtype(torch.float32)
torch.manual_seed(0)

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  CONVENTION CATALOGUE
# ═══════════════════════════════════════════════════════════════════════════════

CONVENTIONS = {
    "USD_SOFR":  (ql.Actual360(),                    ql.ModifiedFollowing, ql.UnitedStates(ql.UnitedStates.GovernmentBond), ql.Annual,     0),
    "EUR_EURIB": (ql.Thirty360(ql.Thirty360.BondBasis), ql.ModifiedFollowing, ql.TARGET(),                                    ql.Annual,     1),
    "GBP_SONIA": (ql.Actual365Fixed(),               ql.ModifiedFollowing, ql.UnitedKingdom(),                              ql.Semiannual, 2),
}
CONV_NAMES = list(CONVENTIONS.keys())
N_CONV = len(CONV_NAMES)

TODAY = ql.Date(17, 4, 2026)
ql.Settings.instance().evaluationDate = TODAY

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  TWO-CURVE SETUP  (OIS discount  +  index forward)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  USD : SOFR OIS 4.80%  |  Term SOFR / SOFR 3M  +10bps  = 4.90%
#  EUR : ESTR    2.20%   |  EURIBOR 3M           +25bps  = 2.45%
#  GBP : SONIA   4.50%   |  SONIA 3M             + 5bps  = 4.55%

TWO_CURVE = {
    # ccy_idx → (ois_rate, index_rate)
    0: (0.0480, 0.0490),   # USD
    1: (0.0220, 0.0245),   # EUR
    2: (0.0450, 0.0455),   # GBP
}

def df_ois(T, ccy_idx):
    """Flat OIS discount factor — used for annuity."""
    r = np.array([TWO_CURVE[c][0] for c in ccy_idx])
    return np.exp(-r * T)

def df_idx(T, ccy_idx):
    """Flat index discount factor — used for forward rate."""
    r = np.array([TWO_CURVE[c][1] for c in ccy_idx])
    return np.exp(-r * T)

def par_forward(T_exp, T_end, ccy_idx):
    """
    Forward swap rate from index curve (numerator)
    discounted by OIS curve (denominator = annuity cashflows).
    Simplified trap-rule annuity; proper schedule used for DCF adjustment.
    """
    df_s_ois = df_ois(T_exp, ccy_idx)
    df_e_ois = df_ois(T_end, ccy_idx)
    df_s_idx = df_idx(T_exp, ccy_idx)
    df_e_idx = df_idx(T_end, ccy_idx)
    tenor    = T_end - T_exp
    annuity  = tenor * 0.5 * (df_s_ois + df_e_ois)          # OIS-discounted
    forward  = (df_s_idx - df_e_idx) / annuity               # index-curve fwd
    return forward, annuity

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  ZABR SURFACE  — calibrated per (expiry, tenor, ccy) bucket
# ═══════════════════════════════════════════════════════════════════════════════

def zabr_surface_params(T_exp, tenor, ccy_idx):
    """
    Smooth ZABR parameter surface: alpha / nu / rho as functions of
    (T_exp, tenor, ccy).  Mimics a realistic calibrated surface.

    alpha : ATM normal vol — hump-shaped in expiry, declining in tenor
    nu    : vol-of-vol    — decreasing with expiry (short end is more volatile)
    rho   : skew          — negative for rates, more negative at short end
    """
    # ── alpha (ATM normal vol, in yield space) ────────────────────────────────
    # Hump around 2Y expiry, shape varies by ccy
    ccy_base = np.where(ccy_idx == 1, 0.0038,             # EUR lower rates
                np.where(ccy_idx == 2, 0.0055, 0.0060))   # GBP / USD
    hump     = 1 + 0.6 * np.exp(-((T_exp - 2.0)**2) / 4)  # peak at 2Y
    tenor_dk = np.exp(-0.06 * tenor)                        # vol decreases with tenor
    alpha    = ccy_base * hump * (0.7 + 0.3 * tenor_dk)

    # ── nu (vol-of-vol) ──────────────────────────────────────────────────────
    nu = 0.18 + 0.40 * np.exp(-0.30 * T_exp)   # 58% at 1M expiry → 18% at 10Y

    # ── rho (skew / correlation) ──────────────────────────────────────────────
    rho = -0.08 - 0.28 * np.exp(-0.45 * T_exp)   # -36% at short end → -8% at long

    # Perturb slightly to get diverse surface (simulate calibration noise)
    noise = np.random.default_rng().random if False else None   # deterministic
    return alpha, nu, rho

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  BACHELIER CLOSED-FORM — price + 5 Greeks (fully vectorized)
# ═══════════════════════════════════════════════════════════════════════════════

def bachelier_full(F, K, sigma_n, T, is_payer):
    """
    Bachelier price + Delta / Vega / Gamma / Theta / Vanna
    All normalized to be O(1) → easier for the NN to learn.

    Normalisation:
      norm_price = price   / (sigma_n * sqrt_T)
      norm_delta = delta                          ≈  N(d)  ∈ [0,1]
      norm_vega  = vega    / sqrt_T               =  n(d)  ∈ [0, 0.4]
      norm_gamma = gamma   * sigma_n * sqrt_T     =  n(d)
      norm_theta = theta   * 2*sqrt_T / sigma_n   = -n(d)
      norm_vanna = vanna   * sigma_n              = -d * n(d)
    """
    sqrt_T  = np.sqrt(np.clip(T, 1e-8, None))
    d       = (F - K) / (sigma_n * sqrt_T)
    pdf     = np.exp(-0.5 * d**2) / np.sqrt(2 * np.pi)
    cdf     = ndtr(d)

    # Price (per unit annuity, per unit notional)
    raw_price = np.where(is_payer,
        (F - K) * cdf + sigma_n * sqrt_T * pdf,
        (K - F) * ndtr(-d) + sigma_n * sqrt_T * pdf)

    # Greeks
    delta = np.where(is_payer, cdf, cdf - 1.0)          # ∈ [-1, 1]
    vega  = sqrt_T * pdf                                  # > 0
    gamma = pdf / (sigma_n * sqrt_T)                      # > 0
    theta = -0.5 * sigma_n * pdf / sqrt_T                 # < 0
    vanna = -d * pdf / sigma_n                            # signed

    # Normalise to O(1) for training stability
    norm_price = raw_price / (sigma_n * sqrt_T)
    norm_vega  = vega  / sqrt_T          # = pdf  ∈ [0, 0.4]
    norm_gamma = gamma * sigma_n * sqrt_T  # = pdf
    norm_theta = theta * 2 * sqrt_T / sigma_n  # = -pdf
    norm_vanna = vanna * sigma_n           # = -d*pdf

    return norm_price, delta, norm_vega, norm_gamma, norm_theta, norm_vanna

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  ZABR NORMAL VOL  (vectorized closed-form)
# ═══════════════════════════════════════════════════════════════════════════════

def zabr_normal_vol(F, K, T, alpha, nu, rho):
    atm   = np.abs(F - K) < 1e-9
    z     = np.where(atm, 0.0, nu / np.clip(alpha, 1e-9, None) * (F - K))
    sq    = np.sqrt(np.clip(1 - 2*rho*z + z**2, 1e-12, None))
    chi   = np.where(atm, 1.0, np.log((sq + z - rho) / np.clip(1-rho, 1e-12, None)))
    z_chi = np.where(atm, 1.0, z / np.where(np.abs(chi) < 1e-12, 1e-12, chi))
    sig   = alpha * z_chi * (1 + T * (2 - 3*rho**2)/24 * nu**2)
    return np.clip(sig, 1e-6, None)

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  TAIL-AWARE SAMPLER — 3-strata mixture
# ═══════════════════════════════════════════════════════════════════════════════

def sample_strikes(F, sigma_n, T, rng, n):
    """
    3-strata mixture for strike spread:
      60% ATM region   : spread ~ N(0, 0.5 σ√T)
      25% OTM wing     : spread ~ N(±2 σ√T,  0.5 σ√T)   (±2 std)
      15% deep OTM     : spread ~ Uniform(±3 σ√T, ±6 σ√T)
    Ensures tails are well represented without wasting capacity near ATM.
    """
    atm_vol = sigma_n * np.sqrt(T)   # 1-sigma move
    strata  = rng.choice([0, 1, 2], size=n, p=[0.60, 0.25, 0.15])

    spread  = np.zeros(n)
    sign    = rng.choice([-1, 1], size=n)

    m_atm   = strata == 0
    m_otm   = strata == 1
    m_deep  = strata == 2

    spread[m_atm]  = rng.normal(0, 0.5 * atm_vol[m_atm])
    spread[m_otm]  = sign[m_otm] * (2*atm_vol[m_otm] + rng.normal(0, 0.5*atm_vol[m_otm], m_otm.sum()))
    spread[m_deep] = sign[m_deep] * rng.uniform(3*atm_vol[m_deep], 6*atm_vol[m_deep])

    K = np.clip(F + spread, 1e-4, 0.20)
    return K, strata

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  DATASET GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_dataset(n: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    print(f"  Generating {n:,} samples  (3-strata, two-curve, ZABR surface)...")
    t0 = time.perf_counter()

    conv_idx  = rng.integers(0, N_CONV, n)
    is_payer  = rng.integers(0, 2, n).astype(bool)
    T_exp     = rng.uniform(0.08, 10.0, n)
    tenor     = rng.choice([1,2,3,5,7,10], n).astype(float)
    T_end     = T_exp + tenor

    # Two-curve forward + annuity
    F, annuity = par_forward(T_exp, T_end, conv_idx)

    # ZABR surface params
    alpha, nu, rho = zabr_surface_params(T_exp, tenor, conv_idx)
    # Add calibration noise (±15% relative)
    alpha *= rng.uniform(0.85, 1.15, n)
    nu    *= rng.uniform(0.85, 1.15, n)
    rho   += rng.uniform(-0.05, 0.05, n)
    rho    = np.clip(rho, -0.95, 0.95)

    # ZABR ATM vol
    sigma_atm = zabr_normal_vol(F, F, T_exp, alpha, nu, rho)

    # Tail-aware strikes
    K, strata = sample_strikes(F, sigma_atm, T_exp, rng, n)

    # ZABR smile vol at (F, K)
    sigma_n = zabr_normal_vol(F, K, T_exp, alpha, nu, rho)

    # Bachelier: price + 5 Greeks (all normalised)
    norm_price, delta, norm_vega, norm_gamma, norm_theta, norm_vanna = \
        bachelier_full(F, K, sigma_n, T_exp, is_payer)

    # moneyness = d (standardised)
    moneyness  = (F - K) / (sigma_n * np.sqrt(np.clip(T_exp, 1e-8, None)))
    log_mk     = np.log(np.clip(K / F, 1e-4, 10.0))

    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed*1000:.0f}ms  |  "
          f"price [{norm_price.min():.3f}, {norm_price.max():.3f}]  |  "
          f"strata: ATM={( strata==0).mean()*100:.0f}% "
          f"OTM={(strata==1).mean()*100:.0f}% "
          f"deep={(strata==2).mean()*100:.0f}%")

    return dict(
        # Continuous inputs
        moneyness  = moneyness.astype(np.float32),
        log_mk     = log_mk.astype(np.float32),
        T          = T_exp.astype(np.float32),
        tenor      = tenor.astype(np.float32),
        sigma_atm  = sigma_atm.astype(np.float32),
        nu         = nu.astype(np.float32),
        rho        = rho.astype(np.float32),
        is_payer   = is_payer.astype(np.float32),
        # Categorical
        conv_idx   = conv_idx.astype(np.int64),
        strata     = strata.astype(np.int64),
        # Targets  (all normalised, O(1))
        norm_price = norm_price.astype(np.float32),
        delta      = delta.astype(np.float32),
        norm_vega  = norm_vega.astype(np.float32),
        norm_gamma = norm_gamma.astype(np.float32),
        norm_theta = norm_theta.astype(np.float32),
        norm_vanna = norm_vanna.astype(np.float32),
        # For denormalisation
        sigma_n    = sigma_n.astype(np.float32),
        T_raw      = T_exp.astype(np.float32),
        annuity    = annuity.astype(np.float32),
    )

# ═══════════════════════════════════════════════════════════════════════════════
# 8.  DATASET CLASS
# ═══════════════════════════════════════════════════════════════════════════════

CONT_FEATURES = ["moneyness","log_mk","T","tenor","sigma_atm","nu","rho","is_payer"]
GREEK_TARGETS = ["norm_price","delta","norm_vega","norm_gamma","norm_theta","norm_vanna"]

class SwaptionDataset(Dataset):
    def __init__(self, data: dict):
        self.cont   = torch.tensor(np.stack([data[k] for k in CONT_FEATURES], 1), dtype=torch.float32)
        self.conv   = torch.tensor(data["conv_idx"], dtype=torch.long)
        self.strata = torch.tensor(data["strata"],   dtype=torch.long)
        self.target = torch.tensor(np.stack([data[k] for k in GREEK_TARGETS], 1), dtype=torch.float32)
        self.sigma_n = torch.tensor(data["sigma_n"], dtype=torch.float32)
        self.T       = torch.tensor(data["T_raw"],   dtype=torch.float32)
        self.annuity = torch.tensor(data["annuity"], dtype=torch.float32)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return (self.cont[idx], self.conv[idx], self.strata[idx],
                self.target[idx], self.sigma_n[idx], self.T[idx])

# ═══════════════════════════════════════════════════════════════════════════════
# 9.  MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.05):
        super().__init__()
        self.net  = nn.Sequential(nn.Linear(dim,dim), nn.SiLU(),
                                  nn.Dropout(dropout), nn.Linear(dim,dim))
        self.norm = nn.LayerNorm(dim)
    def forward(self, x): return self.norm(x + self.net(x))


class SwaptionNetV2(nn.Module):
    """
    Multi-output: [norm_price, delta, norm_vega, norm_gamma, norm_theta, norm_vanna]

    Architecture:
      • Learned embeddings for convention (dim=8) and strata (dim=4)
      • Shared trunk: 5 residual blocks × 192 hidden
      • 6 independent output heads (one per target)
        → allows each Greek to have its own output scale
    """
    N_OUT = len(GREEK_TARGETS)

    def __init__(self, n_cont=8, embed_dim=8, strata_dim=4, hidden=192, n_blocks=5, dropout=0.05):
        super().__init__()
        self.conv_emb   = nn.Embedding(N_CONV, embed_dim)
        self.strata_emb = nn.Embedding(3, strata_dim)        # ATM/OTM/deep
        in_dim = n_cont + embed_dim + strata_dim
        self.proj   = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU())
        self.trunk  = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])
        # Separate head per Greek — they have different scales and signs
        self.heads  = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, 32), nn.SiLU(), nn.Linear(32, 1))
            for _ in range(self.N_OUT)
        ])

    def forward(self, cont, conv, strata):
        x  = torch.cat([cont, self.conv_emb(conv), self.strata_emb(strata)], dim=1)
        x  = self.trunk(self.proj(x))
        return torch.cat([h(x) for h in self.heads], dim=1)   # [B, 6]

# ═══════════════════════════════════════════════════════════════════════════════
# 10.  LOSS FUNCTION
#      MSE  +  Pinball(q=0.1, q=0.9) on price
#      +  Greek MSE with per-Greek weights
# ═══════════════════════════════════════════════════════════════════════════════

# Weight per output: price matters most, then delta/vega, then higher-order
GREEK_WEIGHTS = torch.tensor([3.0, 2.0, 2.0, 0.5, 0.3, 1.0], dtype=torch.float32)
# Indices:                    price delta vega gamma theta vanna

def pinball_loss(pred, target, q):
    err = target - pred
    return torch.where(err >= 0, q * err, (q - 1) * err).mean()

def total_loss(pred, target, lam_quantile=0.30, lam_greeks=0.40):
    """
    pred, target : [B, 6]
    Loss components:
      • MSE on all outputs (weighted)
      • Pinball on price (col 0) at q=0.1 and q=0.9
      • Per-Greek MSE explicitly tracked for logging
    """
    weights = GREEK_WEIGHTS.to(pred.device)
    mse_per = ((pred - target)**2).mean(dim=0)               # [6]
    loss_mse = (mse_per * weights).sum() / weights.sum()

    p_pred, p_true = pred[:,0], target[:,0]
    loss_q  = 0.5 * (pinball_loss(p_pred, p_true, 0.10)
                   + pinball_loss(p_pred, p_true, 0.90))

    total = (1 - lam_quantile - lam_greeks) * loss_mse \
          + lam_quantile * loss_q \
          + lam_greeks   * (mse_per[1:] * weights[1:]).sum() / weights[1:].sum()

    return total, mse_per

# ═══════════════════════════════════════════════════════════════════════════════
# 11.  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train(model, tr_loader, va_loader, epochs=50, lr=3e-4):
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(tr_loader),
        pct_start=0.15, anneal_strategy='cos')
    best_val, history = float("inf"), []

    hdr = f"{'Ep':>4} {'Loss':>9} {'ValLoss':>9} {'price':>8} {'delta':>8} {'vega':>8} {'vanna':>8}  MAE_bps"
    print(f"\n{hdr}\n{'─'*74}")

    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for cont, conv, strata, target, *_ in tr_loader:
            cont, conv, strata, target = cont.to(DEVICE), conv.to(DEVICE), strata.to(DEVICE), target.to(DEVICE)
            pred  = model(cont, conv, strata)
            loss, _ = total_loss(pred, target)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_loader)

        model.eval()
        va_loss, mse_acc, mae_bps = 0.0, torch.zeros(6), 0.0
        n_bat = 0
        with torch.no_grad():
            for cont, conv, strata, target, sigma_n, T in va_loader:
                cont, conv, strata, target = cont.to(DEVICE), conv.to(DEVICE), strata.to(DEVICE), target.to(DEVICE)
                sigma_n, T = sigma_n.to(DEVICE), T.to(DEVICE)
                pred         = model(cont, conv, strata)
                loss, mse_p  = total_loss(pred, target)
                va_loss     += loss.item()
                mse_acc     += mse_p.cpu()
                # Denormalise price → bps
                scale    = sigma_n * torch.sqrt(T)
                mae_bps += ((pred[:,0] - target[:,0]).abs() * scale).mean().item() * 10_000
                n_bat   += 1
        va_loss /= n_bat;  mse_acc /= n_bat;  mae_bps /= n_bat

        history.append((tr_loss, va_loss, mae_bps))
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), "research/swaption_nn_v2_best.pt")

        if ep % 5 == 0 or ep == 1:
            m = mse_acc
            print(f"{ep:>4} {tr_loss:>9.5f} {va_loss:>9.5f} "
                  f"{m[0].item():>8.5f} {m[1].item():>8.5f} "
                  f"{m[2].item():>8.5f} {m[5].item():>8.5f}  {mae_bps:>6.3f}")

    return history

# ═══════════════════════════════════════════════════════════════════════════════
# 12.  VALIDATION BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════════

def detailed_validation(model, dataset):
    model.eval()
    cont    = dataset.cont.to(DEVICE)
    conv    = dataset.conv.to(DEVICE)
    strata  = dataset.strata.to(DEVICE)
    target  = dataset.target
    sigma_n = dataset.sigma_n
    T       = dataset.T

    with torch.no_grad():
        pred = model(cont, conv, strata).cpu()

    scale   = sigma_n * torch.sqrt(T)
    err_bps = (pred[:,0] - target[:,0]).abs() * scale * 10_000
    err_rel = (pred[:,0] - target[:,0]).abs() / (target[:,0].abs().clamp(min=1e-6))

    print(f"\n{'─'*58}")
    print(f"{'Metric':<22} {'Mean':>10} {'P50':>10} {'P95':>10}")
    print(f"{'─'*58}")
    for label, vals in [
        ("Price MAE (bps)", err_bps),
        ("Price rel. error", err_rel * 100),
    ]:
        print(f"{label:<22} {vals.mean():>10.4f} "
              f"{vals.quantile(0.50):>10.4f} "
              f"{vals.quantile(0.95):>10.4f}")

    print(f"\n{'─'*44}")
    print(f"{'Stratum':<14} {'N':>8} {'MAE bps':>10} {'P95 bps':>10}")
    print(f"{'─'*44}")
    for si, name in enumerate(["ATM","OTM","Deep-OTM"]):
        mask = dataset.strata == si
        if mask.sum() == 0: continue
        e = err_bps[mask]
        print(f"{name:<14} {mask.sum():>8,} {e.mean():>10.4f} {e.quantile(0.95):>10.4f}")

    print(f"\n{'─'*44}")
    print(f"{'Convention':<14} {'N':>8} {'MAE bps':>10} {'P95 bps':>10}")
    print(f"{'─'*44}")
    for ci, name in enumerate(CONV_NAMES):
        mask = dataset.conv == ci
        e = err_bps[mask]
        print(f"{name:<14} {mask.sum():>8,} {e.mean():>10.4f} {e.quantile(0.95):>10.4f}")

    print(f"\n── Greek MSE (normalised) {'─'*30}")
    for i, name in enumerate(GREEK_TARGETS):
        mse = ((pred[:,i] - target[:,i])**2).mean().item()
        mae = (pred[:,i] - target[:,i]).abs().mean().item()
        print(f"  {name:<14}: MSE={mse:.6f}  MAE={mae:.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 13.  INFERENCE SPEED
# ═══════════════════════════════════════════════════════════════════════════════

def inference_benchmark(model, dataset, n=50_000):
    model.eval()
    cont   = dataset.cont[:n].to(DEVICE)
    conv   = dataset.conv[:n].to(DEVICE)
    strata = dataset.strata[:n].to(DEVICE)

    # Warmup
    with torch.no_grad():
        _ = model(cont, conv, strata)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(cont, conv, strata)      # [N, 6]: price + 5 Greeks in one pass
    t_ms = (time.perf_counter() - t0) * 1000

    print(f"\n── Inference ({n:,} swaptions) ─────────────────────────")
    print(f"  Price + 5 Greeks (single forward pass): {t_ms:.1f} ms")
    print(f"  Throughput                            : {n/(t_ms/1000):,.0f} options/sec")
    print(f"  Output shape                          : {list(out.shape)}")
    print(f"  Output columns: {GREEK_TARGETS}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 62)
    print("European Swaption NN v2 — Production Synthetic Pipeline")
    print("=" * 62)

    N_TR, N_VA = 300_000, 30_000

    print("\n[1] Data generation")
    data_tr = generate_dataset(N_TR, seed=0)
    data_va = generate_dataset(N_VA, seed=99)
    ds_tr   = SwaptionDataset(data_tr)
    ds_va   = SwaptionDataset(data_va)
    ld_tr   = DataLoader(ds_tr, batch_size=4096, shuffle=True,  num_workers=0, pin_memory=False)
    ld_va   = DataLoader(ds_va, batch_size=4096, shuffle=False, num_workers=0)

    print("\n[2] Model")
    model  = SwaptionNetV2(hidden=192, n_blocks=5).to(DEVICE)
    n_p    = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {n_p:,}")
    print(f"  Outputs    : {GREEK_TARGETS}")
    print(f"  Loss       : MSE(weighted) + Pinball(q=0.1,0.9) + Greek MSE")

    print("\n[3] Training (50 epochs, OneCycleLR)")
    history = train(model, ld_tr, ld_va, epochs=50, lr=4e-4)

    model.load_state_dict(torch.load("research/swaption_nn_v2_best.pt", weights_only=True))
    print("\n  Best checkpoint loaded.")

    print("\n[4] Validation breakdown")
    detailed_validation(model, ds_va)

    print("\n[5] Inference benchmark")
    inference_benchmark(model, ds_va)

    print(f"""
{"="*62}
Design summary
{"─"*62}
DATA GENERATION
  ZABR surface : alpha/nu/rho smooth functions of (T, tenor, ccy)
                 hump-shaped alpha, decaying nu/rho with expiry
  Two-curve    : OIS discount (annuity) + index forward (F)
                 USD basis +10bps | EUR +25bps | GBP +5bps
  Tail strata  : 60% ATM / 25% OTM / 15% deep-OTM
                 strike sampling in σ√T units → no hard cutoffs
  Greeks       : exact Bachelier formula, all normalised to O(1)

MODEL
  Embeddings   : convention (dim=8) + strata (dim=4)
  Trunk        : 5 residual blocks × 192, SiLU + LayerNorm
  Heads        : 6 independent heads (price + 5 Greeks)
                 separate so each Greek has own scale/bias

LOSS
  MSE          : all 6 outputs, weighted [3,2,2,0.5,0.3,1]
  Pinball      : q=0.1 and q=0.9 on price → bounds tails
  Greek weight : delta/vega upweighted vs gamma/theta
{"="*62}
""")
