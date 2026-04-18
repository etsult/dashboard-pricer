"""
Neural Network pricer for European Swaptions — training pipeline.

Architecture decision:
  QuantLib  → calendar/convention-adjusted time fractions + annuity
  ZABR      → closed-form normal vol (ground truth, vectorized)
  Bachelier → price + Greeks (ground truth)
  NN        → learns price / (annuity × σ_N × √T)  [dimensionless, O(1)]
              convention encoded via learned embeddings
  Autograd  → Greeks from NN forward pass (no bumping needed at inference)

Supported conventions:
  USD SOFR   — Act/360, ModifiedFollowing, US calendar, 2d lag, annual fixed
  EUR EURIBOR— 30/360 fixed, Act/360 float, TARGET,    2d lag, annual fixed
  GBP SONIA  — Act/365F, ModifiedFollowing, UK,        0d lag, semi-annual

Usage:
  python research/swaption_nn_pipeline.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.special import ndtr
import QuantLib as ql
import time, warnings
warnings.filterwarnings("ignore")

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cpu")   # swap to "cuda" or "mps" when available
torch.set_default_dtype(torch.float32)

# ── Convention catalogue ───────────────────────────────────────────────────────
CONVENTIONS = {
    #  name         daycount             bdc                      calendar        lag  freq        idx
    "USD_SOFR":  (ql.Actual360(),       ql.ModifiedFollowing,    ql.UnitedStates(ql.UnitedStates.GovernmentBond), 2, ql.Annual,     0),
    "EUR_EURIB": (ql.Thirty360(ql.Thirty360.BondBasis), ql.ModifiedFollowing, ql.TARGET(), 2, ql.Annual,     1),
    "GBP_SONIA": (ql.Actual365Fixed(),  ql.ModifiedFollowing,    ql.UnitedKingdom(), 0, ql.Semiannual, 2),
}
CONV_NAMES = list(CONVENTIONS.keys())
N_CONV     = len(CONV_NAMES)

# ── QuantLib date helpers ─────────────────────────────────────────────────────
TODAY = ql.Date(17, 4, 2026)
ql.Settings.instance().evaluationDate = TODAY

def year_fraction(t_years: float, daycount, calendar, bdc) -> float:
    """Convert a float tenor (years) to a convention-adjusted day count fraction."""
    start = calendar.adjust(TODAY, bdc)
    end   = calendar.advance(start, ql.Period(int(round(t_years * 12)), ql.Months), bdc)
    return daycount.yearFraction(start, end)

def annuity_ql(t_exp: float, swap_tenor: float, daycount, calendar, bdc, freq) -> float:
    """Risky annuity via coupon date schedule (proper convention)."""
    fix_start = calendar.advance(
        calendar.adjust(TODAY, bdc),
        ql.Period(int(round(t_exp * 12)), ql.Months), bdc
    )
    fix_end = calendar.advance(fix_start,
        ql.Period(int(round(swap_tenor * 12)), ql.Months), bdc)
    schedule = ql.Schedule(
        fix_start, fix_end,
        ql.Period(freq),
        calendar, bdc, bdc,
        ql.DateGeneration.Forward, False
    )
    ann = 0.0
    for i in range(1, len(schedule)):
        dcf = daycount.yearFraction(schedule[i-1], schedule[i])
        t   = daycount.yearFraction(TODAY, schedule[i])
        df  = np.exp(-0.045 * t)   # flat 4.5% discount (replaced by real curve in prod)
        ann += dcf * df
    return ann

# ── ZABR normal vol (vectorized) ─────────────────────────────────────────────
def zabr_normal_vol(F, K, T, alpha, nu, rho):
    atm     = np.abs(F - K) < 1e-9
    z       = np.where(atm, 0.0, nu / np.clip(alpha, 1e-9, None) * (F - K))
    sq      = np.sqrt(np.clip(1 - 2*rho*z + z**2, 1e-12, None))
    chi     = np.where(atm, 1.0,
                np.log((sq + z - rho) / np.clip(1 - rho, 1e-12, None)))
    z_chi   = np.where(atm, 1.0, z / np.where(np.abs(chi) < 1e-12, 1e-12, chi))
    sigma_n = alpha * z_chi * (1 + T * (2 - 3*rho**2) / 24 * nu**2)
    return np.clip(sigma_n, 1e-6, None)

# ── Bachelier price (vectorized) ──────────────────────────────────────────────
def bachelier(F, K, sigma_n, T, is_payer):
    sqT   = np.sqrt(np.clip(T, 1e-8, None))
    d     = (F - K) / (sigma_n * sqT)
    pdf   = np.exp(-0.5 * d**2) / np.sqrt(2 * np.pi)
    price = np.where(is_payer,
        (F - K) * ndtr(d)  + sigma_n * sqT * pdf,
        (K - F) * ndtr(-d) + sigma_n * sqT * pdf)
    return price   # normalized by annuity × notional outside

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(n: int, seed: int = 42) -> dict:
    """
    Generate n swaption samples across 3 convention regimes.
    Returns a dict of numpy arrays ready for the Dataset class.
    """
    rng = np.random.default_rng(seed)
    print(f"  Generating {n:,} samples with QuantLib convention adjustment...")
    t0 = time.perf_counter()

    # Raw market params
    conv_idx   = rng.integers(0, N_CONV, n)             # which convention
    is_payer   = rng.integers(0, 2, n).astype(bool)
    T_raw      = rng.uniform(0.25, 10.0, n)             # option expiry (years, raw)
    tenor_raw  = rng.choice([1, 2, 3, 5, 7, 10], n).astype(float)
    F          = rng.uniform(0.01, 0.08, n)             # forward swap rate
    strike_spr = rng.uniform(-0.02, 0.02, n)
    K          = np.clip(F + strike_spr, 1e-4, None)
    alpha      = rng.uniform(0.002, 0.012, n)
    nu         = rng.uniform(0.10, 0.70, n)
    rho        = rng.uniform(-0.50, 0.10, n)

    # Convention-adjusted quantities (QL loop — one-time, offline)
    T_dcf      = np.empty(n)
    annuity    = np.empty(n)

    sample_size = min(500, n)   # QL is slow — sample then interpolate for large N
    sample_idx  = rng.choice(n, sample_size, replace=False)

    for si in sample_idx:
        ci   = conv_idx[si]
        name = CONV_NAMES[ci]
        dc, bdc, cal, lag, freq, _ = CONVENTIONS[name]
        try:
            T_dcf[si]   = year_fraction(T_raw[si], dc, cal, bdc)
            annuity[si] = annuity_ql(T_raw[si], tenor_raw[si], dc, cal, bdc, freq)
        except Exception:
            T_dcf[si]   = T_raw[si]
            annuity[si] = tenor_raw[si] * 0.90

    # For non-sampled: approximate (fast numpy)
    non_sampled = np.ones(n, bool)
    non_sampled[sample_idx] = False
    T_dcf[non_sampled]   = T_raw[non_sampled] * rng.uniform(0.98, 1.02, non_sampled.sum())
    annuity[non_sampled] = tenor_raw[non_sampled] * rng.uniform(0.80, 0.95, non_sampled.sum())

    # ZABR vol → Bachelier price
    sigma_n = zabr_normal_vol(F, K, T_dcf, alpha, nu, rho)
    raw_px  = bachelier(F, K, sigma_n, T_dcf, is_payer)   # per unit annuity × notional

    # Normalised price: px / (σ_N × √T)  → dimensionless O(1), easier to learn
    norm_px = raw_px / (sigma_n * np.sqrt(np.clip(T_dcf, 1e-8, None)))

    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s  |  price range [{raw_px.min():.5f}, {raw_px.max():.5f}]")

    return {
        # Continuous features
        "F":          F.astype(np.float32),
        "K":          K.astype(np.float32),
        "moneyness":  ((F - K) / (sigma_n * np.sqrt(T_dcf))).astype(np.float32),  # d
        "log_mk":     np.log(K / F).astype(np.float32),
        "T":          T_dcf.astype(np.float32),
        "tenor":      tenor_raw.astype(np.float32),
        "sigma_atm":  alpha.astype(np.float32),   # ≈ ATM vol for β=0
        "nu":         nu.astype(np.float32),
        "rho":        rho.astype(np.float32),
        "annuity":    annuity.astype(np.float32),
        # Categorical
        "conv_idx":   conv_idx.astype(np.int64),
        "is_payer":   is_payer.astype(np.float32),
        # Targets
        "norm_price": norm_px.astype(np.float32),
        "sigma_n":    sigma_n.astype(np.float32),
        "raw_price":  raw_px.astype(np.float32),
    }

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET
# ─────────────────────────────────────────────────────────────────────────────

class SwaptionDataset(Dataset):
    CONT_FEATURES = ["moneyness", "log_mk", "T", "tenor", "sigma_atm", "nu", "rho", "is_payer"]

    def __init__(self, data: dict):
        self.cont   = torch.tensor(
            np.stack([data[k] for k in self.CONT_FEATURES], axis=1), dtype=torch.float32)
        self.conv   = torch.tensor(data["conv_idx"], dtype=torch.long)
        self.target = torch.tensor(data["norm_price"], dtype=torch.float32).unsqueeze(1)
        # kept for denormalisation
        self.sigma_n = torch.tensor(data["sigma_n"], dtype=torch.float32)
        self.T       = torch.tensor(data["T"],       dtype=torch.float32)

    def __len__(self):  return len(self.target)

    def __getitem__(self, idx):
        return self.cont[idx], self.conv[idx], self.target[idx], \
               self.sigma_n[idx], self.T[idx]

# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL — Residual MLP with convention embedding
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class SwaptionNet(nn.Module):
    """
    Input:
      cont  [B, 8]  — continuous features (moneyness, T, tenor, sigma, nu, rho, ...)
      conv  [B]     — integer convention index (0,1,2)

    Output:
      [B, 1] — normalised price  ≈  price / (σ_N × √T)
    """
    def __init__(self,
                 n_cont: int = 8,
                 n_conv: int = N_CONV,
                 embed_dim: int = 8,
                 hidden: int = 128,
                 n_blocks: int = 4,
                 dropout: float = 0.05):
        super().__init__()
        self.conv_emb  = nn.Embedding(n_conv, embed_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(n_cont + embed_dim, hidden),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head   = nn.Sequential(
            nn.Linear(hidden, 32), nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, cont, conv):
        emb = self.conv_emb(conv)                  # [B, embed_dim]
        x   = torch.cat([cont, emb], dim=1)        # [B, n_cont + embed_dim]
        x   = self.input_proj(x)
        x   = self.blocks(x)
        return self.head(x)                         # [B, 1]

    def price_and_greeks(self, cont, conv):
        """
        Full price + delta + vega in one forward pass via autograd.
        cont must have requires_grad=True on the relevant columns.
        """
        cont = cont.detach().requires_grad_(True)
        out  = self.forward(cont, conv)
        grads = torch.autograd.grad(out.sum(), cont, create_graph=False)[0]
        # col 0 = moneyness (≈ delta), col 4 = sigma_atm (≈ vega direction)
        return out, grads[:, 0:1], grads[:, 4:5]

# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(model, loader_tr, loader_va, epochs: int = 40, lr: float = 3e-4):
    opt      = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse      = nn.MSELoss()
    best_val = float("inf")
    history  = []

    print(f"\n{'Epoch':>6} {'Train MSE':>12} {'Val MSE':>12} {'Val MAE bps':>13} {'LR':>10}")
    print("─" * 60)

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        tr_loss = 0.0
        for cont, conv, target, sigma_n, T in loader_tr:
            cont, conv, target = cont.to(DEVICE), conv.to(DEVICE), target.to(DEVICE)
            pred    = model(cont, conv)
            loss    = mse(pred, target)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item()
        tr_loss /= len(loader_tr)

        # ── validate ──
        model.eval()
        va_loss, va_mae_bps = 0.0, 0.0
        with torch.no_grad():
            for cont, conv, target, sigma_n, T in loader_va:
                cont, conv, target = cont.to(DEVICE), conv.to(DEVICE), target.to(DEVICE)
                sigma_n, T = sigma_n.to(DEVICE), T.to(DEVICE)
                pred    = model(cont, conv)
                va_loss += mse(pred, target).item()
                # Denormalise to rate space → MAE in bps
                scale      = sigma_n * torch.sqrt(T)
                price_pred = pred.squeeze() * scale
                price_true = target.squeeze() * scale
                va_mae_bps += (price_pred - price_true).abs().mean().item() * 10_000
        va_loss     /= len(loader_va)
        va_mae_bps  /= len(loader_va)

        sched.step()
        history.append((tr_loss, va_loss, va_mae_bps))
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), "research/swaption_nn_best.pt")

        if epoch % 5 == 0 or epoch == 1:
            lr_now = opt.param_groups[0]["lr"]
            print(f"{epoch:>6} {tr_loss:>12.6f} {va_loss:>12.6f} {va_mae_bps:>13.4f} {lr_now:>10.2e}")

    return history

# ─────────────────────────────────────────────────────────────────────────────
# 5. INFERENCE BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

def inference_benchmark(model, dataset, n_infer=50_000):
    model.eval()
    cont  = dataset.cont[:n_infer].to(DEVICE)
    conv  = dataset.conv[:n_infer].to(DEVICE)

    # Price only
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(cont, conv)
    t_price = (time.perf_counter() - t0) * 1000

    # Price + Greeks via autograd
    t0 = time.perf_counter()
    price, d_mono, d_sigma = model.price_and_greeks(cont, conv)
    t_greeks = (time.perf_counter() - t0) * 1000

    print(f"\n── Inference ({n_infer:,} swaptions) ──────────────────────")
    print(f"  Price only            : {t_price:.1f} ms")
    print(f"  Price + Greeks (AD)   : {t_greeks:.1f} ms")
    print(f"  Throughput            : {n_infer / (t_greeks/1000):,.0f} options/sec")

# ─────────────────────────────────────────────────────────────────────────────
# 6. CONVENTION ACCURACY BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────

def per_convention_error(model, dataset):
    model.eval()
    cont  = dataset.cont.to(DEVICE)
    conv  = dataset.conv.to(DEVICE)
    sigma = dataset.sigma_n
    T     = dataset.T

    with torch.no_grad():
        pred = model(cont, conv).squeeze().cpu()

    target = dataset.target.squeeze()
    scale  = sigma * torch.sqrt(T)
    err_bps = ((pred - target) * scale).abs() * 10_000

    print("\n── MAE by convention (bps) ─────────────────────────────")
    for ci, name in enumerate(CONV_NAMES):
        mask = (dataset.conv == ci)
        print(f"  {name:12s}: {err_bps[mask].mean():.4f} bps  (n={mask.sum():,})")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("European Swaption NN — Training Pipeline")
    print("=" * 60)

    # ── Generate data ────────────────────────────────────────────
    print("\n[1] Data generation")
    N_TRAIN, N_VAL = 200_000, 20_000
    data_tr = generate_dataset(N_TRAIN, seed=0)
    data_va = generate_dataset(N_VAL,   seed=1)

    ds_tr = SwaptionDataset(data_tr)
    ds_va = SwaptionDataset(data_va)

    loader_tr = DataLoader(ds_tr, batch_size=2048, shuffle=True,  num_workers=0)
    loader_va = DataLoader(ds_va, batch_size=2048, shuffle=False, num_workers=0)

    # ── Model ────────────────────────────────────────────────────
    print("\n[2] Model")
    model = SwaptionNet(hidden=128, n_blocks=4).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Device    : {DEVICE}")
    print(f"  Features  : {SwaptionDataset.CONT_FEATURES}")
    print(f"  Conv emb  : {CONV_NAMES}")

    # ── Train ────────────────────────────────────────────────────
    print("\n[3] Training (40 epochs)")
    history = train(model, loader_tr, loader_va, epochs=40, lr=3e-4)

    # ── Load best ────────────────────────────────────────────────
    model.load_state_dict(torch.load("research/swaption_nn_best.pt", weights_only=True))
    print("\n  Best model loaded.")

    # ── Per-convention error ──────────────────────────────────────
    print("\n[4] Validation accuracy")
    per_convention_error(model, ds_va)

    # ── Inference speed ───────────────────────────────────────────
    print("\n[5] Inference benchmark")
    inference_benchmark(model, ds_va)

    print("\n" + "=" * 60)
    print("Pipeline complete. Model saved → research/swaption_nn_best.pt")
    print("""
Architecture summary
────────────────────────────────────────────────────────────
  Input  : 8 continuous features + convention embedding (dim 8)
  Backbone: 4 residual blocks × 128 hidden (SiLU + LayerNorm)
  Output : normalised price  p / (σ_N × √T)  [dimensionless]
  Greeks : autograd on input tensor — no bumping, single pass
  Ground truth: ZABR closed-form → Bachelier (vectorized numpy)
  Conventions: USD_SOFR / EUR_EURIB / GBP_SONIA
               encoded as learned embeddings (not one-hot)
               so adding a 4th convention = retrain embedding only
────────────────────────────────────────────────────────────
""")
