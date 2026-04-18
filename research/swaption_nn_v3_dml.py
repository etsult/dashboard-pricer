"""
Differential Machine Learning for European Swaption Pricing
Huge & Savine (2020) — arXiv:2005.02347

Core idea
─────────
Standard supervised learning: minimise  ||price_pred - price_true||²

Differential ML adds:   minimise  ||price_pred - price_true||²
                               + λ · Σᵢ wᵢ ||∂price_pred/∂xᵢ - ∂price_true/∂xᵢ||²

The second term supervises the NN's own Jacobian w.r.t. inputs against
analytical Greek labels.  Because d(norm_price)/d(moneyness) = delta exactly
(proved below), the model is forced to learn the correct shape of the pricing
function — not just point values.

Analytical correspondence (Bachelier, beta=0 ZABR)
───────────────────────────────────────────────────
  norm_price  = price / (σ_N √T)  =  d·N(d) + n(d)   [payer]
                                      [d = (F-K)/(σ_N √T)]

  ∂(norm_price)/∂d        = N(d) = delta               [exact]
  ∂(norm_price)/∂σ_atm   ≈ n(d)  = norm_vega           [approx via ZABR chain]
  ∂(norm_price)/∂T        = (norm_theta - norm_price)/(2T)

  Derivation for ∂/∂T:
    ∂/∂T [price/(σ_N √T)] = [∂price/∂T · σ_N √T - price · σ_N/(2√T)] / (σ_N √T)²
                           = theta_raw/(σ_N √T) - norm_price/(2T)
    theta_raw = -½ σ_N n(d)/√T  →  ∂(norm_price)/∂T = (norm_theta - norm_price)/(2T)

Expected result:  5-10× lower price MAE and Greek MAE vs plain MSE (v2),
                  at the same model size and same number of epochs.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

_src = open("research/swaption_nn_v2.py").read().split("if __name__")[0]
exec(_src, globals())   # imports v2 classes/functions; refactor to module when stable

DEVICE = torch.device("cpu")
torch.manual_seed(42)

# ── Differential input columns: positions in CONT_FEATURES ────────────────────
_DIFF_COLS = [CONT_FEATURES.index(f) for f in ("moneyness", "sigma_atm", "T")]

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  DIFFERENTIAL LABELS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_diff_labels(data: dict) -> np.ndarray:
    """
    Returns [N, 3] array of analytical Jacobian targets:
      col 0 — ∂(norm_price)/∂moneyness = delta        [exact]
      col 1 — ∂(norm_price)/∂sigma_atm ≈ norm_vega   [approx]
      col 2 — ∂(norm_price)/∂T = (θ_n - p_n) / 2T   [exact]
    """
    d_T = ((data["norm_theta"] - data["norm_price"])
           / (2 * np.clip(data["T_raw"], 1e-6, None)))
    return np.stack([data["delta"], data["norm_vega"], d_T], axis=1).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  DATASET  (inherits SwaptionDataset, adds diff_labels)
# ═══════════════════════════════════════════════════════════════════════════════

class DMLDataset(SwaptionDataset):
    def __init__(self, data: dict):
        super().__init__(data)
        self.diff_labels = torch.tensor(compute_diff_labels(data), dtype=torch.float32)

    def __getitem__(self, idx):
        cont, conv, strata, target, sigma_n, T = super().__getitem__(idx)
        return cont, conv, strata, target, self.diff_labels[idx], sigma_n, T


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  DIFFERENTIAL LOSS
# ═══════════════════════════════════════════════════════════════════════════════

def dml_loss(pred, target, cont, diff_labels, diff_scale,
             lambda_price=1.0, lambda_diff=5.0, training=True):
    """
    pred        : [B, 6]   model output
    cont        : [B, 8]   inputs WITH requires_grad=True
    diff_labels : [B, 3]   analytical Jacobian targets
    diff_scale  : [1, 3]   normalisation weights (price_std / diff_std per dim)
    training    : bool     use create_graph=True only during training
    """
    weights = GREEK_WEIGHTS.to(pred.device)
    mse_per = ((pred - target) ** 2).mean(dim=0)
    loss_price = (mse_per * weights).sum() / weights.sum()

    jac = torch.autograd.grad(
        pred[:, 0].sum(), cont,
        create_graph=training,   # second-order graph only needed for backward()
    )[0]   # [B, 8]

    jac_sup = torch.stack([jac[:, c] for c in _DIFF_COLS], dim=1)   # [B, 3]
    diff_err = ((jac_sup - diff_labels) * diff_scale.to(pred.device)) ** 2
    loss_diff = diff_err.mean()

    return lambda_price * loss_price + lambda_diff * loss_diff, mse_per, loss_diff.detach()


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  TRAINING LOOP — DML
# ═══════════════════════════════════════════════════════════════════════════════

def train_dml(model, tr_loader, va_loader, diff_scale,
              epochs=50, lr=4e-4, lambda_diff=5.0):
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(tr_loader),
        pct_start=0.15, anneal_strategy='cos')
    best_val, history = float("inf"), []

    hdr = (f"{'Ep':>4} {'Loss':>9} {'ValLoss':>9} "
           f"{'DiffLoss':>10} {'price':>8} {'delta':>8} {'vega':>8}  MAE_bps")
    print(f"\n{hdr}\n{'─'*78}")

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for cont, conv, strata, target, diff_lbl, *_ in tr_loader:
            cont     = cont.to(DEVICE).requires_grad_(True)
            conv     = conv.to(DEVICE)
            strata   = strata.to(DEVICE)
            target   = target.to(DEVICE)
            diff_lbl = diff_lbl.to(DEVICE)

            pred = model(cont, conv, strata)
            loss, _, _ = dml_loss(pred, target, cont, diff_lbl, diff_scale,
                                  lambda_diff=lambda_diff, training=True)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_loader)

        model.eval()
        va_loss, mse_acc, diff_acc, mae_bps = 0.0, torch.zeros(6), 0.0, 0.0
        n_bat = len(va_loader)
        for cont, conv, strata, target, diff_lbl, sigma_n, T in va_loader:
            cont     = cont.to(DEVICE).requires_grad_(True)
            conv     = conv.to(DEVICE)
            strata   = strata.to(DEVICE)
            target   = target.to(DEVICE)
            diff_lbl = diff_lbl.to(DEVICE)

            pred = model(cont, conv, strata)
            loss, mse_p, d_loss = dml_loss(pred, target, cont, diff_lbl, diff_scale,
                                           lambda_diff=lambda_diff, training=False)
            va_loss  += loss.item()
            mse_acc  += mse_p.cpu()
            diff_acc += d_loss.item()
            scale     = sigma_n * torch.sqrt(T)
            mae_bps  += ((pred[:, 0] - target[:, 0]).abs() * scale).mean().item() * 10_000

        va_loss /= n_bat; mse_acc /= n_bat; diff_acc /= n_bat; mae_bps /= n_bat
        history.append((tr_loss, va_loss, mae_bps, diff_acc))

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), "research/swaption_nn_v3_best.pt")

        if ep % 5 == 0 or ep == 1:
            m = mse_acc
            print(f"{ep:>4} {tr_loss:>9.5f} {va_loss:>9.5f} "
                  f"{diff_acc:>10.6f} {m[0].item():>8.5f} "
                  f"{m[1].item():>8.5f} {m[2].item():>8.5f}  {mae_bps:>6.3f}")

    return history


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def compare(hist_v2, hist_v3):
    print(f"\n{'─'*62}")
    print(f"{'Epoch':>6} {'v2 MAE bps':>12} {'v3 DML bps':>12} {'Improvement':>14}")
    print(f"{'─'*62}")
    for ep in [1, 5, 10, 20, 30, 40, 50]:
        if ep > len(hist_v2) or ep > len(hist_v3): continue
        v2, v3 = hist_v2[ep-1][2], hist_v3[ep-1][2]
        print(f"{ep:>6} {v2:>12.3f} {v3:>12.3f} {v2/v3:>13.1f}×")
    print(f"{'─'*62}")


def per_stratum_comparison(model_v2, model_v3, dataset):
    model_v2.eval(); model_v3.eval()
    cont    = dataset.cont.to(DEVICE)
    conv    = dataset.conv.to(DEVICE)
    strata  = dataset.strata.to(DEVICE)
    target  = dataset.target
    scale   = dataset.sigma_n * torch.sqrt(dataset.T)

    with torch.no_grad():
        p2 = model_v2(cont, conv, strata).cpu()
    with torch.no_grad():
        p3 = model_v3(cont, conv, strata).cpu()

    e2 = (p2[:, 0] - target[:, 0]).abs() * scale * 10_000
    e3 = (p3[:, 0] - target[:, 0]).abs() * scale * 10_000

    print(f"\n{'─'*62}")
    print(f"{'Stratum':<14} {'N':>7} {'v2 bps':>10} {'v3 bps':>10} {'Gain':>10}")
    print(f"{'─'*62}")
    for si, name in enumerate(["ATM", "OTM", "Deep-OTM"]):
        mask = dataset.strata == si
        if not mask.any(): continue
        m2, m3 = e2[mask].mean().item(), e3[mask].mean().item()
        print(f"{name:<14} {mask.sum():>7,} {m2:>10.3f} {m3:>10.3f} {m2/m3:>9.1f}×")

    print(f"\n{'─'*62}")
    print(f"{'Greek':<14} {'v2 MAE':>10} {'v3 MAE':>10} {'Gain':>10}")
    print(f"{'─'*62}")
    for gi, gname in enumerate(GREEK_TARGETS):
        m2 = (p2[:, gi] - target[:, gi]).abs().mean().item()
        m3 = (p3[:, gi] - target[:, gi]).abs().mean().item()
        print(f"{gname:<14} {m2:>10.5f} {m3:>10.5f} {m2/m3:>9.1f}×")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 66)
    print("Differential Machine Learning — Swaption Pricer v3")
    print("Huge & Savine (2020)  arXiv:2005.02347")
    print("=" * 66)

    N_TR, N_VA = 300_000, 30_000

    print("\n[1] Generating data")
    data_tr = generate_dataset(N_TR, seed=0)
    data_va = generate_dataset(N_VA, seed=99)

    # v2 baseline uses SwaptionDataset (6-tuple batches)
    # v3 DML uses DMLDataset (7-tuple batches, adds diff_labels)
    ds_tr_v2 = SwaptionDataset(data_tr)
    ds_va_v2 = SwaptionDataset(data_va)
    ds_tr    = DMLDataset(data_tr)
    ds_va    = DMLDataset(data_va)
    del data_tr, data_va   # tensors now live in datasets

    # Scale weights: price_std / diff_std per dimension (Huge & Savine §3.3)
    diff_scale = (ds_tr.target[:, 0].std() / ds_tr.diff_labels.std(dim=0).clamp(min=1e-6)
                  ).unsqueeze(0)   # [1, 3]
    print(f"  Diff scales — Δ:{diff_scale[0,0]:.3f}  ν:{diff_scale[0,1]:.3f}  T:{diff_scale[0,2]:.3f}")

    ld_tr_v2 = DataLoader(ds_tr_v2, batch_size=4096, shuffle=True,  num_workers=0)
    ld_va_v2 = DataLoader(ds_va_v2, batch_size=4096, shuffle=False, num_workers=0)
    ld_tr    = DataLoader(ds_tr,    batch_size=4096, shuffle=True,  num_workers=0)
    ld_va    = DataLoader(ds_va,    batch_size=4096, shuffle=False, num_workers=0)

    print("\n[2] Models — identical architecture (413k params each)")
    model_v2 = SwaptionNetV2(hidden=192, n_blocks=5).to(DEVICE)
    model_v3 = SwaptionNetV2(hidden=192, n_blocks=5).to(DEVICE)

    print("\n[3] Training v2 baseline (plain MSE + pinball)")
    hist_v2 = train(model_v2, ld_tr_v2, ld_va_v2, epochs=50, lr=4e-4)
    model_v2.load_state_dict(torch.load("research/swaption_nn_v2_best.pt", weights_only=True))

    print("\n[4] Training v3 DML (MSE + pinball + Jacobian matching)")
    hist_v3 = train_dml(model_v3, ld_tr, ld_va, diff_scale, epochs=50, lr=4e-4, lambda_diff=5.0)
    model_v3.load_state_dict(torch.load("research/swaption_nn_v3_best.pt", weights_only=True))

    print("\n[5] Results")
    compare(hist_v2, hist_v3)
    per_stratum_comparison(model_v2, model_v3, ds_va_v2)

    print(f"""
{"="*66}
Differential ML — changes vs v2
{"─"*66}
Loss      : price_MSE + 5.0 × Jacobian_MSE
Jacobian  : supervised on 3 inputs via autograd
              moneyness → delta   (exact: ∂norm_price/∂d = N(d))
              sigma_atm → vega    (approx via ZABR chain)
              T         → theta   (exact: (θ_n - p_n)/2T)
Scale     : price_std / diff_std per dim  (Huge & Savine §3.3)
Cost      : ~2× per training batch  (create_graph=True)
            validation uses create_graph=False — no overhead
{"="*66}
""")
