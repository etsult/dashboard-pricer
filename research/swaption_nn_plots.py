"""
Performance visualisation for SwaptionNetV2.
Loads trained model + generates validation data → saves plots to research/swaption_nn_plots.png
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings("ignore")

# ── re-import everything from v2 ──────────────────────────────────────────────
import sys, importlib, types

# inline re-import of core functions to avoid re-running __main__
exec(open("research/swaption_nn_v2.py").read().split("if __name__")[0])

DEVICE = torch.device("cpu")

# ── 1. Generate fresh validation set + training history ───────────────────────
print("Generating validation data...")
N_VA  = 40_000
data  = generate_dataset(N_VA, seed=77)
ds    = SwaptionDataset(data)

# Re-run training with history captured (25 epochs, fast)
print("Re-training 50 epochs to capture loss history...")
N_TR   = 300_000
data_tr = generate_dataset(N_TR, seed=0)
ds_tr   = SwaptionDataset(data_tr)
ld_tr   = torch.utils.data.DataLoader(ds_tr, batch_size=4096, shuffle=True)
ld_va   = torch.utils.data.DataLoader(ds,    batch_size=4096, shuffle=False)

model  = SwaptionNetV2(hidden=192, n_blocks=5).to(DEVICE)
history = train(model, ld_tr, ld_va, epochs=50, lr=4e-4)
# history: list of (tr_loss, va_loss, mae_bps)

# ── 2. Inference on validation set ───────────────────────────────────────────
model.eval()
cont   = ds.cont.to(DEVICE)
conv   = ds.conv.to(DEVICE)
strata = ds.strata.to(DEVICE)
with torch.no_grad():
    pred = model(cont, conv, strata).cpu().numpy()

target  = ds.target.numpy()
sigma_n = ds.sigma_n.numpy()
T_arr   = ds.T.numpy()
scale   = sigma_n * np.sqrt(T_arr)         # denormalisation factor
strata_np = ds.strata.numpy()
conv_np   = ds.conv.numpy()

# Raw error in bps
err_bps = np.abs(pred[:,0] - target[:,0]) * scale * 1e4
moneyness = ds.cont[:,0].numpy()           # d = (F-K)/(σ√T)

# ── 3. Plot ───────────────────────────────────────────────────────────────────
STRATA_COLORS  = ["#2ecc71", "#e67e22", "#e74c3c"]
STRATA_LABELS  = ["ATM (60%)", "OTM (25%)", "Deep-OTM (15%)"]
CONV_COLORS    = ["#3498db", "#9b59b6", "#1abc9c"]
GREEK_NAMES    = ["norm_price", "delta", "norm_vega", "norm_gamma", "norm_theta", "norm_vanna"]
GREEK_LABELS   = ["Price", "Delta", "Vega", "Gamma", "Theta", "Vanna"]

fig = plt.figure(figsize=(20, 22), facecolor="#0f1117")
fig.suptitle("SwaptionNet v2 — Performance Report\n"
             "300k synthetic samples · ZABR surface · Two-curve · 3-strata tail sampling",
             color="white", fontsize=14, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35,
                       left=0.07, right=0.96, top=0.94, bottom=0.04)

dark_bg  = "#0f1117"
panel_bg = "#1a1d27"
txt_col  = "#e0e0e0"
grid_col = "#2a2d3a"

def style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(panel_bg)
    ax.set_title(title, color=txt_col, fontsize=10, pad=6)
    ax.set_xlabel(xlabel, color=txt_col, fontsize=8)
    ax.set_ylabel(ylabel, color=txt_col, fontsize=8)
    ax.tick_params(colors=txt_col, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(grid_col)
    ax.grid(True, color=grid_col, linewidth=0.5, alpha=0.7)


# ── (0,0) Training & validation loss ─────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
epochs = list(range(1, len(history)+1))
tr_losses  = [h[0] for h in history]
va_losses  = [h[1] for h in history]
mae_hist   = [h[2] for h in history]
ax.plot(epochs, tr_losses, color="#3498db", lw=1.5, label="Train loss")
ax.plot(epochs, va_losses, color="#e74c3c", lw=1.5, label="Val loss")
ax.set_yscale("log")
style_ax(ax, "Loss Curves (log scale)", "Epoch", "Loss")
ax.legend(fontsize=7, facecolor=panel_bg, labelcolor=txt_col, framealpha=0.8)

# ── (0,1) MAE bps over training ───────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
ax.plot(epochs, mae_hist, color="#2ecc71", lw=1.5)
ax.axhline(mae_hist[-1], color="#e74c3c", ls="--", lw=1, label=f"Final: {mae_hist[-1]:.3f} bps")
style_ax(ax, "Price MAE (bps) — Val set", "Epoch", "bps")
ax.legend(fontsize=7, facecolor=panel_bg, labelcolor=txt_col, framealpha=0.8)

# ── (0,2) Error CDF by stratum ───────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
for si, (col, lbl) in enumerate(zip(STRATA_COLORS, STRATA_LABELS)):
    mask = strata_np == si
    e = np.sort(err_bps[mask])
    cdf = np.linspace(0, 1, len(e))
    ax.plot(e, cdf, color=col, lw=1.5, label=lbl)
ax.axvline(1.0, color="white", ls="--", lw=0.8, alpha=0.5, label="1 bps")
ax.set_xlim(0, 5)
style_ax(ax, "Error CDF by Stratum", "MAE (bps)", "Cumulative %")
ax.legend(fontsize=7, facecolor=panel_bg, labelcolor=txt_col, framealpha=0.8)

# ── (1,0) Predicted vs Actual — price ────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
# Subsample for scatter
idx = np.random.choice(N_VA, 4000, replace=False)
for si, (col, lbl) in enumerate(zip(STRATA_COLORS, STRATA_LABELS)):
    mask = strata_np[idx] == si
    ax.scatter(target[idx[mask],0], pred[idx[mask],0],
               s=2, alpha=0.4, color=col, label=lbl, rasterized=True)
lims = [target[:,0].min(), target[:,0].max()]
ax.plot(lims, lims, "w--", lw=0.8, alpha=0.6)
style_ax(ax, "Predicted vs Actual — Price (norm.)", "True", "Predicted")
ax.legend(fontsize=6, facecolor=panel_bg, labelcolor=txt_col, framealpha=0.7,
          markerscale=3)

# ── (1,1) Error vs Moneyness (d) ─────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
bins = np.linspace(-4, 4, 40)
bin_idx = np.digitize(moneyness, bins) - 1
bin_mae, bin_p95, bin_ctr = [], [], []
for b in range(len(bins)-1):
    mask = bin_idx == b
    if mask.sum() < 10: continue
    bin_mae.append(err_bps[mask].mean())
    bin_p95.append(np.percentile(err_bps[mask], 95))
    bin_ctr.append(0.5*(bins[b]+bins[b+1]))
ax.fill_between(bin_ctr, 0, bin_p95, alpha=0.25, color="#e74c3c", label="P95")
ax.plot(bin_ctr, bin_mae, color="#e74c3c", lw=1.5, label="Mean MAE")
ax.axvline(0, color="white", ls="--", lw=0.8, alpha=0.5)
ax.set_xlim(-4, 4)
style_ax(ax, "Price MAE vs Moneyness d=(F-K)/σ√T", "Moneyness (d)", "bps")
ax.legend(fontsize=7, facecolor=panel_bg, labelcolor=txt_col, framealpha=0.8)

# ── (1,2) Error heatmap: moneyness × expiry ──────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
T_arr_plot = T_arr
hm_d = np.clip(moneyness, -3, 3)
hm_T = np.clip(T_arr_plot, 0, 10)
h2d, xe, ye = np.histogram2d(hm_d, hm_T, bins=25,
                               range=[[-3,3],[0,10]],
                               weights=err_bps)
cnt, _, _   = np.histogram2d(hm_d, hm_T, bins=25, range=[[-3,3],[0,10]])
cnt = np.maximum(cnt, 1)
hm = (h2d / cnt).T
im = ax.imshow(hm, origin="lower", aspect="auto",
               extent=[-3,3,0,10], cmap="plasma",
               vmin=0, vmax=np.percentile(err_bps, 95))
cb = plt.colorbar(im, ax=ax)
cb.ax.tick_params(colors=txt_col, labelsize=7)
cb.set_label("MAE bps", color=txt_col, fontsize=7)
ax.axvline(0, color="white", ls="--", lw=0.8, alpha=0.5)
style_ax(ax, "MAE Heatmap: Moneyness × Expiry", "Moneyness (d)", "Expiry (years)")

# ── (2,0-2) Greek accuracy: delta, vega, vanna ───────────────────────────────
for gi, (gidx, glbl, col) in enumerate(zip([1,2,5],
                                            ["Delta","Norm Vega","Norm Vanna"],
                                            ["#3498db","#e67e22","#9b59b6"])):
    ax = fig.add_subplot(gs[2, gi])
    idx2 = np.random.choice(N_VA, 3000, replace=False)
    ax.scatter(target[idx2, gidx], pred[idx2, gidx],
               s=2, alpha=0.35, color=col, rasterized=True)
    lims2 = [target[:,gidx].min(), target[:,gidx].max()]
    ax.plot(lims2, lims2, "w--", lw=0.8, alpha=0.6)
    mae_g = np.abs(pred[:,gidx] - target[:,gidx]).mean()
    style_ax(ax, f"{glbl}: pred vs true\nMAE={mae_g:.5f}", "True", "Predicted")

# ── (3,0) Vanna error vs moneyness ───────────────────────────────────────────
ax = fig.add_subplot(gs[3, 0])
vanna_err = np.abs(pred[:,5] - target[:,5])
bin_mae_v, bin_ctr_v = [], []
for b in range(len(bins)-1):
    mask = (bin_idx == b)
    if mask.sum() < 10: continue
    bin_mae_v.append(vanna_err[mask].mean())
    bin_ctr_v.append(0.5*(bins[b]+bins[b+1]))
ax.plot(bin_ctr_v, bin_mae_v, color="#9b59b6", lw=1.5)
ax.axvline(0, color="white", ls="--", lw=0.8, alpha=0.5)
ax.set_xlim(-4, 4)
style_ax(ax, "Vanna MAE vs Moneyness", "Moneyness (d)", "MAE (norm.)")

# ── (3,1) MAE bps by convention ──────────────────────────────────────────────
ax = fig.add_subplot(gs[3, 1])
conv_names_short = ["USD\nSOFR", "EUR\nEURIB", "GBP\nSONIA"]
means, p95s = [], []
for ci in range(N_CONV):
    mask = conv_np == ci
    means.append(err_bps[mask].mean())
    p95s.append(np.percentile(err_bps[mask], 95))
x = np.arange(N_CONV)
bars = ax.bar(x, means, color=CONV_COLORS, alpha=0.85, width=0.4, label="Mean MAE")
ax.scatter(x, p95s, color="white", s=40, zorder=5, label="P95")
ax.set_xticks(x); ax.set_xticklabels(conv_names_short, color=txt_col, fontsize=8)
for bar, v in zip(bars, means):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.005, f"{v:.3f}", ha="center",
            color=txt_col, fontsize=7)
style_ax(ax, "Price MAE by Convention", "", "bps")
ax.legend(fontsize=7, facecolor=panel_bg, labelcolor=txt_col, framealpha=0.8)

# ── (3,2) MAE bps by expiry bucket ───────────────────────────────────────────
ax = fig.add_subplot(gs[3, 2])
exp_bins  = [0, 0.5, 1, 2, 3, 5, 7, 10]
exp_lbls  = ["<6M","6M-1Y","1-2Y","2-3Y","3-5Y","5-7Y","7-10Y"]
exp_idx   = np.digitize(T_arr, exp_bins) - 1
exp_means, exp_p95s = [], []
valid_lbls = []
for bi in range(len(exp_bins)-1):
    mask = exp_idx == bi
    if mask.sum() < 20:
        continue
    exp_means.append(err_bps[mask].mean())
    exp_p95s.append(np.percentile(err_bps[mask], 95))
    valid_lbls.append(exp_lbls[bi])
xb = np.arange(len(exp_means))
ax.bar(xb, exp_means, color="#1abc9c", alpha=0.85, width=0.5, label="Mean MAE")
ax.scatter(xb, exp_p95s, color="white", s=40, zorder=5, label="P95")
ax.set_xticks(xb); ax.set_xticklabels(valid_lbls, color=txt_col, fontsize=7)
style_ax(ax, "Price MAE by Expiry Bucket", "", "bps")
ax.legend(fontsize=7, facecolor=panel_bg, labelcolor=txt_col, framealpha=0.8)

fig.savefig("research/swaption_nn_plots.png", dpi=150, bbox_inches="tight",
            facecolor=dark_bg)
print("\nSaved → research/swaption_nn_plots.png")
