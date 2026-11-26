"""
Publication figures (reviewer comment 6: legibility, resolution, caption/figure consistency).

  * 300 dpi PNG + vector PDF, large fonts, colour-blind-safe palette, no overlapping text.
  * PROVENANCE: every figure writes <name>.json with the checkpoint, split, image id(s), seed and the
    exact numbers shown -- the manuscript's captions/numbers must be copied from that file, never typed.
    (In the submitted paper Fig. 5 showed betas from a 3-epoch model while the text quoted another run,
    and Fig. 3's caption described a different plot.)
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9,
                     "figure.dpi": 120, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.prop_cycle": matplotlib.cycler(color=PALETTE), "pdf.fonttype": 42})


def save(fig, out_dir, name, provenance=None):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{name}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"), bbox_inches="tight")
    if provenance is not None:
        json.dump(provenance, open(os.path.join(out_dir, f"{name}.json"), "w"), indent=1, default=float)
    plt.close(fig)


def bars_with_error(summary, metric_cols, title, out_dir, name, ylabel="", ref=None):
    """summary: DataFrame config,mean,std(,n_seeds).  Error bars = std over seeds."""
    fig, ax = plt.subplots(figsize=(1.1 * len(summary) + 2, 3.8))
    x = np.arange(len(summary))
    ax.bar(x, summary["mean"], yerr=summary["std"].fillna(0), capsize=4, color=PALETTE[0], alpha=.9)
    lo = (summary["mean"] - summary["std"].fillna(0)).min(); hi = (summary["mean"] + summary["std"].fillna(0)).max()
    ax.set_ylim(lo - 0.3 * (hi - lo + 1e-6), hi + 0.2 * (hi - lo + 1e-6))
    ax.set_xticks(x); ax.set_xticklabels(summary["config"], rotation=30, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title)
    for xi, (m, n) in enumerate(zip(summary["mean"], summary.get("n_seeds", [None] * len(summary)))):
        ax.text(xi, ax.get_ylim()[0], f"n={n}", ha="center", va="bottom", fontsize=8, color="w")
    save(fig, out_dir, name, {"table": summary.to_dict("records"), "metric": ylabel, "error_bars": "std over seeds"})


def training_curves(history_csv, out_dir, name, label="", provenance=None):
    import pandas as pd
    h = pd.read_csv(history_csv)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    ax[0].plot(h.epoch, h.train_loss, label="train"); ax[0].plot(h.epoch, h.val_loss, label="val"); ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(h.epoch, h.val_psnr, color=PALETTE[2]); ax[1].set_title("Validation PSNR (dB)")
    ax[2].plot(h.epoch, h.val_ssim, color=PALETTE[4]); ax[2].set_title("Validation SSIM")
    for a in ax: a.set_xlabel("epoch")
    fig.suptitle(label, y=1.02)
    save(fig, out_dir, name, provenance or {"history": history_csv, "best_val_psnr": float(h.val_psnr.max())})


def component_health(history_csv, out_dir, name, beta_min=0.3, beta_max=2.5, provenance=None):
    import pandas as pd
    h = pd.read_csv(history_csv)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for k in range(3):
        ax[0].plot(h.epoch, h[f"beta{k}"], label=f"$\\beta_{k}$"); ax[2].plot(h.epoch, h[f"usage{k}"], label=f"comp {k}")
    ax[0].axhline(beta_min, ls=":", c="gray"); ax[0].axhline(beta_max, ls=":", c="gray"); ax[0].set_title("Learned $\\beta_k$ (dotted: bounds)"); ax[0].legend()
    ax[1].plot(h.epoch, h.min_beta_dist); ax[1].axhline(0.3, ls="--", c="gray"); ax[1].set_title("Min pairwise $\\beta$ distance (dashed: margin)")
    ax[2].set_title("Mean component usage"); ax[2].legend()
    for a in ax: a.set_xlabel("epoch")
    final = {f"beta{k}": float(h[f"beta{k}"].iloc[-1]) for k in range(3)}
    save(fig, out_dir, name, provenance or {"history": history_csv, "final": final})


def mixture_maps(img, mix_w, beta, out_dir, name, provenance):
    """img [H,W]; mix_w [K,H,W]; beta [K].  The 'dominant' panel makes single-component collapse obvious."""
    K = mix_w.shape[0]
    fig, ax = plt.subplots(1, K + 2, figsize=(2.6 * (K + 2), 2.9))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("Input")
    for k in range(K):
        im = ax[k + 1].imshow(mix_w[k], vmin=0, vmax=1, cmap="viridis"); ax[k + 1].set_title(f"$\\pi_{k}$ ($\\beta$={beta[k]:.2f})")
    dom = mix_w.argmax(0); ax[K + 1].imshow(dom, cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
    fr = np.bincount(dom.ravel(), minlength=K) / dom.size
    ax[K + 1].set_title("Dominant: " + "/".join(f"{f:.0%}" for f in fr))
    for a in ax: a.axis("off")
    fig.colorbar(im, ax=ax[:K + 1], fraction=0.015, pad=0.01)
    provenance = {**provenance, "betas": [float(b) for b in beta], "dominant_fraction": fr.tolist()}
    save(fig, out_dir, name, provenance)


def qualitative_grid(items, out_dir, name, provenance, zoom=None):
    """items: list of (title, 2-D array); first is clean GT. Adds absolute-error maps and an optional zoom box."""
    gt = items[0][1]
    n = len(items)
    fig, ax = plt.subplots(2, n, figsize=(2.6 * n, 5.4))
    for j, (t, a) in enumerate(items):
        ax[0, j].imshow(a, cmap="gray", vmin=0, vmax=1); ax[0, j].set_title(t, fontsize=9)
        e = np.abs(a - gt); ax[1, j].imshow(e, cmap="inferno", vmin=0, vmax=0.25)
        ax[1, j].set_title(f"|err|  mean={e.mean():.3f}" if j else "|error| (0-0.25)", fontsize=9)
        if zoom is not None:
            y0, y1, x0, x1 = zoom
            ax[0, j].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="cyan", lw=1))
        ax[0, j].axis("off"); ax[1, j].axis("off")
    save(fig, out_dir, name, provenance)
    if zoom is not None:
        y0, y1, x0, x1 = zoom
        fig, ax = plt.subplots(1, n, figsize=(2.6 * n, 2.9))
        for j, (t, a) in enumerate(items):
            ax[j].imshow(a[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=1, interpolation="nearest"); ax[j].set_title(t, fontsize=9); ax[j].axis("off")
        save(fig, out_dir, name + "_zoom", provenance)


def scatter_beta(df, feats, out_dir, name, provenance, n_max=20000):
    """hexbin of patch beta vs structural features; caption numbers come from `provenance`."""
    fig, ax = plt.subplots(1, len(feats), figsize=(3.8 * len(feats), 3.4))
    d = df.sample(min(len(df), n_max), random_state=0)
    for a, f in zip(np.atleast_1d(ax), feats):
        hb = a.hexbin(d[f], d.beta, gridsize=40, cmap="viridis", mincnt=1, bins="log")
        a.set_xlabel(f.replace("_", " ")); a.set_ylabel("patch $\\beta$")
        r = np.corrcoef(df[f], df.beta)[0, 1]; a.set_title(f"r = {r:.3f}")
    save(fig, out_dir, name, provenance)
