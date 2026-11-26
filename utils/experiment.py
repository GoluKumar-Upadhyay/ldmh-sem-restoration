"""
Experiment registry + single-run trainer.

Every configuration is trained with the SAME protocol (split manifest, augmentation stream,
optimiser, schedule, epochs, patience, checkpoint rule).  A run is identified by
(config_name, seed) and is skipped when `results.json` already exists, so long jobs on a laptop
GPU can be interrupted and resumed.

Outputs per run  (runs/<config>/seed<seed>/):
    best.pth                best-validation-PSNR weights (+ config)
    history.csv             per-epoch train/val loss, val PSNR/SSIM, LR, component statistics
    test_per_image.csv      per-image TEST metrics at the best-val checkpoint   <- used for paired stats
    results.json            summary (best epoch, val & test means, params, time)
"""
import csv
import json
import os
import random
import time
import numpy as np
import torch

from models.restoration_net import BaselineNet, DistributionMixtureRestorationNet
from models.baselines import BASELINE_REGISTRY
from models.ldmh import mixture_gen_gaussian_nll, beta_diversity_penalty, load_balance_penalty_entropy
from utils.losses import charbonnier_loss, gen_charbonnier_loss
from utils import metrics as M

# ----------------------------------------------------------------------------- registry
# arch: 'baseline' | 'ldmh' | 'ext:<name>' ; loss: 'charbonnier' | 'gen_charbonnier' | 'mixture'
CONFIGS = {
    # --- the ladder in the paper ------------------------------------------------------
    "baseline":        dict(arch="baseline", loss="charbonnier"),
    "global_gc":       dict(arch="baseline", loss="gen_charbonnier"),
    "ldmh_loss_only":  dict(arch="ldmh", film=False, loss="mixture", div=0.1, bal=0.3),
    "full":            dict(arch="ldmh", film=True, loss="mixture", div=0.1, bal=0.3),
    # --- NEW controls requested by the audit ------------------------------------------
    "film_control":    dict(arch="ldmh", film=True, loss="charbonnier"),            # same capacity, no NLL
    "full_no_penalty": dict(arch="ldmh", film=True, loss="mixture", div=0.0, bal=0.0),   # = old Table V loss
    "full_clamp":      dict(arch="ldmh", film=True, loss="mixture", div=0.1, bal=0.3, beta_mode="clamp"),
    "full_no_bal":     dict(arch="ldmh", film=True, loss="mixture", div=0.1, bal=0.0),
    "full_no_div":     dict(arch="ldmh", film=True, loss="mixture", div=0.0, bal=0.3),
    # --- external architectures (same protocol) ---------------------------------------
    **{k: dict(arch=f"ext:{k}", loss="charbonnier") for k in BASELINE_REGISTRY},
}
LADDER = ["baseline", "global_gc", "film_control", "ldmh_loss_only", "full"]
ABLATION = ["baseline", "global_gc", "film_control", "ldmh_loss_only", "full",
            "full_no_penalty", "full_clamp", "full_no_bal", "full_no_div"]
EXTERNAL = ["baseline", "full", "NAFNet-small", "Restormer-small", "SwinIR-small",
            "NAFNet-base", "Restormer-base", "SwinIR-base"]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg):
    a = cfg["arch"]
    if a == "baseline":
        return BaselineNet()
    if a == "ldmh":
        return DistributionMixtureRestorationNet(use_film=cfg.get("film", True),
                                                 beta_mode=cfg.get("beta_mode", "sigmoid"),
                                                 init_scale=cfg.get("init_scale", 0.05))
    if a.startswith("ext:"):
        return BASELINE_REGISTRY[a[4:]]()
    raise ValueError(a)


def is_ldmh(cfg):
    return cfg["arch"] == "ldmh"


def compute_loss(cfg, out, gt, gt_ds, noisy):
    pred = out[0] if is_ldmh(cfg) else out
    kind = cfg["loss"]
    if kind == "charbonnier":
        loss = charbonnier_loss(pred, gt)
    elif kind == "gen_charbonnier":
        loss = gen_charbonnier_loss(pred, gt, p=cfg.get("global_beta", 0.845))
    elif kind == "mixture":
        _, w, beta, scale = out
        nll = mixture_gen_gaussian_nll(noisy - gt_ds, w, beta, scale)
        loss = (charbonnier_loss(pred, gt) + cfg.get("nll", 1.0) * nll
                + cfg.get("div", 0.0) * beta_diversity_penalty(beta, 0.3)
                + cfg.get("bal", 0.0) * load_balance_penalty_entropy(w))
    else:
        raise ValueError(kind)
    return loss, pred


def _param_groups(model, wd):
    proto = [p for n, p in model.named_parameters() if n.endswith("log_beta") or n.endswith("log_scale")]
    ids = {id(p) for p in proto}
    rest = [p for p in model.parameters() if id(p) not in ids]
    g = [{"params": rest, "weight_decay": wd}]
    if proto:
        g.append({"params": proto, "weight_decay": 0.0})
    return g


@torch.no_grad()
def evaluate(model, cfg, loader, device, per_image=False, with_loss=False):
    model.eval()
    rows, losses = [], []
    for noisy, gt, gt_ds, idx in loader:
        noisy, gt, gt_ds = noisy.to(device), gt.to(device), gt_ds.to(device)
        out = model(noisy)
        if with_loss:
            losses.append(compute_loss(cfg, out, gt, gt_ds, noisy)[0].item())
        pred = out[0] if is_ldmh(cfg) else out
        m = M.full_eval(pred, gt, noisy)
        for j in range(noisy.shape[0]):
            rows.append({"idx": int(idx[j]), **{k: float(v[j]) for k, v in m.items()}})
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "idx"}
    if with_loss:
        mean["loss"] = float(np.mean(losses))
    return (mean, rows) if per_image else mean


def run_one(name, seed, splits, run_root="runs", epochs=200, batch_size=8, lr=2e-4, wd=1e-4,
            clip=1.0, patience=30, device=None, cache=True, verbose=True, cfg_override=None):
    from utils.data import make_loaders
    cfg = dict(CONFIGS[name]); cfg.update(cfg_override or {})
    out_dir = os.path.join(run_root, name, f"seed{seed}")
    res_path = os.path.join(out_dir, "results.json")
    if os.path.exists(res_path):
        if verbose: print(f"[skip] {name} seed{seed} (done)")
        return json.load(open(res_path))
    os.makedirs(out_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device == "cuda"
    if device == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))

    set_seed(seed)
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    train_l, val_l, test_l = make_loaders(splits, batch_size, seed=seed, cache=cache)
    opt = torch.optim.AdamW(_param_groups(model, wd), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    fields = ["epoch", "train_loss", "val_loss", "val_psnr", "val_ssim", "lr", "time_sec"]
    if is_ldmh(cfg):
        fields += [f"beta{k}" for k in range(3)] + [f"usage{k}" for k in range(3)] + ["min_beta_dist", "any_beta_at_bound"]
    f = open(os.path.join(out_dir, "history.csv"), "w", newline="")
    wr = csv.DictWriter(f, fieldnames=fields); wr.writeheader()

    best, best_ep, bad, t_start = -1e9, -1, 0, time.time()
    for ep in range(1, epochs + 1):
        t0 = time.time(); model.train(); tl = []
        for noisy, gt, gt_ds, _ in train_l:
            noisy, gt, gt_ds = noisy.to(device), gt.to(device), gt_ds.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=use_amp):
                loss, _ = compute_loss(cfg, model(noisy), gt, gt_ds, noisy)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt); scaler.update(); tl.append(loss.item())
            if verbose and ep == 1 and len(tl) in (10, 50) or (verbose and ep == 1 and len(tl) % 100 == 0):
                dt = time.time() - t0
                print(f"   [{name}|s{seed}] epoch 1: {len(tl)}/{len(train_l)} batches, {dt:.0f}s "
                      f"-> projected {dt / len(tl) * len(train_l) / 60:.1f} min/epoch", flush=True)
        sched.step()

        vm = evaluate(model, cfg, val_l, device, with_loss=True)
        row = {"epoch": ep, "train_loss": np.mean(tl), "val_loss": vm["loss"], "val_psnr": vm["psnr"],
               "val_ssim": vm["ssim"], "lr": opt.param_groups[0]["lr"], "time_sec": time.time() - t0}
        if is_ldmh(cfg):
            with torch.no_grad():
                noisy = next(iter(val_l))[0].to(device)
                _, w, b, s = model(noisy)
                cs = model.ldmh.get_component_stats(w, b, s)
            for k in range(3):
                row[f"beta{k}"], row[f"usage{k}"] = cs["beta"][k], cs["mean_usage"][k]
            row["min_beta_dist"], row["any_beta_at_bound"] = cs["min_beta_pairwise_dist"], int(any(cs["beta_hit_bound"]))
        wr.writerow(row); f.flush()
        if verbose:
            et = time.time() - t0; eta = et * (epochs - ep) / 60
            print(f"[{name}|s{seed}] ep {ep}/{epochs} train_loss={np.mean(tl):.5f} val_psnr={vm['psnr']:.3f} "
                  f"val_ssim={vm['ssim']:.4f} | {et / 60:.1f} min/epoch, ~{eta:.0f} min left for this run", flush=True)
        if vm["psnr"] > best:
            best, best_ep, bad = vm["psnr"], ep, 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": ep, "seed": seed}, os.path.join(out_dir, "best.pth"))
        else:
            bad += 1
            if bad >= patience:
                break
    f.close()

    ck = torch.load(os.path.join(out_dir, "best.pth"), map_location=device)
    model.load_state_dict(ck["model"])
    val_m = evaluate(model, cfg, val_l, device)
    test_m, rows = evaluate(model, cfg, test_l, device, per_image=True)     # TEST touched exactly once
    with open(os.path.join(out_dir, "test_per_image.csv"), "w", newline="") as g:
        w = csv.DictWriter(g, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    res = {"name": name, "seed": seed, "best_epoch": best_ep, "epochs_run": ep, "params": n_params,
           "val": val_m, "test": test_m, "train_minutes": (time.time() - t_start) / 60,
           "device": device, "cfg": cfg}
    json.dump(res, open(res_path, "w"), indent=1)
    return res


def load_results(run_root, names=None):
    """-> (summary DataFrame, dict[name] -> per-image DataFrame list)"""
    import pandas as pd, glob
    rows = []
    for p in glob.glob(os.path.join(run_root, "*", "seed*", "results.json")):
        r = json.load(open(p))
        if names and r["name"] not in names: continue
        rows.append({"config": r["name"], "seed": r["seed"], "params": r["params"], "best_epoch": r["best_epoch"],
                     "minutes": r["train_minutes"], **{f"val_{k}": v for k, v in r["val"].items()},
                     **{f"test_{k}": v for k, v in r["test"].items()}})
    return pd.DataFrame(rows).sort_values(["config", "seed"]).reset_index(drop=True)


def load_per_image(run_root, name, metric="psnr"):
    """-> array [n_seeds, n_test_images] for `metric`, aligned on idx."""
    import pandas as pd, glob
    mats = []
    for p in sorted(glob.glob(os.path.join(run_root, name, "seed*", "test_per_image.csv"))):
        d = pd.read_csv(p).sort_values("idx"); mats.append(d[metric].values)
    return np.stack(mats) if mats else np.empty((0, 0))
