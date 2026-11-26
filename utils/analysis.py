"""Helpers shared by the analysis notebooks (checkpoint loading, restoration, robustness grid, cost)."""
import os
import time
import numpy as np
import pandas as pd
import torch

from utils.experiment import build_model, is_ldmh
from utils import metrics as M
from utils.degrade import degrade


def load_run_model(run_root, name, seed, device="cpu"):
    p = os.path.join(run_root, name, f"seed{seed}", "best.pth")
    ck = torch.load(p, map_location=device)
    m = build_model(ck["cfg"]).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck["cfg"], p


@torch.no_grad()
def restore(model, cfg, x):
    out = model(x)
    return out[0] if is_ldmh(cfg) else out


def finished_seeds(run_root, name):
    d = os.path.join(run_root, name)
    if not os.path.isdir(d): return []
    return sorted(int(s[4:]) for s in os.listdir(d) if os.path.exists(os.path.join(d, s, "results.json")))


# ----------------------------------------------------------------------------- robustness grid
CONDITIONS = {
    "speckle only": dict(speckle=True, gaussian=False),
    "gaussian only": dict(speckle=False, gaussian=True),
    "downsample only": dict(speckle=False, gaussian=False),
    "combined": dict(speckle=True, gaussian=True),
}


@torch.no_grad()
def robustness_grid(run_root, names, seeds_by_name, clean_hr, looks, sigma, noise_seeds=(0, 1, 2), device="cpu", bs=16):
    """clean_hr: [N,H,W] float32 from the TEST split.  Returns long DataFrame:
       config, seed, condition, noise_seed, img, psnr, ssim."""
    rows = []
    N = len(clean_hr)
    for cond, kw in CONDITIONS.items():
        for ns in noise_seeds:
            deg = np.stack([degrade(clean_hr[i], looks, sigma, seed=1000 * ns + i, **kw) for i in range(N)])
            xt = torch.from_numpy(deg)[:, None].to(device); gt = torch.from_numpy(clean_hr)[:, None].to(device)
            for name in names:
                for sd in seeds_by_name[name]:
                    model, cfg, _ = load_run_model(run_root, name, sd, device)
                    for i in range(0, N, bs):
                        p = restore(model, cfg, xt[i:i + bs]).clamp(0, 1)
                        ps, ss = M.psnr(p, gt[i:i + bs]).cpu().numpy(), M.ssim(p, gt[i:i + bs]).cpu().numpy()
                        for j in range(len(ps)):
                            rows.append((name, sd, cond, ns, i + j, float(ps[j]), float(ss[j])))
    return pd.DataFrame(rows, columns=["config", "seed", "condition", "noise_seed", "img", "psnr", "ssim"])


# ----------------------------------------------------------------------------- cost
def count_flops(model, shape=(1, 1, 128, 128), device="cpu"):
    """GFLOPs (2*MACs) of one forward pass counted by torch's FlopCounterMode (conv/matmul/attention)."""
    from torch.utils.flop_counter import FlopCounterMode
    model = model.to(device).eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(torch.randn(*shape, device=device))
    return fc.get_total_flops() / 1e9


@torch.no_grad()
def latency_ms(model, shape, device="cpu", half=False, n_warm=10, n_run=50):
    model = model.to(device).eval()
    x = torch.randn(*shape, device=device)
    if half and device == "cuda":
        model, x = model.half(), x.half()
    for _ in range(n_warm): model(x)
    ts = []
    for _ in range(n_run):
        if device == "cuda":
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); model(x); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
        else:
            t = time.perf_counter(); model(x); ts.append((time.perf_counter() - t) * 1e3)
    ts = np.array(ts)
    return {"median_ms": float(np.median(ts)), "mean_ms": float(ts.mean()), "p95_ms": float(np.percentile(ts, 95)),
            "throughput_img_s": float(shape[0] * 1000.0 / np.median(ts))}
