"""Headless, resumable launcher for the long training jobs (use this instead of keeping notebook 03 open).

    python scripts/run_experiments.py --mode pilot --groups core
    python scripts/run_experiments.py --mode full  --groups core,ablation,external
    python scripts/run_experiments.py --mode full  --groups core --seeds 0,1,2     # subset of seeds
Finished runs (results.json present) are skipped, so the command can simply be re-run after an interruption.
"""
import argparse
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="pilot", choices=["smoke", "pilot", "full"])
ap.add_argument("--groups", default="core")
ap.add_argument("--seeds", default=None)
ap.add_argument("--configs", default=None, help="comma list overriding --groups")
a = ap.parse_args()
os.environ["SEMICON_MODE"] = a.mode
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from utils.experiment import run_one

GROUPS = {"core": ["baseline", "global_gc", "film_control", "ldmh_loss_only", "full"],
          "ablation": ["full_no_penalty", "full_clamp", "full_no_bal", "full_no_div"],
          "external": ["NAFNet-small", "Restormer-small", "SwinIR-small", "NAFNet-base", "Restormer-base", "SwinIR-base"]}
cfgs = a.configs.split(",") if a.configs else [c for g in a.groups.split(",") for c in GROUPS[g]]
seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else C.SEEDS
splits, man = C.get_splits()
C.banner(); plan = [(c, s) for c in cfgs for s in seeds]
print(f"{len(plan)} runs: {cfgs} x seeds {seeds}")
t0 = time.time()
for k, (c, s) in enumerate(plan):
    r = run_one(c, s, splits, run_root=C.RUNS, epochs=C.EPOCHS, batch_size=C.BATCH_SIZE, lr=C.LR, wd=C.WEIGHT_DECAY,
                clip=C.GRAD_CLIP, patience=C.PATIENCE, device=C.device(), cache=C.CACHE, verbose=True)
    print(f"[{k+1}/{len(plan)}] {c} seed{s}: test_psnr={r['test']['psnr']:.3f} ({(time.time()-t0)/60:.1f} min elapsed)", flush=True)
