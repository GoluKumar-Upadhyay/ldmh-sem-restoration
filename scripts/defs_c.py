"""Notebook definitions 07-09."""
M, C = "md", "code"
from defs_a import HEAD

NB = {}

# ======================================================================================= 07
NB["07_Downstream_Defect_Evaluation.ipynb"] = [
(M, '# 07 - Downstream Defect Evaluation\n**Addresses comment 4 (manufacturing relevance).** PSNR/SSIM do not show whether a defect remains visible after restoration. This notebook requires Carinthia-S images **and their masks**. It uses a seeded 70/30 train/test split, a shared frozen defect segmenter, and compares clean, bilinear, and restored outputs. If Carinthia-S is absent, the notebook reports that the analysis was skipped; it does not substitute synthetic or unlabelled data.\n\n**Caveats to state:** restoration models are trained on KLA data and applied to Carinthia-S (domain shift - optionally fine-tune on Carinthia-S train images); the segmenter is small; defects are connected components, with a minimum area and a 10% overlap rule.'),
(C, 'import os, sys, json, warnings, importlib\nPROJECT_ROOT = os.path.abspath("..")\nif not os.path.isdir(os.path.join(PROJECT_ROOT, "notebooks")):\n    PROJECT_ROOT = os.getcwd()\nsys.path.insert(0, PROJECT_ROOT)\n# Prefer a configured data root only when it actually contains the paired folders.\ndef _has_paired_data(root):\n    if not root:\n        return False\n    return ((os.path.isdir(os.path.join(root, "train", "train", "GT")) and\n             os.path.isdir(os.path.join(root, "train", "train", "NoisyLR"))) or\n            (os.path.isdir(os.path.join(root, "GT")) and\n             os.path.isdir(os.path.join(root, "NoisyLR"))))\nif not _has_paired_data(os.environ.get("SEMICON_DATA")):\n    os.environ["SEMICON_DATA"] = PROJECT_ROOT\nwarnings.filterwarnings("ignore")\nimport numpy as np, pandas as pd, torch\nimport matplotlib.pyplot as plt\nimport config as C\nC = importlib.reload(C)\n# Print explicit results and data roots below\nfrom utils import carinthia as CA, downstream as DS, stats as S, figures as FG, metrics as MT\nfrom utils.degrade import degrade, fit_degradation\nfrom utils.analysis import finished_seeds, load_run_model, restore\nRUNS = os.path.abspath(os.environ.get("SEMICON_RUNS", os.path.join(PROJECT_ROOT, "results_kaggle")))\nDATA = os.path.abspath(os.environ.get("SEMICON_DATA", PROJECT_ROOT))\nC.OUT = RUNS\nC.RUNS = os.path.join(RUNS, "runs")\nC.TABLES = os.path.join(RUNS, "tables")\nC.FIGS = os.path.join(RUNS, "figures")\nC.DATA_ROOT = DATA\nprint("RUNS =", RUNS)\nprint("DATA =", DATA)\nif not os.path.isfile(os.path.join(RUNS, "split_manifest.json")):\n    raise FileNotFoundError(f"Kaggle results split manifest not found under RUNS={RUNS}")\nsplits, man = C.get_splits()\nfit_path = os.path.join(C.TABLES, "degradation_fit.json")\nfit = json.load(open(fit_path)) if os.path.exists(fit_path) else fit_degradation(splits["train"])\nLOOKS, SIGMA = fit["looks"], fit["sigma"]\ndev = C.device()\nCARINTHIA_DATA = os.path.abspath(os.environ.get("SEMICON_CARINTHIA", os.path.join(DATA, "data", "data", "data")))\nC.CARINTHIA_ROOT = CARINTHIA_DATA\nprint("CARINTHIA_DATA =", CARINTHIA_DATA)\nroot = CARINTHIA_DATA\nif C.SMOKE:\n    root = os.path.join(C.OUT, "synthetic_carinthia")\n    CA.make_synthetic_carinthia(root, n=24, size=64)\nRUN_DOWNSTREAM = os.path.isdir(root)\nif not RUN_DOWNSTREAM:\n    print("Carinthia-S images and masks were not found at", root)\n    print("Downstream defect evaluation is skipped. Extract Carinthia-S and set SEMICON_CARINTHIA to its dataset root to run it.")\n    items, tr_i, te_i = [], [], []\nelse:\n    SIZE = 64 if C.SMOKE else 256\n    items = CA.load_pairs(root, n=200 if C.MODE == "full" else 60, seed=0, size=SIZE, require_masks=True,\n                          manifest_path=os.path.join(C.TABLES, "carinthia_downstream_manifest.json"))\n    rng = np.random.RandomState(0)\n    order = rng.permutation(len(items)); ntr = int(0.7 * len(items))\n    tr_i, te_i = [items[i] for i in order[:ntr]], [items[i] for i in order[ntr:]]\n    print(f"{len(tr_i)} train / {len(te_i)} test images with masks; defect pixel rate {np.mean([i[\'mask\'].mean() for i in items]):.4f}")\n    T = lambda lst, k: torch.from_numpy(np.stack([i[k] for i in lst]).astype(np.float32))[:, None]'),
(C, 'if RUN_DOWNSTREAM:\n    seg = DS.train_segmenter(T(tr_i, "image"), T(tr_i, "mask"), epochs={"smoke": 3, "pilot": 25, "full": 60}[C.MODE], seed=0, device=dev)\n    clean = T(te_i, "image"); masks = T(te_i, "mask")\n    up = lambda a: torch.from_numpy(np.kron(a, np.ones((2, 2), np.float32)))[None, None]\n    names = [c for c in ["baseline", "film_control", "full", "NAFNet-small", "Restormer-small", "SwinIR-small"] if finished_seeds(C.RUNS, c)]\n    seeds_by = {c: finished_seeds(C.RUNS, c) for c in names}; print(seeds_by)\nelse:\n    names, seeds_by = [], {}\n    print("Segmenter training skipped because Carinthia-S is unavailable.")'),
(C, 'if RUN_DOWNSTREAM:\n    arms = {}\n    noise_draws = (0,) if C.MODE == "smoke" else (0, 1, 2)\n    def rows_to_df(rows, arm, seed, nd):\n        d = pd.DataFrame(rows); d["arm"], d["seed"], d["noise"], d["img"] = arm, seed, nd, np.arange(len(d)); return d\n    all_rows = [rows_to_df(DS.evaluate_arm(seg, clean, clean, clean[:, :, ::2, ::2], masks, dev), "clean (upper bound)", 0, 0)]\n    for nd in noise_draws:\n        deg = torch.from_numpy(np.stack([degrade(clean[k, 0].numpy(), LOOKS, SIGMA, seed=100 * nd + k) for k in range(len(clean))]))[:, None]\n        all_rows.append(rows_to_df(DS.evaluate_arm(seg, torch.cat([up(deg[k, 0].numpy()) for k in range(len(deg))]).clamp(0, 1), clean, deg, masks, dev), "bilinear (lower bound)", 0, nd))\n        for c in names:\n            for sd in seeds_by[c]:\n                model, cfg, _ = load_run_model(C.RUNS, c, sd, dev)\n                with torch.no_grad(): pred = restore(model, cfg, deg.to(dev)).clamp(0, 1).cpu()\n                all_rows.append(rows_to_df(DS.evaluate_arm(seg, pred, clean, deg, masks, dev), c, sd, nd))\n    D = pd.concat(all_rows, ignore_index=True); D.to_csv(os.path.join(C.TABLES, "downstream_long.csv"), index=False)\n    print(D.shape, "rows")\nelse:\n    print("Downstream metric computation skipped because Carinthia-S is unavailable.")'),
(C, 'if RUN_DOWNSTREAM:\n    cols = ["pix_f1", "inst_recall", "inst_precision", "false_alarms_per_img", "defect_cnr_ratio_to_clean", "edge_corr", "hp_err", "flat_err", "psnr"]\n    per = D.groupby(["arm", "img"])[cols].mean().reset_index()\n    rows = []\n    for a, g in per.groupby("arm"):\n        r = {"arm": a, "n_test_images": len(g)}\n        for c in cols:\n            v = g[c].dropna().values; lo, hi = S.bootstrap_ci(v, 2000) if len(v) > 1 else (np.nan, np.nan)\n            r[c] = f"{v.mean():.3f} [{lo:.3f},{hi:.3f}]"\n        rows.append(r)\n    tabD = pd.DataFrame(rows).set_index("arm").loc[[a for a in ["clean (upper bound)", "bilinear (lower bound)"] + names if a in per.arm.unique()]]\n    print(tabD.T.to_string()); tabD.to_csv(os.path.join(C.TABLES, "table_downstream.csv"))\nelse:\n    print("Downstream summary skipped because Carinthia-S is unavailable.")'),
(C, 'if RUN_DOWNSTREAM:\n    out = []\n    for m, hb in [("pix_f1", True), ("inst_recall", True), ("inst_precision", True), ("false_alarms_per_img", False), ("defect_cnr_ratio_to_clean", True), ("edge_corr", True)]:\n        P = {a: per[per.arm == a].sort_values("img")[m].fillna(0).values for a in per.arm.unique()}\n        for ref in [r for r in ["bilinear (lower bound)", "baseline"] if r in P]:\n            t = S.compare_to_reference(P, ref, [a for a in P if a != "clean (upper bound)"], higher_better=hb); t["metric"] = m; out.append(t)\n    tt = pd.concat(out, ignore_index=True) if out else pd.DataFrame()\n    if not tt.empty:\n        tt["p_holm_all"] = S.holm(tt.p_wilcoxon.values)\n        print(tt[["metric", "config", "vs", "mean_diff", "ci95_lo", "ci95_hi", "p_holm_all", "cohens_dz"]].round(4).to_string(index=False))\n        tt.to_csv(os.path.join(C.TABLES, "table_downstream_paired.csv"), index=False)\n        if "full" in names and "baseline" in names:\n            for m in ["inst_recall", "false_alarms_per_img", "pix_f1"]:\n                r = tt[(tt.metric == m) & (tt.config == "full") & (tt.vs == "baseline")]\n                if len(r):\n                    r = r.iloc[0]\n                    print(f"downstream claim \'full beats baseline\' on {m}: {\'SUPPORTED\' if (r.mean_diff>0 and r.ci95_lo>0 and r.p_holm_all<0.05) else \'NOT SUPPORTED\'} (diff {r.mean_diff:+.4f}, CI [{r.ci95_lo:+.4f},{r.ci95_hi:+.4f}])")\n    else:\n        print("No paired comparisons are available.")\nelse:\n    print("Downstream paired tests skipped because Carinthia-S is unavailable.")'),
(C, 'if RUN_DOWNSTREAM:\n    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4)); arms_o = list(tabD.index)\n    for a_, (m, title) in zip(ax, [("inst_recall", "Defect recall"), ("inst_precision", "Defect precision"), ("false_alarms_per_img", "False alarms / image")]):\n        v = [per[per.arm == a][m].mean() for a in arms_o]; e = [S.bootstrap_ci(per[per.arm == a][m].dropna().values, 1000) for a in arms_o]\n        a_.bar(range(len(arms_o)), v, yerr=[[x - l for x, (l, h) in zip(v, e)], [h - x for x, (l, h) in zip(v, e)]], capsize=3); a_.set_title(title)\n        a_.set_xticks(range(len(arms_o))); a_.set_xticklabels([x.split(" (")[0] for x in arms_o], rotation=35, ha="right")\n    FG.save(fig, C.FIGS, "fig_downstream", {"dataset": "Carinthia-S", "n_test": len(te_i), "looks": LOOKS, "sigma": SIGMA, "error_bars": "95% bootstrap CI over images"})\nelse:\n    print("Downstream figure skipped because Carinthia-S is unavailable.")'),
]

# ======================================================================================= 08
NB["08_Efficiency_Benchmark.ipynb"] = [
(M, r"""# 08 - Efficiency Benchmark
Parameters, **FLOPs counted by `torch.utils.flop_counter`** (replaces the hand-written conv-only counter), median / p95 latency with warm-up, CUDA-event timing on GPU, batch 1 and 16, FP32 and FP16.

Internal-consistency checks are **asserted**: throughput must equal `batch * 1000 / median latency` (the submitted Table II listed a 16.22 ms latency next to a throughput that corresponds to 18.22 ms)."""),
(C, HEAD.replace("C.banner()", "# Print explicit results and data roots below") + r"""
from utils.experiment import build_model, CONFIGS
from utils.analysis import count_flops, latency_ms
dev = C.device()
RUNS = os.path.join(PROJECT_ROOT, "results_kaggle")
DATA = PROJECT_ROOT
C.OUT = RUNS
C.RUNS = os.path.join(RUNS, "runs")
C.TABLES = os.path.join(RUNS, "tables")
C.FIGS = os.path.join(RUNS, "figures")
C.DATA_ROOT = DATA
print("RUNS =", RUNS)
print("DATA =", DATA)
names = ["baseline", "film_control", "full", "NAFNet-small", "Restormer-small", "SwinIR-small", "NAFNet-base", "Restormer-base", "SwinIR-base"]
n_run, n_warm = (3, 1) if C.MODE == "smoke" else ((30, 5) if dev == "cuda" else (5, 1))
H = 64 if C.MODE == "smoke" else 128
print(f"hardware={dev}; timing repetitions={n_run}, warmups={n_warm} (CPU defaults are shortened to keep the benchmark responsive)")
rows = []
for c in names:
    print(f"Benchmarking {c}...", flush=True)
    m = build_model(CONFIGS[c]); p = sum(x.numel() for x in m.parameters())
    r = dict(config=c, params=p, GFLOPs=round(count_flops(m, (1, 1, H, H)), 3))
    for b in (1, 16):
        cpu_runs = max(5, n_run // 5) if C.MODE != "smoke" and b == 16 else max(2, n_run // (5 if b == 16 else 1))
        x = latency_ms(m, (b, 1, H, H), "cpu", n_warm=n_warm, n_run=cpu_runs)
        assert abs(x["throughput_img_s"] - b * 1000 / x["median_ms"]) < 1e-6
        r[f"cpu_b{b}_ms"], r[f"cpu_b{b}_img_s"] = round(x["median_ms"], 2), round(x["throughput_img_s"], 1)
        if dev == "cuda":
            for half in (False, True):
                g = latency_ms(m, (b, 1, H, H), "cuda", half=half, n_warm=n_warm, n_run=n_run)
                r[f"gpu_b{b}_{'fp16' if half else 'fp32'}_ms"], r[f"gpu_b{b}_{'fp16' if half else 'fp32'}_img_s"] = round(g["median_ms"], 2), round(g["throughput_img_s"], 1)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats(); m.cuda().eval()(torch.randn(1, 1, H, H, device="cuda")); r["gpu_peak_MB"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(C.TABLES, "table_efficiency.csv"), index=False)
    print(f"Finished {c}.", flush=True)
bench = pd.DataFrame(rows); print(bench.to_string(index=False)); bench.to_csv(os.path.join(C.TABLES, "table_efficiency.csv"), index=False)
print("\nhardware:", torch.cuda.get_device_name(0) if dev == "cuda" else "CPU only", "| input", (1, 1, H, H), "| median over", n_run, "runs after", n_warm, "warm-ups")"""),
]

# ======================================================================================= 09
NB["09_Figures_and_Results_Summary.ipynb"] = [
(M, r"""# 09 - Figures and Results Summary
**Addresses comment 6 (figures/tables/consistency).**  All figures are regenerated from the *final* checkpoints of this protocol, at 300 dpi (PNG + vector PDF), with a **provenance JSON** next to each (checkpoint, split, image id, seed, the exact numbers shown). Qualitative examples are chosen by a **fixed rule, not by eye**: the median-PSNR and the worst-PSNR *test* image of the baseline, seed = first finished seed.

The last cell writes `RESULTS_SUMMARY.md` - every number in the manuscript must be copied from there (or the CSV/JSON files it lists), never typed."""),
(C, HEAD.replace("C.banner()", "# Print explicit results and data roots below") + r"""
from utils.data import load_npy
from utils.experiment import load_results, load_per_image
from utils.analysis import finished_seeds, load_run_model, restore
from utils import figures as FG, metrics as MT
RUNS = os.path.join(PROJECT_ROOT, "results_kaggle")
DATA = PROJECT_ROOT
C.OUT = RUNS
C.RUNS = os.path.join(RUNS, "runs")
C.TABLES = os.path.join(RUNS, "tables")
C.FIGS = os.path.join(RUNS, "figures")
C.DATA_ROOT = DATA
print("RUNS =", RUNS)
print("DATA =", DATA)
if not os.path.isfile(os.path.join(RUNS, "split_manifest.json")):
    raise FileNotFoundError(f"Kaggle results split manifest not found under RUNS={RUNS}")
splits, man = C.get_splits(); dev = C.device()
have = [c for c in ["baseline", "global_gc", "film_control", "ldmh_loss_only", "full", "NAFNet-small", "Restormer-small", "SwinIR-small"] if finished_seeds(C.RUNS, c)]
sd = {c: finished_seeds(C.RUNS, c)[0] for c in have}; print("using seed", sd)
missing = [c for c in ["baseline", "full"] if c not in sd]
if missing:
    raise FileNotFoundError(f"Required runs missing from {C.RUNS}: {missing}. Train them in results_kaggle first.")"""),
(C, r"""# Fig: training curves + component health of the SAME checkpoint that produced the tables
h = os.path.join(C.RUNS, "full", f"seed{sd['full']}", "history.csv")
FG.training_curves(h, C.FIGS, "fig_training_curves_full", "Full model (seed %d)" % sd["full"],
                   {"history": h, "checkpoint": h.replace("history.csv", "best.pth"), "note": "same run as Table (seed %d)" % sd["full"]})
FG.component_health(h, C.FIGS, "fig_component_health_full")
hb = os.path.join(C.RUNS, "baseline", f"seed{sd['baseline']}", "history.csv"); FG.training_curves(hb, C.FIGS, "fig_training_curves_baseline", "Baseline (seed %d)" % sd["baseline"])
print("saved training / component-health figures")"""),
(C, r"""# Qualitative: fixed selection rule
test_gt = [load_npy(g) for g, _ in splits["test"]]; test_ny = [load_npy(n) for _, n in splits["test"]]
pi = load_per_image(C.RUNS, "baseline", "psnr").mean(0); order = np.argsort(pi)
picks = {"median": int(order[len(order) // 2]), "worst": int(order[0])}
for tag, k in picks.items():
    x = torch.from_numpy(test_ny[k])[None, None].to(dev)
    gt = test_gt[k]; arrs = [("Clean GT", gt), ("Input (bilinear x2)", np.clip(np.kron(test_ny[k], np.ones((2, 2))), 0, 1))]
    mix = None
    for c in have:
        m, cfg, pth = load_run_model(C.RUNS, c, sd[c], dev)
        with torch.no_grad(): out = m(x)
        pr = (out[0] if isinstance(out, tuple) else out).clamp(0, 1)[0, 0].cpu().numpy()
        arrs.append((f"{c}\n{MT.psnr(torch.from_numpy(pr)[None,None], torch.from_numpy(gt)[None,None]).item():.2f} dB", pr))
        if c == "full": mix = (out[1][0].cpu().numpy(), out[2].cpu().numpy(), pth)
    name_ = os.path.basename(splits["test"][k][0])
    FG.qualitative_grid(arrs, C.FIGS, f"fig_qualitative_{tag}", {"split": "test", "image": name_, "rule": f"{tag} PSNR of baseline", "seeds": sd})
    if mix is not None:
        FG.mixture_maps(test_ny[k], mix[0], mix[1], C.FIGS, f"fig_mixture_maps_{tag}", {"split": "test", "image": name_, "checkpoint": mix[2], "seed": sd["full"]})
    print(tag, "->", name_)"""),
(C, r"""# provenance audit: every figure must have a JSON sidecar
import glob
figs = sorted(glob.glob(os.path.join(C.FIGS, "*.png")))
audit = [(os.path.basename(f), os.path.exists(f[:-4] + ".json")) for f in figs]
print(pd.DataFrame(audit, columns=["figure", "has_provenance_json"]).to_string(index=False))
assert all(a for _, a in audit), "figure without provenance sidecar"
for f in sorted(glob.glob(os.path.join(C.FIGS, "fig_mixture_maps_*.json"))):
    j = json.load(open(f)); print(os.path.basename(f), "betas:", [round(b, 3) for b in j["betas"]], "dominant fractions:", [round(x, 3) for x in j["dominant_fraction"]])
print("\nThe betas and dominant fractions printed above are the ONLY values that may be quoted for the mixture-map figure.")"""),
(C, r"""# RESULTS_SUMMARY.md : one file to copy numbers from
lines = ["# Results summary (auto-generated; copy numbers from here)\n", f"mode={C.MODE} epochs={C.EPOCHS} seeds={C.SEEDS}\n"]
for f in sorted(glob.glob(os.path.join(C.TABLES, "*.csv"))):
    if "long" in f or "per_image" in f or "all_runs" in f: continue
    d = pd.read_csv(f).round(4)
    try: body = d.to_markdown(index=False)
    except Exception: body = "```\n" + d.to_string(index=False) + "\n```"
    lines += [f"\n## {os.path.basename(f)}\n", body, "\n"]
for f in sorted(glob.glob(os.path.join(C.TABLES, "*.json"))):
    lines += [f"\n## {os.path.basename(f)}\n", "```", json.dumps(json.load(open(f)), indent=1, default=str)[:3000], "```\n"]
open(os.path.join(C.OUT, "RESULTS_SUMMARY.md"), "w").write("\n".join(lines)); print("wrote", os.path.join(C.OUT, "RESULTS_SUMMARY.md"))"""),
]
