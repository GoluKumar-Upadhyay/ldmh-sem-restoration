"""Notebook definitions 04-06."""
M, C = "md", "code"
from defs_a import HEAD

NB = {}

# ======================================================================================= 04
NB["04_Ablation_and_Statistics.ipynb"] = [
(M, r"""# 04 - Full-scale Ablation, Controls and Statistical Tests
**Addresses comments 1 and 3 (and 4 for HP-Err / edge metrics).**

Everything is read from the multi-seed runs of notebook 03 - **same protocol, same epochs, same split** for every row (the submitted ablation used 200 images, 3 epochs and no seeds).

Two levels of statistics, both reported:
* **seed level** - mean +- std over independent training runs;
* **image level** - paired comparison on the *same test images* (seed-averaged per-image metric): paired t, Wilcoxon, paired-bootstrap 95% CI, Cohen's d_z, **Holm-corrected** across all comparisons.

Holm correction is applied within each (metric, reference) family of comparisons. Claims are evaluated by a **pre-registered rule** (below) and the notebook prints SUPPORTED / NOT SUPPORTED - the manuscript must follow it."""),
(C, HEAD.replace("C.banner()", "# Print explicit results and data roots below") + r"""
from utils.experiment import load_results, load_per_image, ABLATION, LADDER
from utils import stats as S, figures as FG
RUNS = os.path.join(PROJECT_ROOT, "results_kaggle")
DATA = PROJECT_ROOT
RUNS_DIR = os.path.join(RUNS, "runs")
TABLES_DIR = os.path.join(RUNS, "tables")
FIGS_DIR = os.path.join(RUNS, "figures")
print("RUNS =", RUNS)
print("DATA =", DATA)
if not os.path.isdir(os.path.join(DATA, "train", "train", "GT")) or not os.path.isdir(os.path.join(DATA, "train", "train", "NoisyLR")):
    raise FileNotFoundError(f"Paired training data not found under DATA={DATA}")
res = load_results(RUNS_DIR)
if res.empty:
    raise FileNotFoundError(f"No completed training results found under RUNS={RUNS_DIR}. Check runs/<model>/seed*/results.json.")
have = [c for c in ABLATION if c in set(res.config)]
if not have:
    raise ValueError(f"No ablation results found under RUNS={RUNS_DIR}; available columns: {list(res.columns)}")
print("configs found:", have); print(res.groupby("config").seed.apply(list).to_string())"""),
(M, "## 1. Main table: mean +- std over seeds (test split)"),
(C, r"""metrics = [("test_psnr", "PSNR", True), ("test_ssim", "SSIM", True), ("test_hp_err", "HP-Err (top-5%)", False),
           ("test_edge_corr", "Edge corr", True), ("test_edge_f1", "Edge F1", True), ("test_flat_err", "Flat-region err", False)]
blocks = {}
for col, nm, hb in metrics:
    blocks[nm] = S.summarize_configs(res, col, have).set_index("config")
tab = pd.DataFrame({nm: [f"{b.loc[c,'mean']:.4f} +- {b.loc[c,'std']:.4f}" for c in have] for nm, b in blocks.items()}, index=have)
tab.insert(0, "params", [int(res[res.config == c].params.iloc[0]) for c in have]); tab.insert(1, "n_seeds", [int((res.config == c).sum()) for c in have])
print(tab.to_string()); tab.to_csv(os.path.join(TABLES_DIR, "table_ablation_main.csv"))"""),
(M, "## 2. Paired image-level tests (Holm-corrected)\nReference rows: `baseline` (is anything better than the plain backbone?) and **`film_control`** (is the gain due to *distribution modelling* or merely to the extra encoder/FiLM capacity?)."""),
(C, r"""per_image = {m: {c: load_per_image(RUNS_DIR, c, m) for c in have} for m in ["psnr", "ssim", "hp_err", "edge_corr", "flat_err"]}
tests = []
for m, hb in [("psnr", True), ("ssim", True), ("hp_err", False), ("edge_corr", True), ("flat_err", False)]:
    for ref in ["baseline", "film_control"]:
        if ref in have:
            t = S.compare_to_reference(per_image[m], ref, have, higher_better=hb); t["metric"] = m; tests.append(t)
tests = pd.concat(tests, ignore_index=True)
show = ["metric", "config", "vs", "mean_diff", "ci95_lo", "ci95_hi", "p_wilcoxon_holm", "cohens_dz", "frac_images_improved", "significant_0.05"]
print("(mean_diff > 0 always means 'config better than reference')")
print(tests[show].round(4).to_string(index=False)); tests.to_csv(os.path.join(TABLES_DIR, "table_ablation_paired_tests.csv"), index=False)"""),
(M, "## 3. Pre-registered claim checks\nSupported only if `mean_diff > 0`, the paired-bootstrap CI excludes 0 **and** Holm-adjusted Wilcoxon p < 0.05."),
(C, r"""def verdict(cfg, ref, metric="psnr"):
    r = tests[(tests.config == cfg) & (tests.vs == ref) & (tests.metric == metric)]
    if r.empty: return "n/a (config or reference not run)"
    r = r.iloc[0]; ok = r.mean_diff > 0 and r.ci95_lo > 0 and r.p_wilcoxon_holm < 0.05
    return f"{'SUPPORTED' if ok else 'NOT SUPPORTED'}  (diff={r.mean_diff:+.4f}, CI [{r.ci95_lo:+.4f},{r.ci95_hi:+.4f}], p_holm={r.p_wilcoxon_holm:.3g})"
claims = {
 "C1  Full > Baseline (PSNR)": verdict("full", "baseline"),
 "C2  Full > FiLM-control  (gain is due to distribution modelling, not capacity)": verdict("full", "film_control"),
 "C3  Full > Global-GC (PSNR)": verdict("full", "global_gc") if "global_gc" in have else "n/a",
 "C4  LDMH-loss-only differs from Baseline? (should be NOT SUPPORTED: backbone unaffected)": verdict("ldmh_loss_only", "baseline"),
 "C5  Full > Baseline (HP-Err, top-5% pixels; lower is better)": verdict("full", "baseline", "hp_err"),
}
for k, v in claims.items(): print(f"{k}\n      -> {v}")
means = {c: blocks['PSNR'].loc[c, 'mean'] for c in have}
order = sorted(means, key=means.get); print("\nobserved PSNR order (low->high):", " < ".join(order))
print("paper's 'monotonic Baseline < Global-GC < LDMH < Full' claim holds in means:", order.index("baseline") < order.index("global_gc") < order.index("ldmh_loss_only") < order.index("full") if all(c in order for c in ["baseline","global_gc","ldmh_loss_only","full"]) else "n/a")
json.dump(claims, open(os.path.join(TABLES_DIR, "claim_checks.json"), "w"), indent=1)"""),
(C, r"""# effect vs. noise floor: seed-to-seed std of each config, and the baseline-vs-ldmh_loss_only gap (functionally identical models)
nf = {c: S.noise_floor(per_image["psnr"][c], None)["std_over_seeds"] for c in have}
print("seed-to-seed std of mean test PSNR:", {c: round(v, 4) for c, v in nf.items()})
if all(c in have for c in ["baseline", "ldmh_loss_only"]):
    g = abs(means["ldmh_loss_only"] - means["baseline"]); print(f"|ldmh_loss_only - baseline| = {g:.4f} dB  <- an upper bound on pure initialisation/seed effect for IDENTICAL functions")
if "full" in have and "baseline" in have:
    print(f"|full - baseline| = {abs(means['full']-means['baseline']):.4f} dB  vs. noise-floor std ~ {max(nf['baseline'], nf['full']):.4f} dB")"""),
(M, "## 4. Does the 'collapse fix' matter?  Component health per run\n`full_clamp` reproduces the original parameterisation; `full_no_penalty` the loss used in the old Table V. A fix is only demonstrated if the clamp variant actually collapses *and* this changes restoration quality (it cannot change the forward pass directly: beta/alpha only enter the auxiliary NLL)."),
(C, r"""import glob
rows = []
for cfg in [c for c in ["full", "full_clamp", "full_no_penalty", "full_no_bal", "full_no_div", "ldmh_loss_only"] if c in have]:
    for p in sorted(glob.glob(os.path.join(RUNS_DIR, cfg, "seed*", "history.csv"))):
        h = pd.read_csv(p).iloc[-1]; b = [h.beta0, h.beta1, h.beta2]; u = [h.usage0, h.usage1, h.usage2]
        rows.append(dict(config=cfg, seed=int(p.split("seed")[-1].split(os.sep)[0]), betas=[round(x, 3) for x in b], usage=[round(x, 3) for x in u],
                         min_beta_dist=round(h.min_beta_dist, 3), any_beta_at_bound=bool(h.any_beta_at_bound), min_usage=round(min(u), 3),
                         all_betas_gt_1=all(x > 1 for x in b), heavy_tailed_component=any(x < 1 for x in b)))
ch = pd.DataFrame(rows); print(ch.to_string(index=False)); ch.to_csv(os.path.join(TABLES_DIR, "table_component_health.csv"), index=False)
if len(ch):
    print("\nAny learned component with beta < 1 (heavy-tailed, as Section IV motivates)?", bool(ch.heavy_tailed_component.any()))
    print("Any run with a beta pinned at a bound (sigmoid/clamp saturation)?", bool(ch.any_beta_at_bound.any()))"""),
(M, "## 5. Figures and LaTeX"),
(C, r"""for col, nm, hb in metrics[:3]:
    s = S.summarize_configs(res, col, have)
    FG.bars_with_error(s, [col], f"{nm} (test, mean +- std over seeds)", FIGS_DIR, f"fig_ablation_{col}", ylabel=nm)
tex = S.latex_table(pd.DataFrame({"Config": have, **{nm: [blocks[nm].loc[c, 'mean'] for c in have] for nm in ["PSNR", "SSIM", "HP-Err (top-5%)"]}}),
                    ["Config", "PSNR", "SSIM", "HP-Err (top-5%)"], "{:.4f}", "Full-scale ablation (test split, mean over seeds).", "tab:ablation")
open(os.path.join(TABLES_DIR, "table_ablation.tex"), "w").write(tex); print(tex)"""),
]

# ======================================================================================= 05
NB["05_External_Baselines.ipynb"] = [
(M, r"""# 05 - Comparison with External Restoration Architectures
**Addresses comment 5.**  NAFNet, Restormer and SwinIR (compact re-implementations, see `models/baselines.py`) are trained from scratch with **the same split, augmentation, epochs, optimiser, schedule, early-stopping rule and multi-seed protocol** as the proposed model (notebook 03). `small` variants match the proposed model's parameter budget (~0.1 M); `base` variants are larger references. Parameters and FLOPs are always reported.

**State in the paper:** these are *re-implementations written for this repository* and not the official code; for a camera-ready table, add the official implementations through `BASELINE_REGISTRY` (same call signature) and report both. Hyper-parameter selection was identical for every method (none beyond the shared protocol)."""),
(C, HEAD.replace("C.banner()", "# Print explicit results and data roots below") + r"""
from utils.experiment import load_results, load_per_image, EXTERNAL, build_model, CONFIGS
from utils.analysis import count_flops
from utils import stats as S, figures as FG
RUNS = os.path.join(PROJECT_ROOT, "results_kaggle")
DATA = PROJECT_ROOT
RUNS_DIR = os.path.join(RUNS, "runs")
TABLES_DIR = os.path.join(RUNS, "tables")
FIGS_DIR = os.path.join(RUNS, "figures")
print("RUNS =", RUNS)
print("DATA =", DATA)
if not os.path.isdir(os.path.join(DATA, "train", "train", "GT")) or not os.path.isdir(os.path.join(DATA, "train", "train", "NoisyLR")):
    raise FileNotFoundError(f"Paired training data not found under DATA={DATA}")
res = load_results(RUNS_DIR)
if res.empty: raise FileNotFoundError(f"No completed runs found under RUNS={RUNS_DIR}")
have = [c for c in EXTERNAL if c in set(res.config)]
print("External configs found:", have)
if not have:
    print("No external baselines are in results_kaggle yet.")
    print(rf'To train them into this folder, run in PowerShell: $env:SEMICON_RUNS = "{RUNS}"; python scripts/run_experiments.py --mode pilot --groups external --seeds 0,1')"""),
(C, r"""rows = []
for c in have:
    g = res[res.config == c]; m = build_model(CONFIGS[c])
    rows.append(dict(config=c, params=int(g.params.iloc[0]), GFLOPs_128=round(count_flops(m, (1, 1, 128, 128)), 2), n_seeds=len(g),
                     PSNR=f"{g.test_psnr.mean():.3f} +- {g.test_psnr.std(ddof=1) if len(g)>1 else float('nan'):.3f}",
                     SSIM=f"{g.test_ssim.mean():.4f} +- {g.test_ssim.std(ddof=1) if len(g)>1 else float('nan'):.4f}",
                     HP_Err=f"{g.test_hp_err.mean():.4f}", edge_corr=f"{g.test_edge_corr.mean():.4f}", epochs_run=int(g.best_epoch.max())))
ext = pd.DataFrame(rows, columns=["config", "params", "GFLOPs_128", "n_seeds", "PSNR", "SSIM", "HP_Err", "edge_corr", "epochs_run"])
print(ext.to_string(index=False) if have else "No external model results to summarize yet.")
ext.to_csv(os.path.join(TABLES_DIR, "table_external_main.csv"), index=False)"""),
(C, r"""pi = {c: load_per_image(RUNS_DIR, c, "psnr") for c in have}
out = []
for ref in [r for r in ["full", "baseline"] if r in have]:
    t = S.compare_to_reference(pi, ref, have); t["metric"] = "psnr"; out.append(t)
pt = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["config", "vs", "mean_diff", "ci95_lo", "ci95_hi", "p_wilcoxon_holm", "cohens_dz", "significant_0.05", "metric"])
if not pt.empty:
    print("(mean_diff > 0 = config better than reference)")
    print(pt[["config", "vs", "mean_diff", "ci95_lo", "ci95_hi", "p_wilcoxon_holm", "cohens_dz", "significant_0.05"]].round(4).to_string(index=False))
else: print("No external paired tests available because external runs have not been trained.")
pt.to_csv(os.path.join(TABLES_DIR, "table_external_paired_tests.csv"), index=False)"""),
(C, r"""# Honest reading rule: is the proposed model competitive at comparable size?
if "full" in have:
    small = [c for c in have if c.endswith("-small")]
    for c in small:
        r = pt[(pt.config == c) & (pt.vs == "full")]
        if len(r):
            r = r.iloc[0]; better = "BETTER than" if (r.mean_diff > 0 and r.ci95_lo > 0) else ("WORSE than" if (r.mean_diff < 0 and r.ci95_hi < 0) else "statistically INDISTINGUISHABLE from")
            print(f"{c:18s} is {better} the proposed model (diff {r.mean_diff:+.4f} dB, CI [{r.ci95_lo:+.4f}, {r.ci95_hi:+.4f}])")
    print("\nIf an external model is better or indistinguishable at equal size, the paper must say the mechanism does not provide a demonstrated advantage over the choice of backbone.")
if have:
    sm = S.summarize_configs(res, "test_psnr", have)
    FG.bars_with_error(sm, ["test_psnr"], "External comparison - test PSNR (mean +- std over seeds)", FIGS_DIR, "fig_external_psnr", ylabel="PSNR (dB)")
else:
    print("External comparison figure skipped until external runs are available.")"""),
]

# ======================================================================================= 06
NB["06_Robustness_and_OOD.ipynb"] = [
(M, r"""# 06 - Per-degradation Robustness and Out-of-Distribution Evaluation
**Addresses comments 1 and 2.**

**Robustness (replaces Table VI).** The submitted test used `list_npy(TRAIN_GT_DIR)[:30]` - the first 30 files of the *training* folder, ~90% of which were training images - one noise seed per image, and hard-coded degradation strengths. Here: **test-split images only**, degradation strengths **estimated from the training pairs** (notebook 00), 3 independent noise draws per image, all seeds of every model, paired statistics.

**OOD (replaces Table VII / Fig. 10).** The submitted run read the first ten files of `data.zip`, which were `data/masks/*.png` - *segmentation masks, not SEM images*. The loader below separates images from masks, **refuses label-map-like files**, and writes a **manifest** (file names, SHA-256 prefixes, original shapes, selection seed) that must be quoted in the paper together with the dataset DOI/version."""),
(C, HEAD.replace("C.banner()", "# Print explicit results and data roots below") + r"""
from utils.data import load_npy
from utils.experiment import load_per_image, LADDER
from utils.analysis import robustness_grid, finished_seeds, load_run_model, restore
from utils.degrade import degrade, fit_degradation
from utils import stats as S, figures as FG, metrics as MT
from utils import carinthia as CA
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
splits, man = C.get_splits()
fp = os.path.join(C.TABLES, "degradation_fit.json")
fit = json.load(open(fp)) if os.path.exists(fp) else fit_degradation(splits["train"])
LOOKS, SIGMA = fit["looks"], fit["sigma"]; print(f"degradation used everywhere: looks={LOOKS:.2f}, sigma={SIGMA:.4f} (estimated from TRAIN pairs)")
names = [c for c in ["baseline", "global_gc", "film_control", "full", "NAFNet-small"] if finished_seeds(C.RUNS, c)]
seeds_by = {c: finished_seeds(C.RUNS, c) for c in names}; print(seeds_by)
dev = C.device()"""),
(M, "## 1. Robustness on the held-out TEST split"),
(C, r"""clean = np.stack([load_npy(g) for g, _ in splits["test"]]).astype(np.float32)
print("test images:", len(clean), "(none of them seen in training/validation)")
grid = robustness_grid(C.RUNS, names, seeds_by, clean, LOOKS, SIGMA, noise_seeds=(0, 1) if C.MODE == "smoke" else (0, 1, 2), device=dev)
grid.to_csv(os.path.join(C.TABLES, "robustness_long.csv"), index=False)
agg = grid.groupby(["config", "condition", "img"])[["psnr", "ssim"]].mean().reset_index()         # average over seeds and noise draws
tab = grid.groupby(["config", "condition", "seed"]).psnr.mean().reset_index().groupby(["config", "condition"]).psnr.agg(["mean", "std"]).round(3)
print(tab.unstack("condition").to_string())"""),
(C, r"""rows = []
for cond in agg.condition.unique():
    P = {c: agg[(agg.config == c) & (agg.condition == cond)].sort_values("img").psnr.values for c in names}
    for ref in [r for r in ["baseline", "film_control"] if r in names]:
        t = S.compare_to_reference(P, ref, names); t["condition"] = cond; rows.append(t)
rt = pd.concat(rows, ignore_index=True); rt["p_wilcoxon_holm_global"] = S.holm(rt.p_wilcoxon.values)
print(rt[["condition", "config", "vs", "mean_diff", "ci95_lo", "ci95_hi", "p_wilcoxon_holm_global", "cohens_dz"]].round(4).to_string(index=False))
rt.to_csv(os.path.join(C.TABLES, "table_robustness_paired_tests.csv"), index=False)
sp = rt[(rt.condition == "speckle only") & (rt.config == "full")]
for _, r in sp.iterrows():
    ok = r.mean_diff > 0 and r.ci95_lo > 0 and r.p_wilcoxon_holm_global < 0.05
    print(f"'speckle-only advantage' of full vs {r.vs}: {'SUPPORTED' if ok else 'NOT SUPPORTED'} (diff {r.mean_diff:+.4f} dB, CI [{r.ci95_lo:+.4f},{r.ci95_hi:+.4f}])")"""),
(C, r"""fig, ax = plt.subplots(figsize=(8, 3.6)); conds = list(agg.condition.unique()); w = 0.8 / len(names)
for k, c in enumerate(names):
    m, lo, hi = [], [], []
    for cond in conds:
        v = agg[(agg.config == c) & (agg.condition == cond)].psnr.values; m.append(v.mean()); l, h = S.bootstrap_ci(v, 2000); lo.append(v.mean() - l); hi.append(h - v.mean())
    ax.bar(np.arange(len(conds)) + k * w, m, w, yerr=[lo, hi], capsize=2, label=c)
ax.set_xticks(np.arange(len(conds)) + 0.4 - w / 2); ax.set_xticklabels(conds); ax.set_ylabel("PSNR (dB)"); ax.legend(ncol=3, fontsize=8)
ax.set_ylim(min(agg.psnr.mean() - 6, ax.get_ylim()[0]), None); ax.set_title("Per-degradation robustness - test split (95% bootstrap CI over images)")
FG.save(fig, C.FIGS, "fig_robustness", {"split": "test", "n_images": int(len(clean)), "looks": LOOKS, "sigma": SIGMA, "configs": names, "seeds": seeds_by})"""),
(M, "## 2. Out-of-distribution SEM images (Carinthia-S)\nDownload the dataset (Zenodo record 16895427 - confirm version/DOI on the record page), unzip, and set `SEMICON_CARINTHIA`. Quote in the paper: DOI + version, number of images, selection procedure (seeded random sample), the manifest file, and the degradation parameters."),
(C, r"""root = C.CARINTHIA_ROOT
if C.SMOKE:
    root = os.path.join(C.OUT, "synthetic_carinthia"); CA.make_synthetic_carinthia(root, n=24, size=64)
OOD = os.path.isdir(root); SIZE = 64 if C.SMOKE else 256
if not OOD:
    print("Carinthia-S folder not found:", root, "\n-> download it and set SEMICON_CARINTHIA; the OOD section is skipped. DO NOT substitute other images.")
else:
    items = CA.load_pairs(root, n=100 if C.MODE == "full" else 30, seed=0, size=SIZE, manifest_path=os.path.join(C.TABLES, "carinthia_manifest.json"))
    print(len(items), "verified SEM images; first files:", [i["name"] for i in items[:3]], "| manifest ->", os.path.join(C.TABLES, "carinthia_manifest.json"))
    assert not any(CA.looks_like_mask(np.round(i["image"] * 255)) for i in items), "a label map slipped through"
    clean_o = np.stack([i["image"] for i in items]).astype(np.float32)
    deg_o = np.stack([degrade(clean_o[k], LOOKS, SIGMA, seed=k) for k in range(len(clean_o))])
    xt, gt = torch.from_numpy(deg_o)[:, None].to(dev), torch.from_numpy(clean_o)[:, None].to(dev)
    rows = []
    for c in names:
        for sd in seeds_by[c]:
            model, cfg, _ = load_run_model(C.RUNS, c, sd, dev)
            p = restore(model, cfg, xt).clamp(0, 1)
            for k in range(len(items)):
                rows.append(dict(config=c, seed=sd, img=k, name=items[k]["name"], psnr=float(MT.psnr(p[k:k+1], gt[k:k+1])), ssim=float(MT.ssim(p[k:k+1], gt[k:k+1]))))
    od = pd.DataFrame(rows); od.to_csv(os.path.join(C.TABLES, "ood_per_image.csv"), index=False)
    s = od.groupby(["config", "img"])[["psnr", "ssim"]].mean().reset_index()
    out = []
    for c in names:
        v = s[s.config == c]; lo, hi = S.bootstrap_ci(v.psnr.values, 2000)
        out.append(dict(config=c, n_images=len(v), PSNR=round(v.psnr.mean(), 3), PSNR_ci=(round(lo, 3), round(hi, 3)), SSIM=round(v.ssim.mean(), 4), SSIM_std_over_images=round(v.ssim.std(), 4)))
    print(pd.DataFrame(out).to_string(index=False)); pd.DataFrame(out).to_csv(os.path.join(C.TABLES, "table_ood.csv"), index=False)
    P = {c: s[s.config == c].sort_values("img").psnr.values for c in names}
    t = S.compare_to_reference(P, "baseline", names); print(t[["config", "mean_diff", "ci95_lo", "ci95_hi", "p_wilcoxon_holm", "significant_0.05"]].round(4).to_string(index=False)); t.to_csv(os.path.join(C.TABLES, "table_ood_paired.csv"), index=False)"""),
(C, r"""if OOD:
    k0 = int(np.argsort(s[s.config == names[0]].psnr.values)[len(items) // 2])           # median image of the first model: fixed rule, not cherry-picked
    sd0 = seeds_by["full"][0] if "full" in names else seeds_by[names[0]][0]
    arrs = [("Clean (SEM)", clean_o[k0]), ("Degraded (bilinear x2)", np.clip(np.kron(deg_o[k0], np.ones((2, 2))) if deg_o.shape[-1] * 2 == clean_o.shape[-1] else deg_o[k0], 0, 1))]
    for c in names:
        model, cfg, _ = load_run_model(C.RUNS, c, seeds_by[c][0], dev)
        arrs.append((c, restore(model, cfg, xt[k0:k0+1]).clamp(0, 1)[0, 0].cpu().numpy()))
    FG.qualitative_grid(arrs, C.FIGS, "fig_ood_qualitative", {"dataset": "Carinthia-S", "file": items[k0]["name"], "sha": items[k0]["sha"], "selection": "median-PSNR image", "seed": seeds_by[names[0]][0], "looks": LOOKS, "sigma": SIGMA})
    print("saved fig_ood_qualitative for", items[k0]["name"])"""),
]
