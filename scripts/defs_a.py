"""Notebook definitions 00-03."""
M, C = "md", "code"

HEAD = r"""import os, sys, json, warnings, importlib
PROJECT_ROOT = os.path.abspath("..")
if not os.path.isdir(os.path.join(PROJECT_ROOT, "notebooks")):
    PROJECT_ROOT = os.getcwd()
sys.path.insert(0, PROJECT_ROOT)
# Prefer a configured data root only when it actually contains the paired folders.
def _has_paired_data(root):
    if not root:
        return False
    return ((os.path.isdir(os.path.join(root, "train", "train", "GT")) and
             os.path.isdir(os.path.join(root, "train", "train", "NoisyLR"))) or
            (os.path.isdir(os.path.join(root, "GT")) and
             os.path.isdir(os.path.join(root, "NoisyLR"))))
if not _has_paired_data(os.environ.get("SEMICON_DATA")):
    os.environ["SEMICON_DATA"] = PROJECT_ROOT
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
import config as C
C = importlib.reload(C)
C.banner()"""

NB = {}

# ======================================================================================= 00
NB["00_Setup_and_Data_Audit.ipynb"] = [
(M, r"""# 00 - Setup and Data Audit
**Addresses reviewer comments 1, 2, 4 (foundation).**  Nothing here trains a model.

1. Builds the **train / validation / test manifest** (80/10/10, fixed seed, saved as JSON). The validation split is used only for
   checkpoint selection; the test split is used once per run. (The submitted paper selected checkpoints, tuned hyper-parameters
   and reported results on the same 320 images.)
2. Audits the data: value ranges, the **definition of the reference residual** `r = y - D(x)`, and a contact sheet of random ground-truth
   images so the paper can state truthfully what the data look like.
3. **Estimates the degradation parameters (speckle looks, Gaussian sigma) from the training pairs** instead of hard-coding them.

Set `SEMICON_MODE` (`smoke` / `pilot` / `full`) and `SEMICON_DATA` before launching Jupyter - see README."""),
(C, HEAD + r"""
from utils.data import load_npy, downsample
from utils.degrade import fit_degradation
from utils import figures as FG"""),
(M, "## 1. Split manifest"),
(C, r"""from utils.data import match_pairs, make_split_manifest, apply_manifest
from utils.degrade import make_synthetic_dataset

DATA_SOURCE = "KLA"
try:
    splits, man = C.get_splits()
except FileNotFoundError:
    # Let this notebook demonstrate the audit when the private KLA dataset is absent.
    # The marker and separate manifest prevent these data being mistaken for real results.
    DATA_SOURCE = "SYNTHETIC FALLBACK — not research data"
    print("KLA data not found at", C.DATA_ROOT)
    print("Generating a small synthetic dataset for notebook demonstration only.")
    synthetic_root = os.path.join(C.OUT, "setup_synthetic_data")
    make_synthetic_dataset(synthetic_root, n=48, hr=128, seed=0)
    pairs = match_pairs(os.path.join(synthetic_root, "GT"), os.path.join(synthetic_root, "NoisyLR"))
    manifest_path = os.path.join(C.OUT, "setup_synthetic_split_manifest.json")
    man = make_split_manifest(pairs, manifest_path, 0.1, 0.1, C.SPLIT_SEED)
    splits = apply_manifest(pairs, man)
if not all(splits[k] for k in ("train", "val", "test")):
    raise ValueError("No matched .npy image pairs found. Check matching filenames in GT and NoisyLR.")
print("DATA SOURCE:", DATA_SOURCE)
print({k: len(v) for k, v in splits.items()}, "| manifest sha1:", man["sha1"][:12])
assert not (set(man["train"]) & set(man["val"])) and not (set(man["train"]) & set(man["test"])) and not (set(man["val"]) & set(man["test"]))
print("OK: train/val/test are disjoint. Manifest saved to", os.path.join(C.OUT, "split_manifest.json"))
env = dict(torch=torch.__version__, cuda=torch.cuda.is_available(),
           gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           mode=C.MODE, epochs=C.EPOCHS, seeds=C.SEEDS, batch_size=C.BATCH_SIZE, lr=C.LR, split_sha1=man["sha1"])
json.dump(env, open(os.path.join(C.TABLES, "environment.json"), "w"), indent=1); print(env)"""),
(M, "## 2. Value ranges and the reference residual"),
(C, r"""rng = np.random.RandomState(0)
sample = [splits["train"][i] for i in rng.choice(len(splits["train"]), min(200, len(splits["train"])), replace=False)]
rows = []
for g, n in sample:
    gt, ny = load_npy(g), load_npy(n); ds = downsample(gt, ny.shape[:2])
    rows.append(dict(gt_min=gt.min(), gt_max=gt.max(), noisy_min=ny.min(), noisy_max=ny.max(),
                     frac_noisy_below0=(ny < 0).mean(), frac_noisy_above1=(ny > 1).mean(),
                     res_mean=(ny - ds).mean(), res_std=(ny - ds).std()))
audit = pd.DataFrame(rows)
print(audit.describe().loc[["mean", "min", "max"]].round(4).to_string())
print("\nGT shape:", gt.shape, " noisy shape:", ny.shape, "-> scale factor", gt.shape[0] // ny.shape[0])
print("NOTE: degraded values leave [0,1]; metrics clamp predictions to [0,1] and use data_range=1 (as the paper states).")
audit.describe().round(5).to_csv(os.path.join(C.TABLES, "data_audit.csv"))"""),
(C, r"""# Is the reference downsampler D(.) the one the data provider used?  If not, 'residual' contains structured edge error.
# Diagnostic: residual std vs. integer misregistration of D(x) relative to y.  The minimum should be at (0,0).
import torch.nn.functional as F
from scipy import ndimage
def resid_std(shift, mode="area"):
    out = []
    for g, n in sample[:60]:
        gt, ny = load_npy(g), load_npy(n)
        if mode == "area": ds = downsample(gt, ny.shape[:2])
        elif mode == "bilinear": ds = F.interpolate(torch.from_numpy(gt)[None, None], size=ny.shape[:2], mode="bilinear", align_corners=False)[0, 0].numpy()
        else: ds = ndimage.zoom(gt, ny.shape[0] / gt.shape[0], order=1)      # what the 'validates the statistical claim' notebooks used
        ds = np.roll(ds, shift, axis=(0, 1)); out.append((ny - ds)[2:-2, 2:-2].std())
    return float(np.mean(out))
grid = pd.DataFrame({f"dx={dx}": [resid_std((dy, dx)) for dy in (-1, 0, 1)] for dx in (-1, 0, 1)}, index=[f"dy={d}" for d in (-1, 0, 1)])
print("residual std vs misregistration (area-average D):\n", grid.round(5).to_string())
modes = {m: resid_std((0, 0), m) for m in ("area", "bilinear", "zoom_order1")}
print("\nresidual std by choice of D:", {k: round(v, 5) for k, v in modes.items()})
best = np.unravel_index(grid.values.argmin(), grid.shape)
print("minimum at", grid.index[best[0]], grid.columns[best[1]], "->", "OK (aligned)" if best == (1, 1) else "WARNING: reference D is mis-registered/different; report it")
grid.to_csv(os.path.join(C.TABLES, "residual_definition_check.csv"))"""),
(M, "## 3. What do the images look like?  (state it honestly in the paper)\nThe submitted figures from the KLA data show a bristled, specimen-like grayscale object, not an obvious wafer pattern. A semiconductor-manufacturing journal will ask. Look at the sheet and describe the data accurately (and, if it is not wafer imagery, say so and rely on the Carinthia-S SEM experiments for domain claims)."),
(C, r"""idx = np.random.RandomState(1).choice(len(splits["train"]), min(16, len(splits["train"])), replace=False)
fig, ax = plt.subplots(4, 4, figsize=(8, 8))
names = []
for a, i in zip(ax.ravel(), idx):
    g, _ = splits["train"][i]; a.imshow(load_npy(g), cmap="gray", vmin=0, vmax=1); a.axis("off"); names.append(os.path.basename(g))
FG.save(fig, C.FIGS, "fig_data_contact_sheet", {"split": "train", "files": names, "selection": "random, seed=1"})
print("saved contact sheet:", names[:4], "...")"""),
(M, "## 4. Estimate the degradation parameters from TRAIN pairs only\nMoment matching `E[r^2 | I] = a I^2 + b` on flat regions (a = 1/(4 L), b = sigma^2/4). `r2_signal_dependence` tells you whether the model form fits at all."),
(C, r"""fit = fit_degradation(splits["train"], n_images=min(400, len(splits["train"])))
json.dump(fit, open(os.path.join(C.TABLES, "degradation_fit.json"), "w"), indent=1)
print(json.dumps(fit, indent=1))
print(f"\nUse looks={fit['looks']:.2f}, sigma={fit['sigma']:.4f} for ALL synthetic degradations (robustness / OOD / downstream).")
if fit["r2_signal_dependence"] < 0.8:
    print("WARNING: weak signal-dependence fit - the multiplicative-speckle + Gaussian model may not describe this data; say so in the paper.")"""),
]

# ======================================================================================= 01
NB["01_Statistical_Analysis.ipynb"] = [
(M, r"""# 01 - Statistical Re-analysis of Section IV
**Motivation for the method; prerequisite for everything else.**

The submitted analysis argued that the *shape* of the residual distribution varies spatially. Three weaknesses (found in the audit):

* The ANOVA grouped patches by **image**, so it tested between-image differences, not spatial variation inside an image; patches within an image are not independent, so p < 1e-300 is meaningless.
* The structural features (variance, edge density, entropy) were computed on the **clean image**.
* An obvious alternative was never tested: with **multiplicative speckle the noise scale depends on intensity**. Pooling pixels of different scales gives heavy tails globally, and a patch straddling an edge mixes scales so its fitted beta is lower. This reproduces *all* the reported correlations with **no shape heterogeneity at all**.

This notebook tests that alternative directly, with image-clustered inference.
**Decision rule (written before looking at results):** the heterogeneity-of-shape claim is supported only if the observed patch-beta spread and beta-feature correlations *exceed* what the heteroscedastic-Gaussian null produces."""),
(C, HEAD + r"""
from scipy import stats
from utils.data import load_npy, downsample, match_pairs, make_split_manifest, apply_manifest
from utils.degrade import make_synthetic_dataset
from utils import distfit as D, figures as FG
DATA_SOURCE = "KLA"
try:
    splits, man = C.get_splits()
except FileNotFoundError:
    DATA_SOURCE = "SYNTHETIC FALLBACK — not research data"
    print("KLA data not found at", C.DATA_ROOT)
    print("Using the synthetic demonstration dataset from notebook 00 (or creating it now).")
    synthetic_root = os.path.join(C.OUT, "setup_synthetic_data")
    make_synthetic_dataset(synthetic_root, n=48, hr=128, seed=0)
    pairs = match_pairs(os.path.join(synthetic_root, "GT"), os.path.join(synthetic_root, "NoisyLR"))
    manifest_path = os.path.join(C.OUT, "setup_synthetic_split_manifest.json")
    man = make_split_manifest(pairs, manifest_path, 0.1, 0.1, C.SPLIT_SEED)
    splits = apply_manifest(pairs, man)
print("DATA SOURCE:", DATA_SOURCE)
tr = splits["train"]     # analysis uses the TRAIN split only (no val/test leakage)
if not tr:
    raise ValueError("The training split is empty; check matched GT/NoisyLR filenames.")
LR = load_npy(tr[0][1]).shape[0]; P = 32 if LR >= 128 else 16
n_img = {"smoke": len(tr), "pilot": 400, "full": len(tr)}[C.MODE]
print("patch size", P, "| images used:", min(n_img, len(tr)))"""),
(M, "## 1. Global model comparison (replaces Table I)\nHomoscedastic families as in the paper, **plus intensity-dependent-scale models**. If a heteroscedastic model beats the global GGD, the 'heavy-tailed' premise must be reframed."),
(C, r"""cmp_, ab = D.global_model_comparison(tr, n_images=min(200, len(tr)))
print(cmp_.round(1).to_string(index=False)); print(ab)
cmp_.to_csv(os.path.join(C.TABLES, "tableI_global_model_comparison.csv"), index=False)
best = cmp_.iloc[0]["model"]
print("\nBest model by AIC:", best, "->", "heteroscedastic structure matters" if best.startswith("Hetero") else "global GGD preferred")"""),
(M, "## 2. Patch-level table\nFast moment estimator of beta (checked against the MLE below), residual statistics and *clean-image* features per 32x32 patch."),
(C, r"""tab = D.patch_table(tr, P, max_images=n_img)
print(len(tab), "patches |", f"beta mean={tab.beta.mean():.3f} std={tab.beta.std():.3f}")
chk = []
for g, n in tr[:min(20, len(tr))]:
    gt, ny = load_npy(g), load_npy(n); r = ny - downsample(gt, ny.shape[:2])
    for i in range(0, ny.shape[0] - P + 1, P):
        for j in range(0, ny.shape[1] - P + 1, P):
            p = r[i:i+P, j:j+P].ravel(); chk.append((D.ggd_beta_moment(p[None])[0], stats.gennorm.fit(p, floc=0)[0]))
chk = np.array(chk)
print(f"moment vs MLE beta over {len(chk)} patches: Spearman rho = {stats.spearmanr(chk[:,0], chk[:,1])[0]:.3f}, mean moment={chk[:,0].mean():.3f}, mean MLE={chk[:,1].mean():.3f}")"""),
(M, "## 3. What does the heterogeneity look like when tested properly?"),
(C, r"""vd = D.variance_decomposition(tab)
print("variance of patch beta  BETWEEN images:", round(vd["var_between"], 4), "| WITHIN images:", round(vd["var_within"], 4), "| ICC =", round(vd["ICC"], 3))
feats = ["edge_density", "gt_var", "entropy"]
obs = {f: D.cluster_bootstrap_corr(tab, f, n_boot=300 if C.MODE != "full" else 1000) for f in feats}
part = {f: D.partial_corr(tab, f, "beta", "gt_mean") for f in feats}
for f in feats:
    print(f"r(beta, {f:13s}) = {obs[f]['r']:+.3f}  image-clustered 95% CI {tuple(round(v,3) for v in obs[f]['ci95'])}   |  partial r | local intensity = {part[f]:+.3f}")"""),
(M, "## 4. Null models\n**Homogeneous-shape null**: every pixel from the *same* GGD (global beta) - what pure estimation noise gives.  \n**Heteroscedastic-Gaussian null**: `r ~ N(0, a I^2 + b)` given the clean intensity - every pixel exactly Gaussian, *no shape heterogeneity*."),
(C, r"""n_null = min(n_img, 300)
null_h = D.homogeneous_null(tr, ab["ggd_beta"], patch=P, max_images=n_null)
null_g = D.heteroscedastic_null(tr, ab["a"], ab["b"], patch=P, max_images=n_null)
table = D.compare_observed_vs_null(tab, {"homogeneous GGD null": null_h, "hetero-Gaussian null": null_g}, feats)
print(table.round(3).to_string(index=False)); table.to_csv(os.path.join(C.TABLES, "section4_observed_vs_null.csv"), index=False)"""),
(C, r"""# ---- decision rule ----
rows = []
for f in feats:
    r_obs, (lo, hi) = obs[f]["r"], obs[f]["ci95"]
    r_null = table.loc[table.table == "hetero-Gaussian null", f"r(beta,{f})"].item()
    rows.append(dict(feature=f, r_observed=r_obs, ci_lo=lo, ci_hi=hi, r_hetero_null=r_null, null_inside_CI=bool(lo <= r_null <= hi)))
dec = pd.DataFrame(rows)
sd_obs, sd_null = tab.beta.std(), null_g.beta.std()
excess_sd = float(np.sqrt(max(sd_obs ** 2 - sd_null ** 2, 0)))
mean_gap = float(tab.beta.mean() - null_g.beta.mean())
print(dec.round(3).to_string(index=False))
print(f"\npatch-beta SD: observed {sd_obs:.3f} vs hetero-Gaussian null {sd_null:.3f}  -> EXCESS SD attributable to genuine shape variation = {excess_sd:.3f}")
print(f"mean patch beta: observed {tab.beta.mean():.3f} vs null {null_g.beta.mean():.3f}  (gap {mean_gap:+.3f})")
explained = dec.null_inside_CI.all() and excess_sd < 0.5 * sd_null
verdict = ("NOT SUPPORTED: signal-dependent (heteroscedastic) Gaussian noise reproduces the heavy-tailed pooled fit, the patch-beta spread and the "
           "feature correlations. Reframe Section IV as 'signal-dependent noise scale', and test the method against a scale-only model."
           if explained else
           "PARTLY/FULLY SUPPORTED: observed heterogeneity exceeds the heteroscedastic-Gaussian null on at least one criterion; "
           "report the excess and keep the shape-heterogeneity motivation, scoped to the quantified excess.")
print("\nVERDICT:", verdict)
json.dump({"global_best_model": best, "ab": ab, "ICC": vd, "observed": obs, "partial_corr_vs_local_intensity": part,
           "excess_sd": excess_sd, "mean_gap": mean_gap, "heterogeneity_of_shape_supported": (not explained), "verdict": verdict},
          open(os.path.join(C.TABLES, "section4_summary.json"), "w"), indent=1, default=float)"""),
(M, "## 5. Figures (caption numbers come from the saved JSON - do not retype them)"),
(C, r"""prov = {"split": "train", "n_patches": int(len(tab)), "patch": P, "estimator": "moment (checked vs MLE)",
        "r_observed": {f: obs[f]["r"] for f in feats}}
FG.scatter_beta(tab, feats, C.FIGS, "fig_beta_vs_structure_observed", prov)
FG.scatter_beta(null_g, feats, C.FIGS, "fig_beta_vs_structure_hetero_null", {**prov, "table": "heteroscedastic-Gaussian null"})
fig, ax = plt.subplots(figsize=(5, 3.4))
ax.hist(tab.beta, bins=50, alpha=.6, density=True, label="observed"); ax.hist(null_g.beta, bins=50, alpha=.6, density=True, label="hetero-Gaussian null")
ax.hist(null_h.beta, bins=50, alpha=.6, density=True, label="homogeneous GGD null"); ax.set_xlabel("patch beta"); ax.legend(); ax.set_title("Is patch-beta spread beyond nulls?")
FG.save(fig, C.FIGS, "fig_patch_beta_vs_nulls", {"sd_obs": sd_obs, "sd_null_hetero": sd_null, "excess_sd": excess_sd})
print("figures saved to", C.FIGS)"""),
]

# ======================================================================================= 02
NB["02_LDMH_Unit_Tests.ipynb"] = [
(M, r"""# 02 - LDMH Unit Tests
**Addresses the Section VI claims and the module's validity (supporting comments 3 and 5).**

Checks, each printing PASS / FAIL (nothing is hidden):
1. model sizes match the paper (86,945 / 116,138 parameters);
2. the **clamp dead-zone** exists, and the diversity-penalty gradient vanishes for tied clamped betas (the paper's claim);
3. the **sigmoid is not exactly gradient-free in theory but IS in float32** beyond a finite pre-activation (the paper's 'never exactly zero' needs a caveat);
4. the old NLL's `clamp(min=eps)` created a zero-gradient region of its own - fixed in v2;
5. the **softmax temperature T=1.5 is really applied** (the submitted code never applied it);
6. **synthetic recovery**: can LDMH recover a known spatial partition of *different noise shapes* when it only sees the observable image? (The submitted test fed it the residual itself and scored 67.4%, below its own 85% threshold - never reported.)"""),
(C, HEAD + r"""
import torch.nn.functional as F
from models.ldmh import (LocalDistributionMixtureHead, mixture_gen_gaussian_nll, beta_diversity_penalty, load_balance_penalty_entropy)
from models.restoration_net import BaselineNet, DistributionMixtureRestorationNet
def check(name, ok, extra=""): print(("PASS " if ok else "FAIL ") + name, extra)
RESULTS = {}"""),
(C, r"""n = lambda m: sum(p.numel() for p in m.parameters())
check("baseline parameter count == 86,945", n(BaselineNet()) == 86945, n(BaselineNet()))
check("full parameter count == 116,138", n(DistributionMixtureRestorationNet()) == 116138, n(DistributionMixtureRestorationNet()))"""),
(C, r"""# 2. clamp: tied betas beyond the ceiling -> zero gradient from the diversity penalty (paper Section VI-A)
l = torch.tensor([5.0, 5.2], requires_grad=True)
b = torch.exp(l).clamp(0.3, 2.5); beta_diversity_penalty(b).backward()
check("clamp: tied betas are bitwise identical", bool(b[0] == b[1]), b.tolist())
check("clamp: diversity-penalty gradient is exactly zero", bool((l.grad == 0).all()), l.grad.tolist())
l2 = torch.tensor([5.0, 5.2], requires_grad=True)
b2 = 0.3 + 2.2 * torch.sigmoid(l2); beta_diversity_penalty(b2).backward()
check("sigmoid: same configuration still receives gradient", bool((l2.grad != 0).all()), l2.grad.tolist())"""),
(C, r"""# 3. float32 saturation of the sigmoid: where does the gradient become EXACTLY zero?
sat = None
for z in np.arange(5, 40, 0.5):
    t = torch.tensor([float(z)], requires_grad=True); (0.3 + 2.2 * torch.sigmoid(t)).sum().backward()
    if t.grad.item() == 0.0: sat = float(z); break
print("float32: d beta / d l becomes exactly 0 at pre-activation l >=", sat)
bound = lambda l: 0.3 + 2.2 / (1 + np.exp(-l))
print("beta at that point:", bound(sat), "-> the paper's 'never exactly zero' holds mathematically but NOT in float32; say 'practically vanishing'.")
RESULTS["sigmoid_float32_saturation_l"] = sat"""),
(C, r"""# 4. the old NLL's clamp(min=eps) on r/scale: conceptually the same defect, but how large is the dead zone in practice?
def nll_old(res, w, beta, scale, eps=1e-6):
    K = beta.shape[0]; r = res.abs()
    b_, s_ = beta.view(1, K, 1, 1), scale.view(1, K, 1, 1)
    lc = torch.log(b_) - torch.log(2 * s_) - torch.lgamma(1.0 / b_) - (r / s_).clamp(min=eps).pow(b_)
    return -torch.logsumexp(torch.log(w + eps) + lc, 1).mean()
def grad_beta(fn, r0):
    res = torch.full((1, 1, 8, 8), r0); w = torch.full((1, 3, 8, 8), 1 / 3)
    be = torch.tensor([0.5, 1.0, 2.0], requires_grad=True); sc = torch.tensor([0.05] * 3, requires_grad=True)
    fn(res, w, be, sc).backward(); return be.grad.clone(), sc.grad.clone()
# isolate the |r/s|^beta term's gradient: difference between a tiny and a typical residual
for nm, fn in [("old (clamp)", nll_old), ("v2 (additive eps)", mixture_gen_gaussian_nll)]:
    g_tiny, _ = grad_beta(fn, 1e-9); g_zero, _ = grad_beta(fn, 0.0)
    print(f"{nm:18s} d NLL/d beta at |r|=1e-9: {g_tiny.tolist()}   |r|=0: {g_zero.tolist()}")
frac = float((np.abs(np.random.RandomState(0).normal(0, 0.085, 10 ** 6)) < 1e-6 * 0.05).mean())
print(f"fraction of N(0,0.085) residuals inside the old dead zone (|r| < eps*scale): {frac:.2e}")
check("old-NLL dead zone is negligible in practice (<1e-5 of pixels): v2 change is a cleanliness fix, NOT a source of gains", frac < 1e-5)"""),
(C, r"""# 5. temperature is applied in v2
torch.manual_seed(0)
x = torch.randn(1, 1, 16, 16)
def ent(T):
    torch.manual_seed(1); m = LocalDistributionMixtureHead(temperature=T)
    w = m(x)[0]; return float(-(w * (w + 1e-9).log()).sum(1).mean())
e1, e15 = ent(1.0), ent(1.5)
check("T=1.5 gives flatter mixture weights than T=1.0 (temperature is applied)", e15 > e1, f"entropy {e1:.4f} -> {e15:.4f}")"""),
(M, "## 6. Synthetic recovery of a known partition\nTruth: blob regions where residuals are heavy-tailed (beta 0.6) vs near-Gaussian (beta 1.9). The head sees only the **observable** noisy image (texture amplitude differs by region), not the residual. Repeated over 5 seeds. Thresholds fixed in advance: pixel agreement > 0.90 and each recovered beta within 25% of truth."),
(C, r"""from scipy import ndimage, stats
def make_field(seed, S=96):
    rng = np.random.RandomState(seed)
    mask = ndimage.gaussian_filter(rng.randn(S, S), 8) > 0
    resid = np.where(mask, stats.gennorm.rvs(0.6, scale=0.03, size=(S, S), random_state=rng), stats.gennorm.rvs(1.9, scale=0.05, size=(S, S), random_state=rng))
    tex = rng.randn(S, S) * np.where(mask, 0.25, 0.05)                           # observable structure differs by region
    return mask, resid.astype(np.float32), (0.5 + tex).astype(np.float32)
steps = 40 if C.MODE == "smoke" else 400
rows = []
for seed in range(5):
    mask, resid, img = make_field(seed)
    torch.manual_seed(seed)
    m = LocalDistributionMixtureHead(base_ch=16, n_components=2, init_scale=0.04)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    xi, rr = torch.from_numpy(img)[None, None], torch.from_numpy(resid)[None, None]
    for _ in range(steps):
        w, be, sc, _ = m(xi); loss = mixture_gen_gaussian_nll(rr, w, be, sc); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): w, be, sc, _ = m(xi)
    heavy = int(be.argmin()); pred = (w[0, heavy] > 0.5).numpy()
    agree = max((pred == mask).mean(), (pred != mask).mean())                  # label-switch invariant
    # context: a NON-learned detector of the same observable cue (local std of the input, Otsu threshold)
    from skimage.filters import threshold_otsu
    ls = np.sqrt(np.maximum(ndimage.uniform_filter(img ** 2, 5) - ndimage.uniform_filter(img, 5) ** 2, 0)); base_pred = ls > threshold_otsu(ls)
    base_agree = max((base_pred == mask).mean(), (base_pred != mask).mean())
    rows.append(dict(seed=seed, agreement=agree, local_std_baseline=base_agree, beta_lo=float(be.min()), beta_hi=float(be.max()), scale_lo=float(sc[be.argmin()]), scale_hi=float(sc[be.argmax()])))
rec = pd.DataFrame(rows); print(rec.round(3).to_string(index=False))
print("truth: beta 0.6 (scale 0.03) vs 1.9 (scale 0.05)")
ok_a = bool((rec.agreement > 0.90).all()); ok_o = bool(rec.agreement.mean() > 0.85)
ok_b = bool(((rec.beta_lo - 0.6).abs() / 0.6 < 0.25).all() and ((rec.beta_hi - 1.9).abs() / 1.9 < 0.25).all())
check("strict criterion (pre-registered): agreement > 0.90 in ALL seeds", ok_a, f"mean={rec.agreement.mean():.3f}")
check("original notebook-04 criterion: mean agreement > 0.85 (the submitted run scored 0.674)", ok_o)
check("prototype betas recovered within 25% of truth in all seeds", ok_b)
print(f"context: non-learned local-std baseline reaches {rec.local_std_baseline.mean():.3f}; LDMH {rec.agreement.mean():.3f}. "
      f"If LDMH is not clearly above this baseline, its spatial routing adds nothing beyond a simple local-variance cue.")
RESULTS.update(recovery_agreement_mean=float(rec.agreement.mean()), local_std_baseline_mean=float(rec.local_std_baseline.mean()),
               recovery_strict_pass=ok_a, recovery_original_criterion_pass=ok_o, beta_recovery_pass=ok_b)
if not (ok_a and ok_b): print("-> Report this honestly. Do NOT claim that LDMH 'learns local distribution structure' beyond what this shows.")
rec.to_csv(os.path.join(C.TABLES, "ldmh_synthetic_recovery.csv"), index=False)
json.dump(RESULTS, open(os.path.join(C.TABLES, "ldmh_unit_tests.json"), "w"), indent=1, default=float)"""),
]

# ======================================================================================= 03
NB["03_Train_All_Models.ipynb"] = [
(M, r"""# 03 - Train All Models (multi-seed, identical protocol)
**Addresses comments 1, 3, 5.**  Every configuration in the ablation ladder, the new **capacity-matched control**, the penalty/parameterisation ablations and the external architectures are trained with *exactly the same* split manifest, augmentation stream, optimiser (AdamW, cosine), epochs, patience and checkpoint rule, for every seed.

* Runs are resumable (`results.json` marks a finished run) - safe to interrupt a laptop job.
* The test split is evaluated once per run at the best-validation checkpoint; per-image test metrics are stored for paired tests.
* For long jobs run `python scripts/run_experiments.py --mode full` in a terminal instead of keeping this notebook open.

**Recommended order:** run `pilot` first (30 epochs, 2 seeds, `core` group). Continue to `full` only if `film_control` vs `full` is not clearly in the control's favour."""),
(C, HEAD + r"""
import time
from utils.experiment import run_one, CONFIGS, load_results
from utils.data import match_pairs, make_split_manifest, apply_manifest
from utils.degrade import make_synthetic_dataset
DEMO_DATA = False
try:
    splits, man = C.get_splits()
except FileNotFoundError:
    DEMO_DATA = True
    print("KLA data not found at", C.DATA_ROOT)
    print("Using synthetic data for a short demonstration only; these metrics are not research results.")
    synthetic_root = os.path.join(C.OUT, "setup_synthetic_data")
    make_synthetic_dataset(synthetic_root, n=48, hr=128, seed=0)
    pairs = match_pairs(os.path.join(synthetic_root, "GT"), os.path.join(synthetic_root, "NoisyLR"))
    manifest_path = os.path.join(C.OUT, "setup_synthetic_split_manifest.json")
    man = make_split_manifest(pairs, manifest_path, 0.1, 0.1, C.SPLIT_SEED)
    splits = apply_manifest(pairs, man)
if not all(splits[k] for k in ("train", "val", "test")):
    raise ValueError("A train/validation/test split is empty; check matched GT/NoisyLR files.")
RUN_ROOT = os.path.join(C.OUT, "synthetic_demo_runs") if DEMO_DATA else C.RUNS
RUN_EPOCHS = 2 if DEMO_DATA else C.EPOCHS
RUN_BATCH_SIZE = min(4, C.BATCH_SIZE) if DEMO_DATA else C.BATCH_SIZE
GROUPS = {
    "core":     ["baseline", "global_gc", "film_control", "ldmh_loss_only", "full"],
    "ablation": ["full_no_penalty", "full_clamp", "full_no_bal", "full_no_div"],
    "external": ["NAFNet-small", "Restormer-small", "SwinIR-small", "NAFNet-base", "Restormer-base", "SwinIR-base"],
}
TO_RUN = {"smoke": ["core", "ablation", "external"], "pilot": ["core"], "full": ["core", "ablation", "external"]}[C.MODE]   # edit freely
plan = [(n, s) for g in TO_RUN for n in GROUPS[g] for s in C.SEEDS]
if DEMO_DATA:
    plan = [("baseline", 0)]
print(f"{len(plan)} runs planned ({TO_RUN}) | epochs={RUN_EPOCHS} seeds={sorted(set(s for _, s in plan))}")"""),
(C, r"""t0 = time.time(); done = []
for k, (name, seed) in enumerate(plan):
    r = run_one(name, seed, splits, run_root=RUN_ROOT, epochs=RUN_EPOCHS, batch_size=RUN_BATCH_SIZE, lr=C.LR, wd=C.WEIGHT_DECAY,
                clip=C.GRAD_CLIP, patience=C.PATIENCE, device=C.device(), cache=C.CACHE, verbose=(C.MODE != "smoke"))
    done.append(r)
    print(f"[{k+1}/{len(plan)}] {name:18s} seed{seed} | best_ep={r['best_epoch']:>3} val_psnr={r['val']['psnr']:.3f} "
          f"test_psnr={r['test']['psnr']:.3f} | {r['train_minutes']:.1f} min | elapsed {(time.time()-t0)/60:.1f} min")"""),
(C, r"""res = load_results(RUN_ROOT)
cols = ["config", "seed", "params", "best_epoch", "val_psnr", "test_psnr", "test_ssim", "test_hp_err", "minutes"]
print(res[cols].round(4).to_string(index=False))
res.to_csv(os.path.join(C.TABLES, "all_runs_summary.csv"), index=False)"""),
]
