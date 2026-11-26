"""
Re-analysis of Section IV (statistical motivation).

The submitted analysis shows (i) global GGD beta=0.845, (ii) patch beta mean 1.45 and (iii) negative
correlation between patch beta and (clean-image) variance/edges/entropy, and concludes that the *shape*
of the residual distribution varies spatially.  An obvious alternative was never tested:

    SIGNAL-DEPENDENT SCALE.  With multiplicative speckle, Var(r | I) grows with intensity I.
    Pooling pixels with different scales gives heavy tails globally (low beta) even if every pixel is
    exactly Gaussian given its local scale, and a 32x32 patch that straddles edges mixes scales, so its
    fitted beta is lower than that of a flat patch -- which reproduces correlations with edge density /
    variance WITHOUT any shape heterogeneity.

Everything here is built to test that alternative explicitly:

  patch_table              per-patch beta (fast moment estimator), residual stats, clean-image features
  homogeneous_null         patch betas if shape were constant (global GGD) -> pure sampling noise
  heteroscedastic_null     patch betas if r ~ N(0, a*I^2 + b)  (Gaussian given intensity)
  global_model_comparison  AIC: Gaussian / Laplace / Student-t / GGD vs heteroscedastic-Gaussian/-GGD
  cluster_bootstrap_corr   image-clustered CI for the beta-feature correlations
  partial_corr             correlation after regressing out local intensity (mean of clean patch)
  variance_decomposition   between-image vs within-image variance of patch beta (ICC)
"""
import numpy as np
import pandas as pd
from math import lgamma
from scipy import stats, ndimage, optimize
from scipy.special import gammaln

from utils.data import load_npy, downsample

_BG = np.linspace(0.15, 4.0, 4000)
_RHO = np.exp(2 * gammaln(2 / _BG) - gammaln(1 / _BG) - gammaln(3 / _BG))     # (E|r|)^2 / E[r^2]


def ggd_beta_moment(r_patches):
    """Vectorised moment estimator of beta for zero-mean GGD. r_patches: [N, M]."""
    m1 = np.abs(r_patches).mean(1)
    m2 = (r_patches ** 2).mean(1)
    rho = np.clip(m1 ** 2 / (m2 + 1e-18), _RHO.min() + 1e-6, _RHO.max() - 1e-6)
    return np.interp(rho, _RHO, _BG)


def _patchify(a, p):
    H, W = a.shape
    nr, nc = H // p, W // p
    return a[:nr * p, :nc * p].reshape(nr, p, nc, p).transpose(0, 2, 1, 3).reshape(nr * nc, p * p), nr, nc


def patch_table(pairs, patch=32, max_images=None, seed=0, residual_fn=None):
    """One row per patch. residual_fn(gt_ds, noisy)->residual (default: noisy-gt_ds)."""
    rng = np.random.RandomState(seed)
    idx = np.arange(len(pairs)) if not max_images or max_images >= len(pairs) else rng.choice(len(pairs), max_images, replace=False)
    out = []
    for i in idx:
        gt, ny = load_npy(pairs[i][0]), load_npy(pairs[i][1])
        ds = downsample(gt, ny.shape[:2])
        r = (ny - ds) if residual_fn is None else residual_fn(ds, ny)
        gm = np.hypot(ndimage.sobel(ds, 0), ndimage.sobel(ds, 1))
        R, nr, nc = _patchify(r, patch); G, _, _ = _patchify(ds, patch); E, _, _ = _patchify(gm, patch)
        beta = ggd_beta_moment(R)
        ent = np.array([_entropy(g) for g in G])
        for k in range(R.shape[0]):
            out.append((int(i), k // nc, k % nc, beta[k], R[k].std(), np.abs(R[k]).mean(), G[k].mean(), G[k].var(), E[k].mean(), ent[k]))
    return pd.DataFrame(out, columns=["img", "row", "col", "beta", "resid_std", "resid_mad", "gt_mean", "gt_var", "edge_density", "entropy"])


def _entropy(g, bins=16):
    h, _ = np.histogram(g, bins=bins, range=(0, 1), density=True)
    h = h[h > 0]; p = h / h.sum()
    return float(-(p * np.log2(p)).sum())


# ----------------------------------------------------------------------------- nulls
def homogeneous_null(pairs, global_beta, patch=32, max_images=300, seed=0):
    """Constant shape = global_beta, but patch-specific scale taken from the data: isolates pure
    estimation noise of beta on a patch of patch^2 samples."""
    rng = np.random.RandomState(seed)
    def fn(ds, ny):
        real = ny - ds
        sc = np.full_like(real, 1.0) * real.std()
        return stats.gennorm.rvs(global_beta, scale=sc / np.sqrt(stats.gennorm.var(global_beta)), size=real.shape,
                                 random_state=rng)
    return patch_table(pairs, patch, max_images, seed, residual_fn=fn)


def heteroscedastic_null(pairs, a, b, patch=32, max_images=300, seed=0):
    """r ~ N(0, a*I^2 + b) given the clean intensity I  (per-pixel Gaussian -> NO shape heterogeneity)."""
    rng = np.random.RandomState(seed)
    def fn(ds, ny):
        return rng.normal(0, np.sqrt(np.maximum(a * ds ** 2 + b, 1e-12)))
    return patch_table(pairs, patch, max_images, seed, residual_fn=fn)


# ----------------------------------------------------------------------------- global model comparison
def _aic(ll, k):
    return 2 * k - 2 * ll


def global_model_comparison(pairs, n_images=200, seed=0, a_b=None):
    """AIC table.  Homoscedastic families (as in paper Table I) vs intensity-dependent-scale models."""
    rng = np.random.RandomState(seed)
    sel = rng.choice(len(pairs), min(n_images, len(pairs)), replace=False)
    R, I = [], []
    for i in sel:
        gt, ny = load_npy(pairs[i][0]), load_npy(pairs[i][1])
        ds = downsample(gt, ny.shape[:2]); R.append((ny - ds).ravel()); I.append(ds.ravel())
    r, I = np.concatenate(R), np.concatenate(I)
    n = len(r); rows = []
    s = r.std(); rows.append(("Gaussian", 1, stats.norm.logpdf(r, 0, s).sum()))
    sc = np.abs(r).mean(); rows.append(("Laplace", 1, stats.laplace.logpdf(r, 0, sc).sum()))
    df, _, ts = stats.t.fit(r, floc=0); rows.append(("Student-t", 2, stats.t.logpdf(r, df, 0, ts).sum()))
    sub = r[rng.choice(n, min(n, 400000), replace=False)]
    be, _, al = stats.gennorm.fit(sub, floc=0)
    rows.append(("GGD (global)", 2, stats.gennorm.logpdf(r, be, 0, al).sum()))
    # intensity-dependent scale: sigma^2 = a I^2 + b
    def nll_g(th):
        a, b = np.exp(th); v = a * I ** 2 + b
        return 0.5 * (np.log(2 * np.pi * v) + r ** 2 / v).sum()
    o = optimize.minimize(nll_g, np.log([0.01, 0.002]), method="Nelder-Mead")
    rows.append(("Hetero-Gaussian  N(0,aI^2+b)", 2, -o.fun)); ab = np.exp(o.x)
    def nll_gg(th):
        a, b, be_ = np.exp(th[0]), np.exp(th[1]), 0.3 + 3.7 / (1 + np.exp(-th[2]))
        v = np.sqrt(a * I ** 2 + b)                       # scale
        return -(stats.gennorm.logpdf(r, be_, 0, v * np.sqrt(np.exp(gammaln(1 / be_) - gammaln(3 / be_))))).sum()
    o2 = optimize.minimize(nll_gg, [o.x[0], o.x[1], 0.0], method="Nelder-Mead", options={"maxiter": 400})
    be2 = 0.3 + 3.7 / (1 + np.exp(-o2.x[2]))
    rows.append((f"Hetero-GGD (beta={be2:.2f})", 3, -o2.fun))
    df_ = pd.DataFrame(rows, columns=["model", "k", "loglik"])
    df_["AIC"] = [_aic(l, k) for l, k in zip(df_.loglik, df_.k)]
    df_["dAIC_vs_best"] = df_.AIC - df_.AIC.min()
    return df_.sort_values("AIC").reset_index(drop=True), {"a": float(ab[0]), "b": float(ab[1]), "ggd_beta": float(be), "ggd_alpha": float(al), "n": int(n)}


# ----------------------------------------------------------------------------- inference on correlations
def cluster_bootstrap_corr(df, x, y="beta", n_boot=1000, seed=0):
    """Pearson r with image-clustered bootstrap CI (patches inside an image are NOT independent)."""
    rng = np.random.RandomState(seed)
    groups = {k: g[[x, y]].values for k, g in df.groupby("img")}
    keys = np.array(list(groups))
    r0 = stats.pearsonr(df[x], df[y])[0]
    rs = []
    for _ in range(n_boot):
        samp = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
        rs.append(np.corrcoef(samp[:, 0], samp[:, 1])[0, 1])
    return {"r": float(r0), "ci95": (float(np.quantile(rs, .025)), float(np.quantile(rs, .975))), "n_patches": len(df), "n_images": len(keys)}


def partial_corr(df, x, y, control):
    """corr(x,y | control) via residualisation (linear + quadratic in control)."""
    def res(v, c):
        A = np.stack([np.ones_like(c), c, c ** 2], 1)
        return v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
    return float(np.corrcoef(res(df[x].values, df[control].values), res(df[y].values, df[control].values))[0, 1])


def variance_decomposition(df, col="beta"):
    """One-way random-effects ICC: share of patch-beta variance that is BETWEEN images."""
    g = df.groupby("img")[col]
    k = g.size().mean(); n = g.ngroups
    msb = (g.size() * (g.mean() - df[col].mean()) ** 2).sum() / (n - 1)
    msw = ((df[col] - g.transform("mean")) ** 2).sum() / (len(df) - n)
    icc = (msb - msw) / (msb + (k - 1) * msw)
    return {"ICC": float(icc), "var_between": float(max(msb - msw, 0) / k), "var_within": float(msw), "patches_per_image": float(k)}


def compare_observed_vs_null(obs, null_tables, feats=("edge_density", "gt_var", "entropy")):
    rows = []
    for name, t in [("observed", obs)] + list(null_tables.items()):
        r = {"table": name, "beta_mean": t.beta.mean(), "beta_std": t.beta.std()}
        for f in feats:
            r[f"r(beta,{f})"] = stats.pearsonr(t[f], t.beta)[0]
        rows.append(r)
    return pd.DataFrame(rows)
