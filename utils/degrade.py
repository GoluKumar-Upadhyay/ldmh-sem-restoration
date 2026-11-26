"""
Degradation model  y = D( x * s + n ),  s ~ Gamma(L, 1/L) (unit-mean speckle), n ~ N(0, sigma^2),
D = area-average x2 downsampling.

  * `degrade`            : apply it (optionally only one component, for the per-type robustness test)
  * `fit_degradation`    : ESTIMATE (L, sigma) from real (GT, NoisyLR) pairs by moment matching
                           E[r^2 | I] = a*I^2 + b  with a = 1/(4L), b = sigma^2/4, on flat GT regions.
                           Returns the goodness-of-fit so a bad model is visible instead of assumed.
                           (The submitted OOD test hard-coded L=4, sigma=0.05 while claiming they
                           were the "fitted" values.)
  * `make_synthetic_dataset` : tiny synthetic KLA-like dataset used ONLY for smoke-testing the code.
"""
import os
import numpy as np
from scipy import ndimage

from utils.data import downsample


def degrade(clean, looks=4.0, sigma=0.05, factor=2, speckle=True, gaussian=True, down=True, seed=0):
    rng = np.random.RandomState(seed)
    x = clean.astype(np.float64)
    if speckle:
        x = x * rng.gamma(shape=looks, scale=1.0 / looks, size=x.shape)
    if gaussian:
        x = x + rng.normal(0, sigma, size=x.shape)
    if down:
        x = downsample(x, (x.shape[0] // factor, x.shape[1] // factor))
    return x.astype(np.float32)


def fit_degradation(pairs, n_images=400, n_bins=25, flat_pct=40, seed=0):
    """pairs: list of (gt_path, noisy_path).  Returns dict(looks, sigma, a, b, r2, bias, ...)."""
    from utils.data import load_npy
    rng = np.random.RandomState(seed)
    sel = rng.choice(len(pairs), min(n_images, len(pairs)), replace=False)
    I, R2, R = [], [], []
    for i in sel:
        gt, ny = load_npy(pairs[i][0]), load_npy(pairs[i][1])
        ds = downsample(gt, ny.shape[:2])
        r = ny - ds
        gm = ndimage.generic_gradient_magnitude(ds, ndimage.sobel)
        flat = gm <= np.percentile(gm, flat_pct)
        I.append(ds[flat]); R2.append(r[flat] ** 2); R.append(r[flat])
    I, R2, R = np.concatenate(I), np.concatenate(R2), np.concatenate(R)
    edges = np.quantile(I, np.linspace(0, 1, n_bins + 1))
    cen, mr2 = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (I >= lo) & (I <= hi)
        if m.sum() > 50:
            cen.append(I[m].mean()); mr2.append(R2[m].mean())
    cen, mr2 = np.array(cen), np.array(mr2)
    A = np.stack([cen ** 2, np.ones_like(cen)], 1)
    (a, b), *_ = np.linalg.lstsq(A, mr2, rcond=None)
    pred = A @ np.array([a, b])
    r2 = 1 - ((mr2 - pred) ** 2).sum() / (((mr2 - mr2.mean()) ** 2).sum() + 1e-12)
    looks = 1.0 / (4 * a) if a > 1e-9 else float("inf")
    sigma = float(np.sqrt(max(4 * b, 0.0)))
    return {"looks": float(looks), "sigma": sigma, "a": float(a), "b": float(b), "r2_signal_dependence": float(r2),
            "mean_residual": float(R.mean()), "n_pixels": int(I.size),
            "note": "moment-matching estimate; check r2_signal_dependence before trusting looks/sigma"}


# ----------------------------------------------------------------------------- smoke-test data
def _synthetic_clean(size, rng):
    x = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, x)
    img = 0.45 + 0.2 * np.sin(2 * np.pi * rng.uniform(1, 4) * xx) * np.cos(2 * np.pi * rng.uniform(1, 4) * yy)
    for _ in range(rng.randint(4, 9)):
        cy, cx = rng.randint(0, size, 2); h, w = rng.randint(size // 16, size // 4, 2)
        img[max(0, cy - h):cy + h, max(0, cx - w):cx + w] = rng.uniform(0.1, 0.95)
    return np.clip(ndimage.gaussian_filter(img, 0.7), 0, 1).astype(np.float32)


def make_synthetic_dataset(root, n=48, hr=128, looks=4.0, sigma=0.05, seed=0):
    rng = np.random.RandomState(seed)
    gt_dir, ny_dir = os.path.join(root, "GT"), os.path.join(root, "NoisyLR")
    os.makedirs(gt_dir, exist_ok=True); os.makedirs(ny_dir, exist_ok=True)
    for i in range(n):
        g = _synthetic_clean(hr, rng)
        np.save(os.path.join(gt_dir, f"{i:06d}.npy"), g)
        np.save(os.path.join(ny_dir, f"{i:06d}.npy"), degrade(g, looks, sigma, seed=i))
    return gt_dir, ny_dir
