"""
Metrics (v2).  All functions are per-image (return a tensor [B]) so that paired statistics
can be computed on the test set.

  * psnr / ssim         : SSIM is the standard Gaussian-window (11x11, sigma 1.5) of Wang et al. [25].
                          (the submitted code used a 7x7 *uniform* window -> not what the paper cites)
  * hp_err              : MAE on the top-q% highest-|residual| pixels (full-scale, per image)
  * edge_preservation   : normalised cross-correlation of Sobel gradient magnitudes (restored vs clean)
  * edge_f1             : F1 of thresholded edges (threshold = percentile of the clean edge map)
  * artifact_energy     : mean |restored - clean| on FLAT regions of the clean image (over-smoothing /
                          ringing is visible at edges, residual noise and texture artefacts in flats)
  * cnr                 : contrast-to-noise ratio of a region of interest vs. its surround
"""
import numpy as np
import torch
import torch.nn.functional as F


def psnr(pred, gt, data_range=1.0):
    mse = ((pred - gt) ** 2).flatten(1).mean(1).clamp(min=1e-10)
    return 10 * torch.log10(data_range ** 2 / mse)


def _gauss(win=11, sigma=1.5, device="cpu"):
    ax = torch.arange(win, dtype=torch.float32, device=device) - (win - 1) / 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(pred, gt, data_range=1.0, win=11, sigma=1.5):
    k = _gauss(win, sigma, pred.device)
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    f = lambda x: F.conv2d(x, k)           # 'valid' window, as in skimage / Wang et al.
    mx, my = f(pred), f(gt)
    sxx, syy, sxy = f(pred * pred) - mx ** 2, f(gt * gt) - my ** 2, f(pred * gt) - mx * my
    s = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))
    return s.flatten(1).mean(1)


def hp_err(pred, gt, noisy_lr, q=0.05):
    """MAE on the top-q fraction of pixels where the bilinear-upsampled input is worst."""
    up = F.interpolate(noisy_lr, size=gt.shape[-2:], mode="bilinear", align_corners=False)
    base = (up - gt).abs().flatten(1)
    err = (pred - gt).abs().flatten(1)
    k = max(1, int(q * base.shape[1]))
    idx = torch.topk(base, k, dim=1).indices
    return torch.gather(err, 1, idx).mean(1)


def _sobel_mag(x):
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device)[None, None]
    gx, gy = F.conv2d(x, kx, padding=1), F.conv2d(x, kx.transpose(-1, -2), padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)


def edge_preservation(pred, gt):
    a, b = _sobel_mag(pred).flatten(1), _sobel_mag(gt).flatten(1)
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12)


def edge_f1(pred, gt, pct=90):
    mg, mp = _sobel_mag(gt).flatten(1), _sobel_mag(pred).flatten(1)
    th = torch.quantile(mg, pct / 100, dim=1, keepdim=True)
    tp_ = torch.quantile(mp, pct / 100, dim=1, keepdim=True)
    g, p = (mg >= th).float(), (mp >= tp_).float()
    tp = (g * p).sum(1)
    prec, rec = tp / (p.sum(1) + 1e-9), tp / (g.sum(1) + 1e-9)
    return 2 * prec * rec / (prec + rec + 1e-9)


def artifact_energy(pred, gt, flat_pct=30):
    m = _sobel_mag(gt).flatten(1)
    th = torch.quantile(m, flat_pct / 100, dim=1, keepdim=True)
    flat = (m <= th).float()
    e = (pred - gt).abs().flatten(1)
    return (e * flat).sum(1) / (flat.sum(1) + 1e-9)


def cnr(img, roi_mask, surround_dilate=7):
    """img,[B,1,H,W]; roi_mask [B,1,H,W] bool/0-1.  CNR = |mu_roi - mu_bg| / std_bg."""
    out = []
    m = roi_mask.float()
    ring = F.max_pool2d(m, surround_dilate, 1, surround_dilate // 2) - m
    for i in range(img.shape[0]):
        r, b = img[i][m[i] > 0.5], img[i][ring[i] > 0.5]
        if r.numel() < 4 or b.numel() < 4:
            out.append(torch.tensor(float("nan"))); continue
        out.append((r.mean() - b.mean()).abs() / (b.std() + 1e-6))
    return torch.stack(out)


@torch.no_grad()
def full_eval(pred, gt, noisy_lr):
    """Dict of per-image metrics (tensors [B]) for a restored batch."""
    p = pred.clamp(0, 1)
    return {"psnr": psnr(p, gt), "ssim": ssim(p, gt), "hp_err": hp_err(p, gt, noisy_lr),
            "edge_corr": edge_preservation(p, gt), "edge_f1": edge_f1(p, gt),
            "flat_err": artifact_energy(p, gt)}
