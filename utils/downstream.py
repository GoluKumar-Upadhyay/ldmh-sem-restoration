"""
Downstream, manufacturing-relevant evaluation (reviewer comment 4).

PSNR/SSIM do not tell whether a defect is still visible after restoration.  Protocol:

  1. Train ONE small U-Net defect segmenter on CLEAN images (Carinthia-S train split) -- fixed for all arms.
  2. Take held-out clean test images, degrade them with the FITTED degradation (utils.degrade), restore
     with every candidate model (and bilinear-upsample as the lower bound; clean as the upper bound).
  3. Run the SAME segmenter on each restored image and score, against the ground-truth defect masks:
        pixel  : precision / recall / F1 / IoU
        defect : instance recall (miss rate = 1-recall), instance precision, false-alarm rate
                 (spurious connected components per image) -- 'instance' = connected component
  4. Add restoration-quality-at-defects metrics: defect CNR (restored vs clean), edge correlation, flat-region
     artefact energy and full-scale HP-Err (see utils.metrics).

Per-image scores are returned so the same paired tests as the main experiments can be applied.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

from utils import metrics as M


class TinyUNet(nn.Module):
    def __init__(self, c=16):
        super().__init__()
        blk = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(4, o), nn.GELU(),
                                         nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(4, o), nn.GELU())
        self.e1, self.e2, self.e3 = blk(1, c), blk(c, 2 * c), blk(2 * c, 4 * c)
        self.d2, self.d1 = blk(6 * c, 2 * c), blk(3 * c, c)
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        a = self.e1(x); b = self.e2(F.max_pool2d(a, 2)); c = self.e3(F.max_pool2d(b, 2))
        b = self.d2(torch.cat([b, F.interpolate(c, scale_factor=2)], 1))
        a = self.d1(torch.cat([a, F.interpolate(b, scale_factor=2)], 1))
        return self.out(a)


def train_segmenter(images, masks, epochs=30, lr=2e-3, bs=8, seed=0, device="cpu"):
    """images [N,1,H,W] float, masks [N,1,H,W] {0,1}.  Dice+BCE.  Returns model (eval)."""
    torch.manual_seed(seed)
    net = TinyUNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    pos = masks.mean().clamp(min=1e-4)
    pw = ((1 - pos) / pos).clamp(max=50.0)
    n = len(images)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            x, y = images[idx].to(device), masks[idx].to(device)
            if torch.rand(1) < 0.5: x, y = x.flip(-1), y.flip(-1)
            if torch.rand(1) < 0.5: x, y = x.flip(-2), y.flip(-2)
            lo = net(x)
            p = torch.sigmoid(lo)
            dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
            loss = F.binary_cross_entropy_with_logits(lo, y, pos_weight=pw) + dice
            opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


@torch.no_grad()
def predict_mask(net, img, thr=0.5, device="cpu"):
    return (torch.sigmoid(net(img.to(device))) > thr).float().cpu()


def defect_scores(pred, gt, min_area=3, iou_thr=0.1):
    """pred, gt: [H,W] {0,1} numpy.  Instance = connected component."""
    tp = float((pred * gt).sum()); fp = float((pred * (1 - gt)).sum()); fn = float(((1 - pred) * gt).sum())
    prec, rec = tp / (tp + fp + 1e-9), tp / (tp + fn + 1e-9)
    out = {"pix_precision": prec, "pix_recall": rec, "pix_f1": 2 * prec * rec / (prec + rec + 1e-9),
           "pix_iou": tp / (tp + fp + fn + 1e-9)}
    gl, ng = ndimage.label(gt); pl, npred = ndimage.label(pred)
    gt_hit = 0
    for k in range(1, ng + 1):
        g = gl == k
        if g.sum() < min_area: continue
        inter = (pred.astype(bool) & g).sum()
        if inter / (g.sum() + 1e-9) >= iou_thr: gt_hit += 1
    n_gt = sum(1 for k in range(1, ng + 1) if (gl == k).sum() >= min_area)
    pred_ok, false_alarm = 0, 0
    for k in range(1, npred + 1):
        p = pl == k
        if p.sum() < min_area: continue
        if (gt.astype(bool) & p).sum() > 0: pred_ok += 1
        else: false_alarm += 1
    out.update({"inst_recall": gt_hit / n_gt if n_gt else np.nan, "inst_miss_rate": 1 - gt_hit / n_gt if n_gt else np.nan,
                "inst_precision": pred_ok / (pred_ok + false_alarm) if (pred_ok + false_alarm) else np.nan,
                "false_alarms_per_img": float(false_alarm), "n_gt_defects": n_gt})
    return out


@torch.no_grad()
def evaluate_arm(net, restored, clean, noisy_lr, masks, device="cpu"):
    """restored/clean [N,1,H,W] tensors, masks [N,1,H,W].  -> list of per-image dicts."""
    rows = []
    q = M.full_eval(restored.clamp(0, 1), clean, noisy_lr)
    cn = M.cnr(restored.clamp(0, 1), masks)
    cn_clean = M.cnr(clean, masks)
    for i in range(len(restored)):
        pm = predict_mask(net, restored[i:i + 1].clamp(0, 1), device=device)[0, 0].numpy()
        r = defect_scores(pm, masks[i, 0].numpy())
        r.update({k: float(v[i]) for k, v in q.items()})
        r["defect_cnr"] = float(cn[i]); r["defect_cnr_ratio_to_clean"] = float(cn[i] / (cn_clean[i] + 1e-9))
        rows.append(r)
    return rows
