"""
Verified loader for the Carinthia-S SEM wafer-defect dataset (Zenodo record 16895427; confirm the exact
version/DOI on the record page and write it in the paper).

WHY THIS EXISTS: the submitted out-of-distribution experiment read the first ten image files of data.zip and
the first one was  data/masks/<hash>.png , i.e. a SEGMENTATION MASK, not an SEM image.  This loader
  * separates `images` from `masks` by directory name,
  * REFUSES to use a file as an image if it looks like a label map (few distinct grey levels),
  * records a manifest (file name, sha256, original shape, selection seed) that is saved next to the results
    and must be quoted in the paper.
Expected layout (auto-detected, or pass explicit dirs):
    <root>/**/images/*.png|jpg|tif      and      <root>/**/masks/*.png     (stem-matched)
"""
import glob
import hashlib
import json
import os
import numpy as np
from PIL import Image

EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def looks_like_mask(arr, max_levels=8):
    a = np.asarray(arr)
    if a.ndim == 3:
        a = a[..., 0]
    return len(np.unique(a)) <= max_levels


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()[:16]


def discover(root, image_dir=None, mask_dir=None):
    files = [p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True) if p.lower().endswith(EXTS)]
    mask_f = [p for p in files if "mask" in p.lower().replace("\\", "/").split("/")[-2]] if mask_dir is None \
        else glob.glob(os.path.join(mask_dir, "*"))
    img_f = [p for p in files if p not in set(mask_f)] if image_dir is None else glob.glob(os.path.join(image_dir, "*"))
    img_f = [p for p in img_f if p.lower().endswith(EXTS)]
    return sorted(img_f), sorted(mask_f)


def load_pairs(root, n=None, seed=0, size=256, image_dir=None, mask_dir=None, manifest_path=None, require_masks=False):
    """-> list of dict(image float32 [size,size] in [0,1], mask uint8 [size,size] or None, name, sha)."""
    imgs, masks = discover(root, image_dir, mask_dir)
    if not imgs:
        raise FileNotFoundError(f"no image files found under {root}")
    mk = {os.path.splitext(os.path.basename(p))[0]: p for p in masks}
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(imgs))
    out, bad = [], []
    for i in order:
        p = imgs[i]
        a = np.asarray(Image.open(p).convert("L"))
        if looks_like_mask(a):
            bad.append(p); continue                                   # never use a label map as an image
        stem = os.path.splitext(os.path.basename(p))[0]
        m = None
        if stem in mk:
            m = np.asarray(Image.open(mk[stem]).convert("L").resize((size, size), Image.NEAREST))
            m = (m > 0).astype(np.uint8)
        if require_masks and m is None:
            continue
        im = np.asarray(Image.fromarray(a).resize((size, size), Image.BICUBIC), dtype=np.float32) / 255.0
        out.append({"image": np.clip(im, 0, 1), "mask": m, "name": os.path.basename(p), "sha": _sha(p),
                    "orig_shape": a.shape, "path": p})
        if n and len(out) >= n:
            break
    if manifest_path:
        os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
        json.dump({"root": str(root), "seed": seed, "n": len(out), "skipped_mask_like_files": len(bad),
                   "items": [{k: (list(v) if k == "orig_shape" else v) for k, v in o.items() if k in ("name", "sha", "orig_shape")} for o in out]},
                  open(manifest_path, "w"), indent=1)
    if not out:
        raise RuntimeError("all candidate files looked like masks -- check the directory layout")
    return out


def make_synthetic_carinthia(root, n=30, size=64, seed=0):
    """SMOKE-TEST ONLY: fake 'SEM' images with blob defects + masks in the layout load_pairs expects."""
    from scipy import ndimage
    rng = np.random.RandomState(seed)
    di, dm = os.path.join(root, "images"), os.path.join(root, "masks")
    os.makedirs(di, exist_ok=True); os.makedirs(dm, exist_ok=True)
    for i in range(n):
        base = ndimage.gaussian_filter(rng.rand(size, size), 3); base = 0.3 + 0.4 * (base - base.min()) / (np.ptp(base) + 1e-9)
        m = np.zeros((size, size), np.uint8)
        for _ in range(rng.randint(1, 4)):
            cy, cx = rng.randint(6, size - 6, 2); r = rng.randint(2, 5)
            yy, xx = np.ogrid[:size, :size]; blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
            m[blob] = 1; base[blob] = np.clip(base[blob] + rng.choice([-0.3, 0.3]), 0, 1)
        Image.fromarray((np.clip(base, 0, 1) * 255).astype(np.uint8)).save(os.path.join(di, f"img_{i:04d}.png"))
        Image.fromarray(m * 255).save(os.path.join(dm, f"img_{i:04d}.png"))
    return root
