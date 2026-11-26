"""
Data utilities (v2).

Fixes vs. submitted code
  * THREE-way split (train / val / test) stored in a JSON manifest.  Validation is used ONLY
    for early stopping / checkpoint selection; the test split is touched once, at the end.
    (The paper used the same 320 images for selection, tuning and reporting.)
  * One shared definition of the reference residual r = y - D(x) with D = area-average
    (documented, used everywhere; notebooks assert it).
  * Optional in-RAM cache (the old loader re-read .npy files from disk each step).
  * Reproducible augmentation (torch.Generator seeded per run).
"""
import glob
import hashlib
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ----------------------------------------------------------------------------- pairing
def list_npy(folder):
    paths = sorted(glob.glob(os.path.join(str(folder), "*.npy")))
    return [p for p in paths if "__MACOSX" not in p and not os.path.basename(p).startswith("._")]


def match_pairs(gt_dir, noisy_dir):
    g = {os.path.basename(p): p for p in list_npy(gt_dir)}
    n = {os.path.basename(p): p for p in list_npy(noisy_dir)}
    common = sorted(set(g) & set(n))
    return [(g[k], n[k]) for k in common]


def load_npy(path):
    return np.load(path).astype(np.float32)


def downsample(img, out_hw):
    """Area-average D(.) -- THE reference downsampler used for every residual in the paper."""
    H, W = img.shape[:2]
    oh, ow = out_hw
    fh, fw = H // oh, W // ow
    return img[:oh * fh, :ow * fw].reshape(oh, fh, ow, fw).mean(axis=(1, 3))


# ----------------------------------------------------------------------------- split manifest
def make_split_manifest(pairs, path, val_frac=0.1, test_frac=0.1, seed=42):
    """Create (or load, if present) a deterministic train/val/test manifest keyed by file name."""
    if os.path.exists(path):
        return json.load(open(path))
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(pairs))
    n_val, n_test = int(len(pairs) * val_frac), int(len(pairs) * test_frac)
    names = [os.path.basename(p[0]) for p in pairs]
    man = {
        "seed": seed, "n_total": len(pairs),
        "test": sorted(names[i] for i in idx[:n_test]),
        "val": sorted(names[i] for i in idx[n_test:n_test + n_val]),
        "train": sorted(names[i] for i in idx[n_test + n_val:]),
    }
    man["sha1"] = hashlib.sha1(json.dumps([man["train"], man["val"], man["test"]]).encode()).hexdigest()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(man, open(path, "w"), indent=1)
    return man


def apply_manifest(pairs, man):
    by = {os.path.basename(p[0]): p for p in pairs}
    out = {k: [by[n] for n in man[k]] for k in ("train", "val", "test")}
    assert not (set(man["train"]) & set(man["val"])) and not (set(man["train"]) & set(man["test"])) \
        and not (set(man["val"]) & set(man["test"])), "split leakage!"
    return out


# ----------------------------------------------------------------------------- dataset
class PairedRestorationDataset(Dataset):
    """returns (noisy_LR, gt_HR, gt_ds, index)  -- all [1,H,W] float32."""

    def __init__(self, pairs, augment=False, cache=True, seed=0):
        self.pairs, self.augment = pairs, augment
        self.gen = torch.Generator().manual_seed(seed)
        self._cache = None
        if cache:
            self._cache = [self._read(i) for i in range(len(pairs))]

    def _read(self, i):
        gt_p, ny_p = self.pairs[i]
        gt, ny = load_npy(gt_p), load_npy(ny_p)
        ds = downsample(gt, ny.shape[:2])
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).unsqueeze(0).float()
        return t(ny), t(gt), t(ds)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ny, gt, ds = self._cache[i] if self._cache is not None else self._read(i)
        if self.augment:
            r = torch.randint(0, 2, (2,), generator=self.gen).tolist()
            k = int(torch.randint(0, 4, (1,), generator=self.gen))
            def tf(x):
                if r[0]: x = torch.flip(x, [-1])
                if r[1]: x = torch.flip(x, [-2])
                return torch.rot90(x, k, [-2, -1]) if k else x
            ny, gt, ds = tf(ny), tf(gt), tf(ds)
        return ny, gt, ds, i


def make_loaders(splits, batch_size=8, seed=0, cache=True, num_workers=0):
    tr = PairedRestorationDataset(splits["train"], augment=True, cache=cache, seed=seed)
    va = PairedRestorationDataset(splits["val"], augment=False, cache=cache)
    te = PairedRestorationDataset(splits["test"], augment=False, cache=cache)
    g = torch.Generator().manual_seed(seed)
    return (DataLoader(tr, batch_size, shuffle=True, drop_last=True, generator=g, num_workers=num_workers),
            DataLoader(va, batch_size, shuffle=False, num_workers=num_workers),
            DataLoader(te, batch_size, shuffle=False, num_workers=num_workers))
