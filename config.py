"""
Single place for paths and the experimental protocol.

    SEMICON_MODE = smoke | pilot | full        (default: pilot)
        smoke : synthetic data, 2 epochs  -> only checks that the code runs (numbers are meaningless)
        pilot : your KLA data, 30 epochs, 2 seeds  -> decide quickly whether the claim survives
        full  : your KLA data, 200 epochs, 5 seeds -> the numbers for the paper
    SEMICON_DATA      folder containing train/train/GT and train/train/NoisyLR  (or GT/ and NoisyLR/)
    SEMICON_CARINTHIA folder with the unzipped Carinthia-S data (images + masks)
    SEMICON_RUNS      where runs/, figures/, tables/ are written (default: <project>/results_<mode>)
"""
import os
import sys

PROJECT = os.path.dirname(os.path.abspath(__file__))
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

MODE = os.environ.get("SEMICON_MODE", "pilot").lower()
SMOKE = MODE == "smoke"
assert MODE in ("smoke", "pilot", "full"), MODE

PROTOCOL = {
    "smoke": dict(epochs=2, seeds=[0, 1], batch_size=4, patience=5, cache=True),
    "pilot": dict(epochs=30, seeds=[0, 1], batch_size=8, patience=30, cache=True),
    "full":  dict(epochs=200, seeds=[0, 1, 2, 3, 4], batch_size=8, patience=30, cache=True),
}[MODE]
EPOCHS, SEEDS, BATCH_SIZE, PATIENCE, CACHE = (PROTOCOL[k] for k in ("epochs", "seeds", "batch_size", "patience", "cache"))
LR, WEIGHT_DECAY, GRAD_CLIP = 2e-4, 1e-4, 1.0
SPLIT_SEED = 42                               # split is fixed for ALL seeds / configs / models

# Prefer SEMICON_DATA only when it has the paired folder layout. A stale environment
# variable (for example, an old data/KLA path) should not hide data extracted in ./train.
_CANDIDATE_ROOTS = [os.environ.get("SEMICON_DATA"), PROJECT, os.path.join(PROJECT, "data", "KLA")]
def _has_pair_folders(root):
    if not root:
        return False
    return ((os.path.isdir(os.path.join(root, "train", "train", "GT")) and
             os.path.isdir(os.path.join(root, "train", "train", "NoisyLR"))) or
            (os.path.isdir(os.path.join(root, "GT")) and
             os.path.isdir(os.path.join(root, "NoisyLR"))))
DATA_ROOT = next((os.path.abspath(d) for d in _CANDIDATE_ROOTS if _has_pair_folders(d)),
                 os.path.abspath(os.environ.get("SEMICON_DATA") or PROJECT))
CARINTHIA_ROOT = os.environ.get("SEMICON_CARINTHIA", os.path.join(PROJECT, "data", "Carinthia-S"))
OUT = os.environ.get("SEMICON_RUNS", os.path.join(PROJECT, f"results_{MODE}"))
RUNS, FIGS, TABLES = (os.path.join(OUT, d) for d in ("runs", "figures", "tables"))
for _d in (RUNS, FIGS, TABLES):
    os.makedirs(_d, exist_ok=True)


def find_pair_dirs():
    for gt, ny in [(("train", "train", "GT"), ("train", "train", "NoisyLR")), (("GT",), ("NoisyLR",))]:
        g, n = os.path.join(DATA_ROOT, *gt), os.path.join(DATA_ROOT, *ny)
        if os.path.isdir(g) and os.path.isdir(n):
            return g, n
    raise FileNotFoundError(f"Could not find GT/NoisyLR under SEMICON_DATA={DATA_ROOT}")


def get_pairs():
    from utils.data import match_pairs
    if SMOKE:
        from utils.degrade import make_synthetic_dataset
        root = os.path.join(OUT, "synthetic_data")
        if not os.path.isdir(os.path.join(root, "GT")):
            make_synthetic_dataset(root, n=48, hr=64)
        return match_pairs(os.path.join(root, "GT"), os.path.join(root, "NoisyLR"))
    g, n = find_pair_dirs()
    pairs = match_pairs(g, n)
    if not pairs:
        raise FileNotFoundError(f"GT/NoisyLR folders exist under {DATA_ROOT}, but contain no matching .npy filenames")
    return pairs


def get_splits():
    """-> (splits dict(train/val/test -> pairs), manifest).  The manifest is saved in OUT/split_manifest.json."""
    from utils.data import make_split_manifest, apply_manifest
    pairs = get_pairs()
    man = make_split_manifest(pairs, os.path.join(OUT, "split_manifest.json"), 0.1, 0.1, SPLIT_SEED)
    return apply_manifest(pairs, man), man


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def banner():
    print(f"MODE={MODE}  epochs={EPOCHS}  seeds={SEEDS}  device={device()}  data={DATA_ROOT}  out={OUT}")
