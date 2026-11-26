# semicon_v2 — corrected, reviewer-driven experiment suite

Replaces the original notebooks/utils/models. It does **not** contain results: the KLA data were not available where this was written, so everything was verified end-to-end on **synthetic data** (`scripts/smoke_test.py`: all 10 notebooks execute). Real numbers appear only when you run it on your data. The code is written to report whatever the evidence says (pre-registered decision rules print SUPPORTED / NOT SUPPORTED).

## Setup
```bash
pip install -r requirements.txt
export SEMICON_DATA=/path/to/KLA        # contains train/train/GT and train/train/NoisyLR  (or GT/ and NoisyLR/)
export SEMICON_CARINTHIA=/path/to/Carinthia-S   # unzipped; images + masks (Zenodo 16895427 — confirm version/DOI)
export SEMICON_MODE=pilot               # smoke | pilot | full
cd notebooks && jupyter lab
```
Windows PowerShell: `$env:SEMICON_MODE="pilot"` etc. Outputs go to `results_<mode>/` (`runs/`, `figures/`, `tables/`, `RESULTS_SUMMARY.md`).

## Recommended order (with stop rules)
1. `00_Setup_and_Data_Audit` — split manifest, residual-definition check, contact sheet, **estimated degradation parameters**.
2. `01_Statistical_Analysis` — **decision gate 1.** If it returns *NOT SUPPORTED* (signal-dependent noise explains the heavy tails), the paper's premise must be reframed before spending GPU time.
3. `02_LDMH_Unit_Tests`.
4. `SEMICON_MODE=pilot` → `03_Train_All_Models` (core group, 30 epochs, 2 seeds) → `04_Ablation_and_Statistics` — **decision gate 2.** If `full` does not beat `film_control`, stop and rethink the claim.
5. `SEMICON_MODE=full` → `python scripts/run_experiments.py --mode full --groups core,ablation,external` (resumable; run overnight / on a cloud GPU) → notebooks 04, 05, 06, 07, 08, 09.

| Notebook | Reviewer comment |
|---|---|
| 00 Setup & audit | 1, 2, 4 (foundation) |
| 01 Statistical analysis | motivation (Sec. IV) |
| 02 LDMH unit tests | Sec. VI claims, module validity |
| 03 Train all | 1, 3, 5 |
| 04 Ablation & statistics | **1, 3**, 4 (HP-Err, edge) |
| 05 External baselines | **5** |
| 06 Robustness & OOD | **1, 2** |
| 07 Downstream defects | **4** |
| 08 Efficiency | 6 (Table II) |
| 09 Figures & summary | **6** |

## Layout
```
config.py                  paths + protocol (modes)
models/   backbone.py  ldmh.py  restoration_net.py  baselines.py (NAFNet/Restormer/SwinIR, re-implemented)
utils/    data  metrics  losses  experiment  stats  degrade  distfit  carinthia  downstream  analysis  figures
scripts/  run_experiments.py  smoke_test.py  make_notebooks.py (+ defs_*.py: notebook sources)
notebooks/ 00 … 09
RESPONSE_TO_REVIEWERS.md   MANUSCRIPT_CORRECTIONS.md
```
Edit notebook content in `scripts/defs_*.py` and run `python scripts/make_notebooks.py` to regenerate.

## What changed vs the submitted code (all tested)
* three-way split manifest; test split touched once; reproducible augmentation; RAM cache
* `film_control` (capacity-matched, no NLL) + penalty / clamp ablations; `ldmh_loss_only` kept only as seed-noise reference
* `ldmh.py`: temperature actually applied; scale initialised near the empirical scale; fp32 NLL; no weight decay on prototypes; bound-saturation diagnostic; `beta_mode='clamp'` to *test* the collapse claim
* SSIM = Gaussian 11×11 (matches scikit-image to 6 decimals); HP-Err, edge correlation/F1, flat-region error, CNR
* degradation parameters estimated from data; verified Carinthia loader that rejects masks
* paired tests, bootstrap CIs, Holm correction, effect sizes; provenance JSON for every figure

## Honest limitations
* Not run on real data here — no numbers are claimed. Smoke-test results on synthetic data are meaningless.
* NAFNet/Restormer/SwinIR are compact re-implementations, not official code (disclose; add official versions via `BASELINE_REGISTRY` if possible).
* Timing: your earlier logs show ≈70–155 s/epoch on an RTX 3050. A full run (≈15 configs × 5 seeds × 200 epochs) is on the order of 100+ GPU-hours on that laptop — use `--seeds 0,1,2`, fewer epochs (check convergence), or a cloud GPU; prioritise the `core` group.
* The downstream segmenter is small and the restorers are applied across a domain shift (KLA → Carinthia-S); consider fine-tuning on Carinthia-S train images and say so.
* The moment-matching degradation fit assumes independent per-pixel speckle; check `r2_signal_dependence` before trusting looks/σ.
