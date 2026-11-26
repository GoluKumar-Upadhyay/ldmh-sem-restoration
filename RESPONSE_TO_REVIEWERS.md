# Response to Reviewer 1 — point-by-point (template)

> **Status:** structure and evidence mapping are final; every `«…»` must be filled **from `results_full/RESULTS_SUMMARY.md`** after the full run. No result is pre-written. Where a pre-registered rule in the notebooks returns *NOT SUPPORTED*, the manuscript text must say so (see MANUSCRIPT_CORRECTIONS.md §C).

Overall comment — we thank the reviewer. While addressing the points we audited our own code and figures and found additional problems (listed in the cover note). We corrected all of them; several conclusions of the original manuscript did not survive the corrected protocol and have been rewritten accordingly.

---

## 1. Single run; need multiple seeds, mean/std and statistical tests

**Change.** Every configuration is now trained with **«5» independent seeds** under one identical protocol (`notebooks/03`, `utils/experiment.py`). The split is a fixed three-way manifest (train/val/test = 80/10/10); validation is used only for early stopping and checkpoint selection, the test split is evaluated once per run.
**Statistics** (`utils/stats.py`, notebook 04): (a) mean ± std over seeds; (b) paired image-level tests on the same test images — paired t, Wilcoxon, paired-bootstrap 95 % CI, Cohen's d_z — **Holm-corrected** within each (metric, reference) family.
**Result.** Full vs Baseline: «Δ PSNR = … dB, 95 % CI […], p_Holm = …». Seed-to-seed std: «…» dB; a functionally identical pair (LDMH-loss-only vs Baseline; the backbone never sees the LDMH output) differs by «…» dB, which bounds the pure initialisation effect.
**Evidence:** `tables/table_ablation_main.csv`, `table_ablation_paired_tests.csv`, `claim_checks.json`.

## 2. Out-of-distribution images (Carinthia-S) do not correspond to SEM wafer images

**The reviewer is correct.** Our notebook took the first ten files of the archive, which were `data/masks/*.png` — segmentation masks — and Fig. 10 showed unrelated photographs. We withdraw Table VII and Fig. 10.
**Change.** `utils/carinthia.py` separates images from masks by directory, **refuses label-map-like files**, selects images by a seeded random sample and writes a manifest (file name, SHA-256 prefix, original shape, seed) → `tables/carinthia_manifest.json`. The experiment is re-run on **«N»** verified Carinthia-S images («DOI / version»); degradation strengths are **estimated from the training pairs** (looks «…», σ «…»; notebook 00) instead of being hard-coded.
**Result.** `tables/table_ood.csv`, `table_ood_paired.csv`, `figures/fig_ood_qualitative.*` (+ provenance JSON naming the file shown).

## 3. Ablation inconsistent with full-scale conclusions

**Change.** The reduced-scale ablation (200 images, 3 epochs, no penalties) is **removed**. The ablation is re-run at full scale under the same protocol (same epochs, split, seeds) with the **corrected loss** (diversity + load-balance). We added the control that was missing: **`film_control`** — identical encoder/FiLM capacity, trained *without* the mixture NLL — so the effect of distribution modelling can be separated from extra parameters. Additional rows: no-penalty, no-balance, no-diversity, and the original **clamp** parameterisation.
**Result.** Full vs FiLM-control: «…». Claim "each mechanism brings further improvement": «SUPPORTED / NOT SUPPORTED by the pre-registered rule». Note: the LDMH-only row is functionally the baseline (the backbone does not use the LDMH output); it is reported only as a seed-noise reference.
**Evidence:** `table_ablation_*.csv`, `table_component_health.csv`, `claim_checks.json`.

## 4. Manufacturing relevance; downstream evaluation

**Change.** (i) Full-scale **HP-Err** (top-5 % highest-error pixels), **edge correlation**, **edge F1**, and **flat-region artefact error** for every model (`utils/metrics.py`). (ii) A **downstream defect-detection protocol** (notebook 07): a segmenter trained on clean Carinthia-S images and frozen is run on images restored by each model; we report pixel precision/recall/F1/IoU, **defect-level recall (miss rate), precision and false alarms per image**, and defect **CNR** relative to the clean image, with clean (upper bound) and bilinear (lower bound) reference arms and paired tests.
**Result.** `tables/table_downstream.csv`, `table_downstream_paired.csv`. Statement for the paper: «Full vs Baseline on defect recall: …; false alarms/image: …».
**Limitations to state:** restoration models trained on KLA data, evaluated on Carinthia-S (domain shift); small segmenter; component-based defect matching with a 10 % overlap rule.

## 5. Comparison with external restoration methods

**Change.** NAFNet, Restormer and SwinIR (`models/baselines.py`) are trained from scratch with the **same split, augmentation, optimiser, schedule, budget, early-stopping rule and seeds** as the proposed model; `small` variants match the proposed parameter budget (~0.1 M), `base` variants are larger. Parameters and FLOPs (torch FlopCounterMode) are reported. **Disclosure:** these are compact re-implementations, not the official code; «official implementations added: yes/no».
**Result.** `tables/table_external_main.csv`, `table_external_paired_tests.csv`. Statement: «At comparable size, <model> is better / indistinguishable / worse than the proposed model (Δ …, CI …)». If an external model is better or indistinguishable, the paper states that no advantage of the distribution-adaptive mechanism beyond backbone choice is demonstrated.

## 6. Writing, figures and tables

* **Fig. 3 caption/content mismatch** — confirmed (the figure was a β-vs-pixel-value plot). Replaced by `fig_beta_vs_structure_observed` with image-clustered correlations.
* **Fig. 5 β values (0.401/1.086/2.461 vs 1.73/2.20/2.50)** — confirmed: the figure had been produced from a 3-epoch reduced-scale checkpoint, not the model in the text. All figures are regenerated from the final checkpoints with a **provenance JSON** (checkpoint, split, image id, seed, the numbers shown); the quoted values are copied from it.
* Additional inconsistencies we found and fixed are listed in MANUSCRIPT_CORRECTIONS.md §A.
* Abstract condensed to ≈150 words (MANUSCRIPT_CORRECTIONS.md §B).
* All figures: 300 dpi PNG + vector PDF, larger fonts, colour-blind-safe palette, error bars defined in the caption.

---

## Additional corrections we made (not raised by the reviewer)

Held-out claim in Table VI (the 30 images were the first 30 files of the whole training folder); ANOVA grouped by image not location; clean-image features; untested signal-dependent-noise alternative (notebook 01); softmax temperature T = 1.5 stated but not applied in the submitted code; SSIM implementation (7×7 uniform window, not Wang et al.); epoch budget (200 in code vs 300 in text); training-time column (non-comparable wall clock); Table II latency typo (16.22 ms vs 18.22 ms); unlabeled graphical abstract with illustrative r-values; no held-out test split.
