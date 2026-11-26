# Manuscript corrections and claim ladder

## A. Factual errors found in the submitted manuscript (verified against the notebooks/figures)

| # | Where | Problem | Fix |
|---|---|---|---|
| 1 | Sec. XI, Table VII, Fig. 10 | "Carinthia-S" images were `data/masks/*.png` (segmentation masks); Fig. 10 shows unrelated photographs | Re-run with verified SEM images (`utils/carinthia.py`, manifest); state DOI/version, N, selection |
| 2 | Table VI | "30 held-out images" = `list_npy(TRAIN_GT_DIR)[:30]`, ~90 % were training images; single noise seed; hard-coded looks=4, σ=0.05 described as "fitted" | Test-split images only, ≥3 noise draws, parameters estimated from train pairs |
| 3 | Table IV | Selection, tuning and reporting on the same 320 validation images; no test set | 80/10/10 manifest; test touched once |
| 4 | Table IV | "Time (min)" column compares wall clock of runs disturbed by sleep/throttling (epochs of 5 000–8 000 s) | Remove, or report FLOPs/epoch-time under identical conditions |
| 5 | Tables IV–V | LDMH-only ≡ Baseline (backbone never uses LDMH output); a 0.019 dB gap was presented as a monotonic step | Report as seed-noise reference; add capacity-matched `film_control` |
| 6 | Table V | Reduced-scale ablation trained with `recon + NLL` only (no diversity/balance), 3 epochs, 20 val images, no seeds, yet labelled "corrected architecture" | Removed; full-scale ablation with corrected loss |
| 7 | Sec. VI-C | β = (1.73, 2.20, 2.50): all components Gaussian-or-lighter, β₃ at the 2.5 bound, min pairwise distance = margin 0.300 exactly → components pinned by the regulariser, none heavy-tailed | State plainly; report `any_beta_at_bound`, `heavy_tailed_component` (notebook 04) |
| 8 | Sec. VI-D | "σ(z) never saturates" — true mathematically, but in float32 the gradient is exactly 0 for l ≥ ≈17 (notebook 02) | Say "practically vanishing"; remove the claim that the collapse fix explains PSNR changes (β, α enter only the auxiliary NLL) |
| 9 | Sec. V-B | Softmax temperature T = 1.5 described, but the submitted `ldmh.py` applies no temperature | Fixed in v2 (tested); disclose |
| 10 | Sec. IV-C | ANOVA groups by image → tests between-image differences, not spatial variation; p < 1e-300 ignores non-independence; features from clean image | Image-clustered CIs, ICC, partial correlations (notebook 01) |
| 11 | Sec. IV | Signal-dependent noise scale never tested as the explanation for heavy tails/β-feature correlations | Heteroscedastic nulls, AIC (notebook 01) — **follow its verdict** |
| 12 | Fig. 2 | Labelled "Noisy (σ = 0.1)" and a 5×4 patch map (implies 160×128 image) — not the KLA degradation/size | Regenerate from data |
| 13 | Fig. 3 | Caption (β vs edge/variance/entropy, 51 200 patches) ≠ content (β vs pixel value, σ = 0.1) | Regenerate |
| 14 | Fig. 4 | Shows bilinear upsample of π/logits added to the output; Eq. 6/code upsample the **input** y | Correct the diagram |
| 15 | Fig. 5 | β = 0.401/1.086/2.461 from a 3-epoch model; "dominant component" panel is a single colour, contradicting the caption's "genuine per-pixel routing" | Regenerate; caption from provenance JSON |
| 16 | Fig. 6 | Curves plateau ≈ 28.85 dB / 0.776, Table IV says 28.206 / 0.7533 | Regenerate from the table's run |
| 17 | Fig. 11 | Panel "LDMH (Ours)" 30.14 dB vs "Baseline (Full Model)" 30.24 dB (names mixed; baseline better); caption says "loss functions"; image likely a training image; 3-epoch checkpoints | Regenerate with fixed rule (median/worst test image) |
| 18 | Final unlabeled graphic | Illustrative β-map and r = −0.59/−0.68/−0.49 match no reported result | Remove, or label "schematic, illustrative values" |
| 19 | Table II | Full GPU latency 16.22 ms; notebook gives 18.22 ms (throughput 54.89 img/s corresponds to 18.22) | Regenerated with asserted consistency |
| 20 | Sec. VIII/XIII-E | "max 300 epochs"; code ran `EPOCHS = 200` (cosine schedule to 200; early stopping never fired) | State the real protocol |
| 21 | Sec. VIII | SSIM "follows Wang et al." but code used a 7×7 uniform window | Gaussian 11×11 σ = 1.5 (verified vs scikit-image) |
| 22 | Data description | Figures from the KLA data show a bristled, specimen-like grayscale object, not an evident wafer pattern | Look at `fig_data_contact_sheet`; describe the data accurately; rely on Carinthia-S for domain claims |

## B. Condensed abstract (≈ 150 words) — fill the «» from RESULTS_SUMMARY.md

> Restoring noisy, low-resolution semiconductor inspection images is limited by assumptions about the residual distribution. We study whether the residual of a speckle + Gaussian + downsampling degradation is spatially heterogeneous in *shape*, using image-clustered inference and explicit signal-dependent-noise null models, and find «supported / explained by intensity-dependent scale». We propose a Local Distribution Mixture Head (LDMH) that conditions a compact NAFNet-style restorer on a per-pixel mixture of generalized-Gaussian prototypes, and document a gradient-saturation failure of hard-clamped shape parameters with a smooth reparameterization. Under a multi-seed, held-out-test protocol with a capacity-matched control and external baselines (NAFNet, Restormer, SwinIR), the full model changes PSNR by «Δ [CI]» versus the baseline and «Δ [CI]» versus the control; on defect-detection downstream tests it changes defect recall by «…» and false alarms per image by «…». We report which claims the evidence does and does not support.

## C. Claim ladder — write what the pre-registered rules return

| Notebook output | Sentence allowed in the paper |
|---|---|
| 01: hetero null reproduces β-spread and correlations | "The apparent heavy tails and the β–structure correlations are explained by signal-dependent noise scale; we found no evidence of additional shape heterogeneity." Reframe the method as scale-aware modelling and compare against a scale-only model. |
| 01: excess SD > 0 and nulls fall outside the CIs | "Shape heterogeneity exceeds the heteroscedastic-Gaussian null by SD «…»." |
| 04 C2 SUPPORTED | "Gains exceed those of a capacity-matched FiLM control (Δ «…», CI «…», p_Holm «…»)." |
| 04 C2 NOT SUPPORTED | "The improvement is not distinguishable from that of an equal-capacity conditioning module without distribution modelling; we therefore do not attribute it to modelling residual distributions." |
| 04 C4 shows a gap ≈ seed noise | "LDMH-only is functionally identical to the baseline; its difference to the baseline (Δ «…») characterises seed variability." |
| 05: external model ≥ ours at equal size | "An equal-size <model> is better / equivalent; no advantage beyond backbone choice is established." |
| 06/07 speckle-only or downstream gain NOT SUPPORTED | Do not claim a speckle-specific or defect-detection benefit. |
| 02 recovery fails | Add to limitations: "On a synthetic known-partition test, LDMH recovers prototype shapes but only «x» % of the spatial partition (a local-std baseline reaches «y» %)." |

## D. Integrity notes

* Quote real numbers only from `RESULTS_SUMMARY.md` / CSVs / figure provenance JSONs.
* Disclose in the resubmission cover letter that earlier figures and tables were regenerated after an internal audit, and which ones.
* State that NAFNet/Restormer/SwinIR are re-implementations unless the official code is added.
* A smaller, accurately-scoped paper (diagnostic of signal-dependent noise + the negative/neutral results) is preferable to defending claims the corrected protocol does not support.
