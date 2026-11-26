"""
Statistical tests for the results tables (reviewer comment 1).

Two complementary levels -- report BOTH:
  (A) seed level  : mean +- std over independent training runs (captures optimisation variance);
                    Welch t-test / Mann-Whitney on the per-seed test means.
  (B) image level : paired comparison of two configurations on the SAME test images, using the
                    seed-averaged per-image metric.  Paired t, Wilcoxon signed-rank and a paired
                    bootstrap CI of the mean difference.  (Captures data-sampling variance.)
Multiple comparisons are controlled with Holm-Bonferroni.  Effect size = Cohen's d_z (paired).
"""
import numpy as np
import pandas as pd
from scipy import stats


def mean_std(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else float("nan")


def bootstrap_ci(d, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.RandomState(seed)
    d = np.asarray(d, float)
    idx = rng.randint(0, len(d), (n_boot, len(d)))
    m = d[idx].mean(1)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def paired_image_test(a, b, higher_better=True, n_boot=10000, seed=0):
    """a,b: arrays [n_seeds, n_images] OR [n_images].  Tests H0: mean(a-b)=0.  Returns dict."""
    a, b = np.atleast_2d(a).mean(0), np.atleast_2d(b).mean(0)
    d = a - b
    if not higher_better:
        d = -d
    n = len(d)
    t, p_t = stats.ttest_rel(a, b)
    try:
        w, p_w = stats.wilcoxon(d)
    except ValueError:
        w, p_w = np.nan, 1.0
    lo, hi = bootstrap_ci(d, n_boot, seed=seed)
    dz = d.mean() / (d.std(ddof=1) + 1e-12)
    return {"mean_diff": float(d.mean()), "ci95_lo": lo, "ci95_hi": hi, "p_ttest": float(p_t),
            "p_wilcoxon": float(p_w), "cohens_dz": float(dz), "n_images": n,
            "frac_images_improved": float((d > 0).mean())}


def seed_level_test(x, y):
    """x,y: per-seed test means of two configs."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = {"mean_x": x.mean(), "mean_y": y.mean(), "diff": x.mean() - y.mean(), "n_x": len(x), "n_y": len(y)}
    if len(x) > 1 and len(y) > 1:
        out["p_welch"] = float(stats.ttest_ind(x, y, equal_var=False).pvalue)
        out["p_mannwhitney"] = float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)
    return out


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        val = (len(p) - rank) * p[i]
        run = max(run, val)
        adj[i] = min(1.0, run)
    return adj


def summarize_configs(summary_df, metric="test_psnr", configs=None):
    """mean +- std (over seeds) for each config -> DataFrame."""
    rows = []
    for c, g in summary_df.groupby("config"):
        if configs and c not in configs: continue
        m, s = mean_std(g[metric])
        rows.append({"config": c, "n_seeds": len(g), "mean": m, "std": s, "min": g[metric].min(),
                     "max": g[metric].max(), "params": int(g["params"].iloc[0])})
    out = pd.DataFrame(rows)
    if configs:
        out["o"] = out["config"].map({c: i for i, c in enumerate(configs)}); out = out.sort_values("o").drop(columns="o")
    return out.reset_index(drop=True)


def compare_to_reference(per_image, ref, configs, higher_better=True):
    """per_image: dict name -> [n_seeds, n_images] array.  Compare every config with `ref`, Holm-corrected."""
    rows = []
    for c in configs:
        if c == ref or c not in per_image or per_image[c].size == 0: continue
        r = paired_image_test(per_image[c], per_image[ref], higher_better)
        r["config"], r["vs"] = c, ref
        rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        df["p_wilcoxon_holm"] = holm(df["p_wilcoxon"].values)
        df["p_ttest_holm"] = holm(df["p_ttest"].values)
        df["significant_0.05"] = df["p_wilcoxon_holm"] < 0.05
    return df


def noise_floor(per_image_a, per_image_b):
    """Seed-to-seed spread of the mean metric for a config: what a 'real' effect must exceed."""
    m = np.atleast_2d(per_image_a).mean(1)
    return {"seed_means": m.tolist(), "std_over_seeds": float(m.std(ddof=1)) if len(m) > 1 else float("nan")}


def latex_table(df, cols, fmt="{:.3f}", caption="", label=""):
    head = " & ".join(cols) + r" \\"
    body = "\n".join(" & ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                for v in row) + r" \\" for row in df[cols].itertuples(index=False))
    return (f"\\begin{{table}}[t]\n\\centering\\caption{{{caption}}}\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{{'l' + 'c' * (len(cols) - 1)}}}\n\\hline\n{head}\n\\hline\n{body}\n\\hline\n"
            f"\\end{{tabular}}\n\\end{{table}}")
