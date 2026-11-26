# Results summary (auto-generated; copy numbers from here)

mode=pilot epochs=30 seeds=[0, 1]


## data_audit.csv

| Unnamed: 0   |   gt_min |   gt_max |   noisy_min |   noisy_max |   frac_noisy_below0 |   frac_noisy_above1 |   res_mean |   res_std |
|:-------------|---------:|---------:|------------:|------------:|--------------------:|--------------------:|-----------:|----------:|
| count        |      200 |      200 |    200      |    200      |            200      |            200      |   200      |  200      |
| mean         |        0 |        1 |     -0.0056 |      1.3896 |              0.0033 |              0.0358 |    -0      |    0.0832 |
| std          |        0 |        0 |      0.0236 |      0.1993 |              0.0184 |              0.065  |     0.0007 |    0.031  |
| min          |        0 |        1 |     -0.1268 |      0.7221 |              0      |              0      |    -0.0024 |    0.0114 |
| 25%          |        0 |        1 |     -0.0126 |      1.2641 |              0      |              0.0016 |    -0.0004 |    0.0619 |
| 50%          |        0 |        1 |     -0.0021 |      1.4027 |              0.0001 |              0.0086 |    -0      |    0.0824 |
| 75%          |        0 |        1 |      0.005  |      1.5405 |              0.0003 |              0.0312 |     0.0004 |    0.103  |
| max          |        0 |        1 |      0.1115 |      1.9384 |              0.2163 |              0.3748 |     0.0018 |    0.1636 |



## ldmh_synthetic_recovery.csv

|   seed |   agreement |   local_std_baseline |   beta_lo |   beta_hi |   scale_lo |   scale_hi |
|-------:|------------:|---------------------:|----------:|----------:|-----------:|-----------:|
|      0 |      0.878  |               0.9704 |    0.4928 |    2.2451 |     0.0271 |     0.0537 |
|      1 |      0.7156 |               0.9569 |    0.4995 |    2.3005 |     0.0278 |     0.0534 |
|      2 |      0.8406 |               0.9703 |    0.4922 |    2.1191 |     0.0271 |     0.0514 |
|      3 |      0.8406 |               0.9632 |    0.4898 |    2.2117 |     0.0282 |     0.0541 |
|      4 |      0.7123 |               0.9701 |    0.4961 |    2.2289 |     0.0284 |     0.0538 |



## paired_tests_vs_baseline.csv

|   mean_diff |   ci95_lo |   ci95_hi |   p_ttest |   p_wilcoxon |   cohens_dz |   n_images |   frac_images_improved | config       | vs       |   p_wilcoxon_holm |   p_ttest_holm | significant_0.05   |
|------------:|----------:|----------:|----------:|-------------:|------------:|-----------:|-----------------------:|:-------------|:---------|------------------:|---------------:|:-------------------|
|     -0.0163 |   -0.0245 |   -0.0091 |         0 |            0 |     -0.2343 |        320 |                 0.3531 | global_gc    | baseline |                 0 |              0 | True               |
|      0.0732 |    0.0612 |    0.0862 |         0 |            0 |      0.6345 |        320 |                 0.8375 | film_control | baseline |                 0 |              0 | True               |
|      0.1745 |    0.1567 |    0.1938 |         0 |            0 |      1.0248 |        320 |                 0.9688 | full         | baseline |                 0 |              0 | True               |



## paired_tests_vs_film_control.csv

|   mean_diff |   ci95_lo |   ci95_hi |   p_ttest |   p_wilcoxon |   cohens_dz |   n_images |   frac_images_improved | config    | vs           |   p_wilcoxon_holm |   p_ttest_holm | significant_0.05   |
|------------:|----------:|----------:|----------:|-------------:|------------:|-----------:|-----------------------:|:----------|:-------------|------------------:|---------------:|:-------------------|
|     -0.0732 |   -0.0862 |   -0.0612 |         0 |            0 |     -0.6345 |        320 |                 0.1625 | baseline  | film_control |                 0 |              0 | True               |
|     -0.0895 |   -0.1041 |   -0.076  |         0 |            0 |     -0.6889 |        320 |                 0.1281 | global_gc | film_control |                 0 |              0 | True               |
|      0.1014 |    0.087  |    0.1158 |         0 |            0 |      0.7679 |        320 |                 0.8688 | full      | film_control |                 0 |              0 | True               |



## residual_definition_check.csv

| Unnamed: 0   |   dx=-1 |   dx=0 |   dx=1 |
|:-------------|--------:|-------:|-------:|
| dy=-1        |  0.1207 | 0.1085 | 0.1225 |
| dy=0         |  0.1096 | 0.0833 | 0.1102 |
| dy=1         |  0.122  | 0.1082 | 0.1203 |



## section4_observed_vs_null.csv

| table                |   beta_mean |   beta_std |   r(beta,edge_density) |   r(beta,gt_var) |   r(beta,entropy) |
|:---------------------|------------:|-----------:|-----------------------:|-----------------:|------------------:|
| observed             |      1.4504 |     0.4449 |                -0.3726 |          -0.5033 |           -0.4198 |
| homogeneous GGD null |      0.8051 |     0.0526 |                -0.0445 |          -0.0295 |           -0.0276 |
| hetero-Gaussian null |      1.4413 |     0.4519 |                -0.4802 |          -0.562  |           -0.4958 |



## summary_by_model.csv

| config       |   params |   test_psnr_n_seeds |   test_psnr_mean |   test_psnr_std |   test_ssim_n_seeds |   test_ssim_mean |   test_ssim_std |   test_hp_err_n_seeds |   test_hp_err_mean |   test_hp_err_std |
|:-------------|---------:|--------------------:|-----------------:|----------------:|--------------------:|-----------------:|----------------:|----------------------:|-------------------:|------------------:|
| baseline     |    86945 |                   2 |          27.6118 |          0.0513 |                   2 |           0.7278 |          0.0017 |                     2 |             0.1078 |            0.0008 |
| global_gc    |    86945 |                   2 |          27.5955 |          0.0491 |                   2 |           0.7276 |          0.0016 |                     2 |             0.108  |            0.0007 |
| film_control |   116138 |                   2 |          27.685  |          0.0505 |                   2 |           0.7299 |          0.0013 |                     2 |             0.1068 |            0.0008 |
| full         |   116138 |                   2 |          27.7863 |          0.0097 |                   2 |           0.7337 |          0.0002 |                     2 |             0.1055 |            0.0002 |



## tableI_global_model_comparison.csv

| model                        |   k |      loglik |          AIC |    dAIC_vs_best |
|:-----------------------------|----:|------------:|-------------:|----------------:|
| Hetero-GGD (beta=1.53)       |   3 | 4.56298e+06 | -9.12596e+06 |     0           |
| Hetero-Gaussian  N(0,aI^2+b) |   2 | 4.52826e+06 | -9.05651e+06 | 69456.7         |
| GGD (global)                 |   2 | 3.67545e+06 | -7.3509e+06  |     1.77506e+06 |
| Laplace                      |   1 | 3.65122e+06 | -7.30243e+06 |     1.82353e+06 |
| Student-t                    |   2 | 3.5806e+06  | -7.1612e+06  |     1.96476e+06 |
| Gaussian                     |   1 | 3.2848e+06  | -6.5696e+06  |     2.55636e+06 |



## table_ablation_main.csv

| Unnamed: 0   |   params |   n_seeds | PSNR              | SSIM             | HP-Err (top-5%)   | Edge corr        | Edge F1          | Flat-region err   |
|:-------------|---------:|----------:|:------------------|:-----------------|:------------------|:-----------------|:-----------------|:------------------|
| baseline     |    86945 |         4 | 27.6459 +- 0.0553 | 0.7290 +- 0.0019 | 0.1074 +- 0.0008  | 0.8523 +- 0.0011 | 0.5966 +- 0.0016 | 0.0233 +- 0.0002  |
| global_gc    |    86945 |         2 | 27.5955 +- 0.0491 | 0.7276 +- 0.0016 | 0.1080 +- 0.0007  | 0.8512 +- 0.0005 | 0.5954 +- 0.0011 | 0.0234 +- 0.0002  |
| film_control |   116138 |         4 | 27.6908 +- 0.0468 | 0.7295 +- 0.0013 | 0.1067 +- 0.0008  | 0.8535 +- 0.0011 | 0.5986 +- 0.0019 | 0.0233 +- 0.0002  |
| full         |   116138 |         4 | 27.7690 +- 0.0278 | 0.7331 +- 0.0007 | 0.1057 +- 0.0004  | 0.8557 +- 0.0005 | 0.6011 +- 0.0009 | 0.0229 +- 0.0001  |



## table_ablation_paired_tests.csv

|   mean_diff |   ci95_lo |   ci95_hi |   p_ttest |   p_wilcoxon |   cohens_dz |   n_images |   frac_images_improved | config       | vs           |   p_wilcoxon_holm |   p_ttest_holm | significant_0.05   | metric    |
|------------:|----------:|----------:|----------:|-------------:|------------:|-----------:|-----------------------:|:-------------|:-------------|------------------:|---------------:|:-------------------|:----------|
|     -0.0505 |   -0.0585 |   -0.0431 |    0      |       0      |     -0.7176 |        320 |                 0.1094 | global_gc    | baseline     |            0      |         0      | True               | psnr      |
|      0.0449 |    0.0341 |    0.0566 |    0      |       0      |      0.4376 |        320 |                 0.7781 | film_control | baseline     |            0      |         0      | True               | psnr      |
|      0.1231 |    0.1102 |    0.1366 |    0      |       0      |      1.0034 |        320 |                 0.9531 | full         | baseline     |            0      |         0      | True               | psnr      |
|     -0.0449 |   -0.0566 |   -0.0341 |    0      |       0      |     -0.4376 |        320 |                 0.2219 | baseline     | film_control |            0      |         0      | True               | psnr      |
|     -0.0954 |   -0.1105 |   -0.0817 |    0      |       0      |     -0.7277 |        320 |                 0.1344 | global_gc    | film_control |            0      |         0      | True               | psnr      |
|      0.0781 |    0.0666 |    0.0898 |    0      |       0      |      0.7428 |        320 |                 0.8625 | full         | film_control |            0      |         0      | True               | psnr      |
|     -0.0013 |   -0.0016 |   -0.0011 |    0      |       0      |     -0.5566 |        320 |                 0.2625 | global_gc    | baseline     |            0      |         0      | True               | ssim      |
|      0.0006 |    0.0001 |    0.001  |    0.0242 |       0      |      0.1266 |        320 |                 0.6312 | film_control | baseline     |            0      |         0.0242 | True               | ssim      |
|      0.0041 |    0.0036 |    0.0046 |    0      |       0      |      0.9174 |        320 |                 0.875  | full         | baseline     |            0      |         0      | True               | ssim      |
|     -0.0006 |   -0.001  |   -0.0001 |    0.0242 |       0      |     -0.1266 |        320 |                 0.3688 | baseline     | film_control |            0      |         0.0242 | True               | ssim      |
|     -0.0019 |   -0.0024 |   -0.0014 |    0      |       0      |     -0.4117 |        320 |                 0.2625 | global_gc    | film_control |            0      |         0      | True               | ssim      |
|      0.0036 |    0.0029 |    0.0043 |    0      |       0      |      0.5863 |        320 |                 0.7625 | full         | film_control |            0      |         0      | True               | ssim      |
|     -0.0006 |   -0.0007 |   -0.0005 |    0      |       0      |     -0.5493 |        320 |                 0.2469 | global_gc    | baseline     |            0      |         0      | True               | hp_err    |
|      0.0007 |    0.0006 |    0.0009 |    0      |       0      |      0.5437 |        320 |                 0.775  | film_control | baseline     |            0      |         0      | True               | hp_err    |
|      0.0017 |    0.0015 |    0.0019 |    0      |       0      |      0.7749 |        320 |                 0.8781 | full         | baseline     |            0      |         0      | True               | hp_err    |
|     -0.0007 |   -0.0009 |   -0.0006 |    0      |       0      |     -0.5437 |        320 |                 0.225  | baseline     | film_control |            0      |         0      | True               | hp_err    |
|     -0.0013 |   -0.0015 |   -0.0012 |    0      |       0      |     -0.9089 |        320 |                 0.1    | global_gc    | film_control |            0      |         0      | True               | hp_err    |
|      0.0009 |    0.0007 |    0.0012 |    0      |       0      |      0.3695 |        320 |                 0.6469 | full         | film_control |            0      |         0      | True               | hp_err    |
|     -0.0012 |   -0.0015 |   -0.0009 |    0      |       0      |     -0.5007 |        320 |                 0.225  | global_gc    | baseline     |            0      |         0      | True               | edge_corr |
|      0.0012 |    0.0008 |    0.0015 |    0      |       0      |      0.3586 |        320 |                 0.6781 | film_control | baseline     |            0      |         0      | True               | edge_corr |
|      0.0034 |    0.003  |    0.0038 |    0      |       0      |      0.923  |        320 |                 0.9219 | full         | baseline     |            0      |         0      | True               | edge_corr |
|     -0.0012 |   -0.0015 |   -0.0008 |    0      |       0      |     -0.3586 |        320 |                 0.3219 | baseline     | film_control |            0      |         0      | True               | edge_corr |
|     -0.0023 |   -0.0029 |   -0.0019 |    0      |       0      |     -0.5292 |        320 |                 0.1844 | global_gc    | film_control |            0      |         0      | True               | edge_corr |
|      0.0022 |    0.0018 |    0.0026 |    0      |       0      |      0.5897 |        320 |                 0.7906 | full         | film_control |            0      |         0      | True               | edge_corr |
|     -0.0001 |   -0.0001 |   -0.0001 |    0      |       0      |     -0.3879 |        320 |                 0.2812 | global_gc    | baseline     |            0      |         0      | True               | flat_err  |
|      0      |    0      |    0.0001 |    0.0495 |       0.0022 |      0.1102 |        320 |                 0.5688 | film_control | baseline     |            0.0022 |         0.0495 | True               | flat_err  |
|      0.0004 |    0.0004 |    0.0005 |    0      |       0      |      0.892  |        320 |                 0.9156 | full         | baseline     |            0      |         0      | True               | flat_err  |
|     -0      |   -0.0001 |   -0      |    0.0495 |       0.0022 |     -0.1102 |        320 |                 0.4312 | baseline     | film_control |            0.0022 |         0.0495 | True               | flat_err  |
|     -0.0001 |   -0.0002 |   -0.0001 |    0      |       0      |     -0.3092 |        320 |                 0.3375 | global_gc    | film_control |            0      |         0      | True               | flat_err  |
|      0.0004 |    0.0003 |    0.0004 |    0      |       0      |      0.6324 |        320 |                 0.8594 | full         | film_control |            0      |         0      | True               | flat_err  |



## table_component_health.csv

| config   |   seed | betas                                                     | usage                                                     |   min_beta_dist | any_beta_at_bound   |   min_usage | all_betas_gt_1   | heavy_tailed_component   |
|:---------|-------:|:----------------------------------------------------------|:----------------------------------------------------------|----------------:|:--------------------|------------:|:-----------------|:-------------------------|
| full     |      0 | [np.float64(0.504), np.float64(1.554), np.float64(2.38)]  | [np.float64(0.194), np.float64(0.443), np.float64(0.363)] |           0.826 | False               |       0.194 | False            | True                     |
| full     |      1 | [np.float64(0.503), np.float64(1.548), np.float64(2.385)] | [np.float64(0.189), np.float64(0.444), np.float64(0.367)] |           0.837 | False               |       0.189 | False            | True                     |
| full     |      2 | [np.float64(0.504), np.float64(1.558), np.float64(2.377)] | [np.float64(0.193), np.float64(0.443), np.float64(0.364)] |           0.819 | False               |       0.193 | False            | True                     |
| full     |      3 | [np.float64(0.504), np.float64(1.553), np.float64(2.386)] | [np.float64(0.192), np.float64(0.446), np.float64(0.362)] |           0.833 | False               |       0.192 | False            | True                     |



## table_efficiency.csv

| config          |   params |   GFLOPs |   cpu_b1_ms |   cpu_b1_img_s |   cpu_b16_ms |   cpu_b16_img_s |
|:----------------|---------:|---------:|------------:|---------------:|-------------:|----------------:|
| baseline        |    86945 |    3.898 |      267.09 |            3.7 |      4932.81 |             3.2 |
| film_control    |   116138 |    4.844 |      410.31 |            2.4 |      6459.21 |             2.5 |
| full            |   116138 |    4.844 |      395.23 |            2.5 |      6699.66 |             2.4 |
| NAFNet-small    |   112817 |    0.728 |       74.72 |           13.4 |      1353.86 |            11.8 |
| Restormer-small |   111917 |    3.866 |      289.01 |            3.5 |      5722.29 |             2.8 |
| SwinIR-small    |   100813 |    3.973 |      270.5  |            3.7 |      4423.93 |             3.6 |
| NAFNet-base     |   247945 |    1.595 |       98.16 |           10.2 |      1808.88 |             8.8 |
| Restormer-base  |   294129 |    9.923 |      637.74 |            1.6 |     12712.8  |             1.3 |
| SwinIR-base     |   263713 |    9.911 |      562.5  |            1.8 |     11027    |             1.5 |



## table_external_main.csv

| config   |   params |   GFLOPs_128 |   n_seeds | PSNR            | SSIM             |   HP_Err |   edge_corr |   epochs_run |
|:---------|---------:|-------------:|----------:|:----------------|:-----------------|---------:|------------:|-------------:|
| baseline |    86945 |         3.9  |         4 | 27.646 +- 0.055 | 0.7290 +- 0.0019 |   0.1074 |      0.8523 |           30 |
| full     |   116138 |         4.84 |         4 | 27.769 +- 0.028 | 0.7331 +- 0.0007 |   0.1057 |      0.8557 |           30 |



## table_external_paired_tests.csv

|   mean_diff |   ci95_lo |   ci95_hi |   p_ttest |   p_wilcoxon |   cohens_dz |   n_images |   frac_images_improved | config   | vs       |   p_wilcoxon_holm |   p_ttest_holm | significant_0.05   | metric   |
|------------:|----------:|----------:|----------:|-------------:|------------:|-----------:|-----------------------:|:---------|:---------|------------------:|---------------:|:-------------------|:---------|
|     -0.1231 |   -0.1366 |   -0.1102 |         0 |            0 |     -1.0034 |        320 |                 0.0469 | baseline | full     |                 0 |              0 | True               | psnr     |
|      0.1231 |    0.1102 |    0.1366 |         0 |            0 |      1.0034 |        320 |                 0.9531 | full     | baseline |                 0 |              0 | True               | psnr     |



## table_robustness_paired_tests.csv

|   mean_diff |   ci95_lo |   ci95_hi |   p_ttest |   p_wilcoxon |   cohens_dz |   n_images |   frac_images_improved | config       | vs           |   p_wilcoxon_holm |   p_ttest_holm | significant_0.05   | condition       |   p_wilcoxon_holm_global |
|------------:|----------:|----------:|----------:|-------------:|------------:|-----------:|-----------------------:|:-------------|:-------------|------------------:|---------------:|:-------------------|:----------------|-------------------------:|
|     -0.0393 |   -0.0444 |   -0.034  |    0      |       0      |     -0.8098 |        320 |                 0.1281 | global_gc    | baseline     |            0      |         0      | True               | combined        |                   0      |
|      0.0215 |    0.0097 |    0.0342 |    0.0007 |       0.0109 |      0.1924 |        320 |                 0.5656 | film_control | baseline     |            0.0109 |         0.0007 | True               | combined        |                   0.0653 |
|      0.0904 |    0.0783 |    0.1028 |    0      |       0      |      0.7989 |        320 |                 0.9031 | full         | baseline     |            0      |         0      | True               | combined        |                   0      |
|     -0.0215 |   -0.0342 |   -0.0097 |    0.0007 |       0.0109 |     -0.1924 |        320 |                 0.4344 | baseline     | film_control |            0.0109 |         0.0007 | True               | combined        |                   0.0653 |
|     -0.0608 |   -0.0741 |   -0.0486 |    0      |       0      |     -0.522  |        320 |                 0.2375 | global_gc    | film_control |            0      |         0      | True               | combined        |                   0      |
|      0.0689 |    0.0552 |    0.0823 |    0      |       0      |      0.556  |        320 |                 0.8719 | full         | film_control |            0      |         0      | True               | combined        |                   0      |
|     -0.1635 |   -0.1934 |   -0.1357 |    0      |       0      |     -0.6237 |        320 |                 0.2    | global_gc    | baseline     |            0      |         0      | True               | downsample only |                   0      |
|      0.0557 |    0.0311 |    0.0814 |    0      |       0.0007 |      0.2424 |        320 |                 0.5656 | film_control | baseline     |            0.0013 |         0      | True               | downsample only |                   0.0067 |
|     -0.0798 |   -0.1692 |    0.004  |    0.0736 |       0.1106 |     -0.1003 |        320 |                 0.575  | full         | baseline     |            0.1106 |         0.0736 | False              | downsample only |                   0.4422 |
|     -0.0557 |   -0.0814 |   -0.0311 |    0      |       0.0007 |     -0.2424 |        320 |                 0.4344 | baseline     | film_control |            0.0013 |         0      | True               | downsample only |                   0.0067 |
|     -0.2191 |   -0.2599 |   -0.181  |    0      |       0      |     -0.605  |        320 |                 0.2594 | global_gc    | film_control |            0      |         0      | True               | downsample only |                   0      |
|     -0.1355 |   -0.2301 |   -0.0476 |    0.004  |       0.1995 |     -0.1619 |        320 |                 0.6062 | full         | film_control |            0.1995 |         0.004  | False              | downsample only |                   0.5985 |
|     -0.1419 |   -0.1677 |   -0.1179 |    0      |       0      |     -0.6247 |        320 |                 0.1938 | global_gc    | baseline     |            0      |         0      | True               | gaussian only   |                   0      |
|      0.0229 |    0.0026 |    0.0436 |    0.0285 |       0.2442 |      0.123  |        320 |                 0.5062 | film_control | baseline     |            0.2442 |         0.0569 | False              | gaussian only   |                   0.5985 |
|      0.0273 |   -0.0445 |    0.0942 |    0.4401 |       0      |      0.0432 |        320 |                 0.6406 | full         | baseline     |            0      |         0.4401 | True               | gaussian only   |                   0      |
|     -0.0229 |   -0.0436 |   -0.0026 |    0.0285 |       0.2442 |     -0.123  |        320 |                 0.4938 | baseline     | film_control |            0.2442 |         0.0569 | False              | gaussian only   |                   0.5985 |
|     -0.1647 |   -0.1989 |   -0.1331 |    0      |       0      |     -0.5508 |        320 |                 0.2844 | global_gc    | film_control |            0      |         0      | True               | gaussian only   |                   0      |
|      0.0044 |   -0.07   |    0.0733 |    0.9042 |       0      |      0.0067 |        320 |                 0.6938 | full         | film_control |            0      |         0.9042 | True               | gaussian only   |                   0      |
|     -0.0444 |   -0.0501 |   -0.0387 |    0      |       0      |     -0.8445 |        320 |                 0.1188 | global_gc    | baseline     |            0      |         0      | True               | speckle only    |                   0      |
|      0.0244 |    0.0126 |    0.0371 |    0.0001 |       0.0012 |      0.2183 |        320 |                 0.5844 | film_control | baseline     |            0.0012 |         0.0001 | True               | speckle only    |                   0.0093 |
|      0.0871 |    0.0756 |    0.0992 |    0      |       0      |      0.8047 |        320 |                 0.9062 | full         | baseline     |            0      |         0      | True               | speckle only    |                   0      |
|     -0.0244 |   -0.0371 |   -0.0126 |    0.0001 |       0.0012 |     -0.2183 |        320 |                 0.4156 | baseline     | film_control |            0.0012 |         0.0001 | True               | speckle only    |                   0.0093 |
|     -0.0688 |   -0.0825 |   -0.0563 |    0      |       0      |     -0.5728 |        320 |                 0.225  | global_gc    | film_control |            0      |         0      | True               | speckle only    |                   0      |
|      0.0628 |    0.0494 |    0.0757 |    0      |       0      |      0.5192 |        320 |                 0.8656 | full         | film_control |            0      |         0      | True               | speckle only    |                   0      |



## claim_checks.json

```
{
 "C1  Full > Baseline (PSNR)": "SUPPORTED  (diff=+0.1231, CI [+0.1102,+0.1366], p_holm=7.52e-51)",
 "C2  Full > FiLM-control  (gain is due to distribution modelling, not capacity)": "SUPPORTED  (diff=+0.0781, CI [+0.0666,+0.0898], p_holm=1.47e-38)",
 "C3  Full > Global-GC (PSNR)": "n/a (config or reference not run)",
 "C4  LDMH-loss-only differs from Baseline? (should be NOT SUPPORTED: backbone unaffected)": "n/a (config or reference not run)",
 "C5  Full > Baseline (HP-Err, top-5% pixels; lower is better)": "SUPPORTED  (diff=+0.0017, CI [+0.0015,+0.0019], p_holm=1.36e-42)"
}
```


## degradation_fit.json

```
{
 "looks": 9.259709358215332,
 "sigma": 0.04252737760543823,
 "a": 0.02699868753552437,
 "b": 0.00045214444980956614,
 "r2_signal_dependence": 0.9977861642837524,
 "mean_residual": 8.677723235450685e-05,
 "n_pixels": 2621600,
 "note": "moment-matching estimate; check r2_signal_dependence before trusting looks/sigma"
}
```


## environment.json

```
{
 "torch": "2.11.0+cu130",
 "cuda": true,
 "gpu": "Tesla T4",
 "mode": "pilot",
 "epochs": 30,
 "seeds": [
  0,
  1
 ],
 "batch_size": 8,
 "lr": 0.0002,
 "split_sha1": "31d7258b8c36e2fd1262b33776880d7ec6ed4ce2"
}
```


## ldmh_unit_tests.json

```
{
 "sigmoid_float32_saturation_l": 17.0,
 "recovery_agreement_mean": 0.7974392361111111,
 "local_std_baseline_mean": 0.9661675347222223,
 "recovery_strict_pass": false,
 "recovery_original_criterion_pass": false,
 "beta_recovery_pass": true
}
```


## section4_summary.json

```
{
 "global_best_model": "Hetero-GGD (beta=1.53)",
 "ab": {
  "a": 0.03406723900807617,
  "b": 5.1708286740514505e-05,
  "ggd_beta": 0.7978815716514408,
  "ggd_alpha": 0.04100733841282936,
  "n": 3276800
 },
 "ICC": {
  "ICC": 0.5156882242317752,
  "var_between": 0.10218789091623796,
  "var_within": 0.09597038789353719,
  "patches_per_image": 16.0
 },
 "observed": {
  "edge_density": {
   "r": -0.37262279505088697,
   "ci95": [
    -0.43075039951884103,
    -0.3191514024389909
   ],
   "n_patches": 6400,
   "n_images": 400
  },
  "gt_var": {
   "r": -0.5032841686048142,
   "ci95": [
    -0.5392471199895542,
    -0.472067749624866
   ],
   "n_patches": 6400,
   "n_images": 400
  },
  "entropy": {
   "r": -0.41979560225655704,
   "ci95": [
    -0.46675660639047045,
    -0.3743446944109743
   ],
   "n_patches": 6400,
   "n_images": 400
  }
 },
 "partial_corr_vs_local_intensity": {
  "edge_density": -0.4700015215594708,
  "gt_var": -0.6778462294037932,
  "entropy": -0.6342307984507887
 },
 "excess_sd": 0.0,
 "mean_gap": 0.009110640842696638,
 "heterogeneity_of_shape_supported": false,
 "verdict": "NOT SUPPORTED: signal-dependent (heteroscedastic) Gaussian noise reproduces the heavy-tailed pooled fit, the patch-beta spread and the feature correlations. Reframe Section IV as 'signal-dependent noise scale', and test the method against a scale-only model."
}
```
