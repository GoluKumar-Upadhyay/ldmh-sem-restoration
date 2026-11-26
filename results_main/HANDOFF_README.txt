Results of the master notebook (Kaggle / Colab)  -  2026-11-08 07:24
models=['baseline', 'global_gc', 'film_control', 'full'] seeds=[0, 1] epochs=30   split manifest sha1=31d7258b8c36

HOW TO USE THESE RESULTS IN VS CODE (notebooks 04-09 of the project)
1. Unzip results_latest.zip into a NEW folder, e.g.  <project>/results_kaggle/    (it contains runs/, tables/, figures/, split_manifest.json)
2. In the terminal that starts Jupyter / VS Code:
     PowerShell:  $env:SEMICON_RUNS = "<project>\results_kaggle"
                  $env:SEMICON_DATA = "<folder that contains train\train\GT and NoisyLR>"
                  $env:SEMICON_MODE = "pilot"
3. Run notebooks 04, 05, 06, 07, 08, 09. They read runs/<model>/seed<k>/ (best.pth, history.csv, test_per_image.csv, results.json),
   tables/degradation_fit.json and split_manifest.json from that folder. The split is keyed by file NAME, so it is identical on your PC.
4. Notebook 04 compares the models that exist; ablation / external models that were not trained here (full_no_penalty, full_clamp,
   NAFNet..., ...) are simply absent until you train them with  python scripts/run_experiments.py --groups ablation,external  .
