Run from November/Week 2:
    python .\Code\fit_mlp_rf_conditional_means_48h.py

This starts NEW training. Existing models/results are preserved.

What it compares
----------------
Separate UP and DOWN MLPs, and separate UP and DOWN Random Forests.
Both families target ordinary-return conditional means using squared error.
MLP: earlier 8/4 SiLU correction over a frozen Ridge skip, three seeds.
RF: earlier 128-tree/depth3/leaf128 architecture, squared-error leaf means.
The RF here is standalone, not the Ridge/RF hybrid.

For each side and quarterly fold, compare the existing 4/5/8 feature packs.
Earlier validation MSE selects a pack per family and a learned family per side.
Baseline averages are scored as benchmarks; they are not selection candidates.
Zero MLP correction epochs retains fitted Ridge predictions, not a constant mean.
Selection uses the first two months after the cutoff three months before issue.
All fit labels must finish strictly before their fit cutoff. Final evaluation
uses the same 18,481 historical timestamps as the completed GRU/hybrid run.

Outputs
-------
New timestamped directory in Code/experiments/mlp_rf_conditional_means_48h_v1/:
  magnitude_metrics.csv: UP/DOWN MAE, RMSE, MSE, bias and rank IC for each pack,
    validation-selected MLP, validation-selected RF, selected learned family,
    previous GRU/hybrid and training-average benchmark.
  signed_metrics.csv: combined signed-return accuracy using the same probabilities.
  signed_return_predictions.csv: both magnitudes and signed forecasts for every row.
  SELECTIONS.json: selected model filenames and validation scores per side/fold.
  *.joblib: trained models with preprocessing; replay checked after saving.
  validation_scores.csv, FIT_AUDIT.json, CONTRACT.json, SUMMARY.txt, COMPLETE.json.

The primary selected forecast columns are:
  up_mean_bps, down_mean_magnitude_bps (both positive)
  up_mean_pct, down_mean_magnitude_pct
  signed_return_forecast_bps, signed_return_forecast_pct
Also retains separate mlp_signed_return_pct and rf_signed_return_pct.
Formula: P(UP)*UP mean - P(DOWN)*DOWN mean magnitude.
Percent columns use 1.5 to mean 1.5%. No log units in these outputs.

The reference is conditional_means_signed_48h_v1/20261112T063535834150Z.
Direction is direction_v45_style_v1/20261112T045009318177Z and is not retrained.
Defaults can be overridden with --reference-run, --direction-run, --prepared,
and --output-dir. Hashes, timestamps, truths and side baselines are checked.

Dependencies
------------
Keep fit_conditional_48h_mean_features.py and fit_conditional_means_signed_48h.py
alongside this file. The former imports the existing Week 1 helper modules.
Requires the same installed Python packages and Week 1 prepared data as prior runs.
predict_artifact(artifact, x, keys) supports label-free magnitude inference from
properly prepared features; this script does not fetch live market data.

Status: source reviewed, not executed/runtime-tested by assistant. Results are
unknown until run. Mean targeting does not guarantee accurate/calibrated means.
Evaluation is historically exposed; no fresh confirmation or uncertainty claim.
