Run from November/Week 2:
    python .\Code\controlled_mean_feature_ablation_48h.py

To include MLP as well as Ridge and Random Forest:
    python .\Code\controlled_mean_feature_ablation_48h.py --include-mlp

This starts new training. Existing experiments are preserved.
Only positive UP/DOWN conditional mean magnitudes are predicted.
NO direction probabilities, signed-return forecasts or trading calculations.

What is controlled
------------------
Expanding training history for every model/feature set. Same eligible rows,
three earlier validation windows, seed48, target and fixed per-family parameters.
Model settings do not change during a single-feature contrast. No hyperparameter
search is mixed into the feature ablation.
Ridge: alpha=.01*training rows. RF:128 trees, depth3, leaf128, bootstrap4096.
Optional sklearn MLP:8/4 ReLU, alpha.1, learning rate.001,160 epochs,
no random validation split. It is not the older Ridge-skip MLP architecture.
All use squared error with train-only affine target normalization. No target
clipping, log-target regression or exclusion of large moves. Outputs >=0.

Feature experiments
-------------------
18 feature sets:
  core4 and full8;
  remove each of4 core features individually;
  add each of4 extra features to core4 individually;
  remove each of8 features from full8 individually.
The core4: 15m/5m historical liquidation activity,4h/24h volatility ratio,
1h liquidation venue breadth. Extra4:1h historical liquidation activity,
168h volatility,5m funding persistence,1h basis level.
Exact field names and mappings are written to CONTRACT.json.

Four outer folds (Jan/Apr/Jul/Oct2024), both sides. Training labels end strictly
before fit cutoff. Each candidate is evaluated at three earlier cutoffs9/6/3
months before issue; validation uses the first two following months, matured
before that three-month block ends. Only earlier validation MSE selects the
feature set per family and the overall learned model. Both side predictions
are made on every evaluation origin; actual side is used only for scoring.

Default:864 inner fits +288 outer fits =1152 fits (Ridge and RF).
With --include-mlp:1296 inner +432 outer =1728 fits.
Progress is printed for each inner fit. Serialization replay is checked for
every final model. Training does not resume automatically from partial runs;
rerunning writes a separate timestamped directory.

Where to read results
---------------------
Output: Code/experiments/controlled_mean_ablation_48h_v1/<timestamp>/

feature_effects_pooled.csv:
  Feature, model family, addition/removal context, error reduction and number
  of outer periods with reduced MSE. Positive reduction means including that
  feature reduced error in that comparison. Negative means it worsened error.
feature_effects_by_fold.csv: the same effects separately by outer period.
feature_effects_validation.csv: paired effects in the earlier validation windows.
These are conditional predictive contributions, not causal effects or proof
of future usefulness. Correlated features can substitute for one another.
There is no automatic pruning from the evaluation-period effects.

magnitude_metrics.csv: selected and fixed-feature model errors, plus historical
training-average and previous tuning-run benchmarks, pooled and by period.
tail_errors.csv: errors and share of squared error in ordinary/large moves.
  Large = above the side's90th percentile from matured expanding training data
  at that fold cutoff. Thresholds never use evaluation outcomes. Pooled groups
  therefore combine period-specific thresholds, not one full-test percentile.
calibration.csv: selected-model prediction bins using earlier-validation
  prediction cutoffs; actual means/errors reported later. No fitted correction.
training_diagnostics.csv: in-sample errors for all final models.
magnitude_predictions.csv: all model/feature predictions plus:
  UP_selected_mean_bps, DOWN_selected_mean_bps, and percent equivalents.
  Both magnitudes are positive.100bps=1%, percentage value1.5 means1.5%.
*.joblib: trained artifacts with preprocessing, model and target scaling.
SELECTIONS.json, FIT_AUDIT.json, validation_metrics.csv, CONTRACT.json,
source/helper snapshots, SUMMARY.txt and COMPLETE.json provide provenance.

Requirements
------------
Keep tune_conditional_mean_magnitudes_48h.py alongside this runner; its shared
fit/predict routines are used without running its main tuning loop.
Same numpy/pandas/scipy/scikit-learn/joblib/threadpoolctl dependencies as prior run.
Uses Week1 prepared.joblib and completed tuning run20261112T072620594416Z.
Override with --prepared, --reference-run, --output-dir or --threads.

Status: source reviewed only, not executed or runtime-tested by assistant.
The historical evaluation has already been inspected. No improvement or
statistical significance is claimed before the experiment is run and reviewed.
