Run from November/Week 2:
    python .\Code\tune_conditional_mean_magnitudes_48h.py

NEW TRAINING: UP and DOWN conditional mean magnitudes only.
No direction model, probability blending, signed-return forecasts or trading.

Search
------
Default 25 configurations, balanced across five families:
1. MLP: hidden layers 8/4,16/8,32/16,32/16/8; ReLU/tanh; weight penalty,
   learning rate and 80/160-epoch budgets. Uses sklearn squared-error MLP,
   not the prior frozen-Ridge-skip MLP. Random early stopping is disabled.
2. RF: 128/256 trees, depth3/6/10, leaf64/128/256, feature subsampling,
   bootstrap4096 versus all eligible training rows.
3. Extra Trees: the same bounded tree settings with randomized splits.
4. Histogram gradient boosting: squared loss, leaves3/7/15, leaf128/256/512,
   60/120/200 iterations, learning rate and L2 penalty. No random validation split.
5. Ridge: five regularization strengths scaled by training-row count.

Each candidate also chooses the established 4/5/8 feature pack and expanding,
365-day or540-day training history. The seeded candidate list is written to
CONTRACT.json before fitting. This is bounded randomized search, not an exhaustive
search of all combinations or a guarantee of the lowest achievable error.

Selection and evaluation
------------------------
Tune UP and DOWN separately. For each quarterly issue (Jan/Apr/Jul/Oct2024),
evaluate each candidate at three earlier cutoffs: issue minus9/6/3 months.
Validation uses the first two months after each cutoff. Only matured labels
strictly before each training cutoff may enter the corresponding fit.
The row-weighted validation MSE selects the best candidate and top3 candidates.
Refit those three using eligible history before the outer issue.
Save two predeclared forecasts: best model and equal-weight top3 ensemble.
Both policies are evaluated; test outcomes do not choose between them.

Default budget: 600 inner fits plus24 final fits. Some MLP fits reach their
epoch limit; convergence warnings are recorded in validation_trials.csv and
FIT_AUDIT.json. The script prints progress for every inner fit.
For a smaller initial run: --trials 10
For a larger search: --trials 50
More trials do not guarantee better performance; comparing many runs increases
historical selection exposure. Supported range5-100, deterministic seed48.
The newest seed-specific trial schedule is frozen before evaluation, not learned
from development metrics. Single stochastic seed per candidate, no seed search.

Units and target
----------------
Positive absolute ordinary-bps endpoint returns, conditional on actual UP or DOWN.
Input labels are converted per observation from log bps before magnitude fitting.
Mean target is preserved by train-only AFFINE target normalization; no log-target
regression, target clipping, large-move deletion, Huber or MAE training.
Feature preprocessing uses training-only medians and mean/std, clipping features
to +/-8 standard units and adding missing flags. Predictions project to >=0.
All models predict both side scenarios at every eligible timestamp; no future
realized direction selects a branch. Flat training outcomes enter neither side.

Outputs
-------
New timestamped folder: Code/experiments/conditional_mean_magnitude_tuning_v1/
magnitude_predictions.csv:
    UP_best_mean_bps, DOWN_best_mean_bps,
    UP_ensemble_mean_bps, DOWN_ensemble_mean_bps, plus percent equivalents.
    100 bps = 1%; percentage value1.5 means1.5%.
    Includes actual magnitude/direction for evaluation, not as model inputs.
magnitude_metrics.csv: per-side MAE/RMSE/MSE/bias/rankIC, pooled and per fold.
calibration_by_fold.csv: descriptive forecast-tertile mean prediction vs outcome.
training_diagnostics.csv: training errors for the three retained candidates.
validation_trials.csv, SELECTIONS.json, FIT_AUDIT.json, CONTRACT.json,
source_snapshot.py, SUMMARY.txt and COMPLETE.json.
YYYY-MM_UP_mean_models.joblib and YYYY-MM_DOWN_mean_models.joblib:
    saved preprocessing, models, target scaling and training boundaries.
    Serialization replay is checked on all evaluation rows before accepting a fold.

Comparison rows: earlier validation-selected MLP, RF and UP-GRU/DOWN-hybrid,
plus an expanding training-side average benchmark. The benchmark is not a candidate
and never replaces the selected learned model. No claimed improvement until scores
are available. Previously exposed history is not an untouched holdout.

Inputs and dependencies
-----------------------
Week1 Code/experiments/eight_feature_exhaustive_v1/prepared.joblib.
Completed mlp_rf_conditional_means_48h_v1/20261112T064753697944Z is used ONLY for
matching outcomes/timestamps and prior magnitude comparisons; its probabilities
and signed-return forecasts are not loaded. Override --prepared/--reference-run
if needed. Outputs can be redirected with --output-dir; no overwrite.
Requires numpy, pandas, scipy, scikit-learn, joblib, threadpoolctl. No PyTorch.

predict_bundle(bundle, features) is a label-free inference function returning
best_mean_bps and ensemble_mean_bps for the bundle's side. It does not fetch live
features. Preserve source feature definitions and point-in-time availability.

Delivery status: source reviewed only. Assistant has not executed the runner,
training, or runtime tests. Model improvement is not established at delivery.
