200-HOUR BTC UP/DOWN EXCURSION FEATURE SEARCH

PowerShell:
cd "C:\Users\Siddh\Desktop\ML\November\Week 2"
python .\Code\discover_excursion_features_200h.py

Optional read-only input audit (no fitting):
python .\Code\discover_excursion_features_200h.py --check-inputs

Dependencies: numpy, pandas, scipy, scikit-learn, duckdb, joblib, threadpoolctl.
Uses the existing local Week 1 X package and canonical 12000-minute BTCUSDT Y.
Paths may be overridden with --data-dir and --y-dir. No network downloads.

TARGETS
up_magnitude_log_bps = 10000*ln(highest future path price / origin price)
down_magnitude_log_bps = 10000*ln(origin price / lowest future path price)
Canonical one-minute reference-price path includes origin and ends 200 hours later.
All valid observations, including zero excursions, enter BOTH models. No direction filter.
Targets and predictions remain log bps; these are not endpoint signed returns.

DATES AND WALK FORWARD
Exact 29,615 initial training keys: 2022-04-22 00:30 to 2023-12-29 23:30 UTC.
Start from the original 18,481 development keys; score only valid 200h labels.
Fits: Jan/Apr/Jul/Oct 2024, expanding over supplied original keys.
Only labels ending strictly before each fit may enter training.
Jan 2025 uses the October fit and is reported separately from Oct-Dec.
Original 48h boundary gap remains absent; no new origins are synthesized.
Additional origins are excluded from each fit by strict 200h label maturity.
The 29615 initial keys describe the input inventory, not the January fit count.
Canonical INVALID_PRICE_PATH/OUTSIDE_RELEASE rows are excluded with an audit; missing keys, wrong horizon/symbol or malformed VALID labels stop the run.

SEARCH
Independent search per target and quarterly refit. No mandatory old features.
522 supplied masked F4Z value columns; no new derived PRICE inputs in this version.
Three earlier 3-month validation windows, starting 9/6/3 months before each fit.
Both training and validation labels are purged at their respective boundaries.
Every inner training window requires >=80% finite coverage, >=1000 readings,
and variation. Missing inputs never remove evaluation observations.
Training-only medians, missing indicators and scaling.
Screen: depth-2 absolute-error tree, maximum 4096 evenly spaced training rows;
scores use every inner validation row. Not a full-data exhaustive subset search.
Rank by pooled MAE, remove redundant columns at absolute Spearman >=.90 using
only the earliest training window (>=1000 common observations).
Keep up to 16 candidates. Greedy addition up to 6 features for each model:
  Ridge alpha=.01*n (squared-error fit, MAE selection)
  HistGradientBoosting absolute_error, 80 trees, rate .05, leaves7/depth3,
  minimum leaf128, L2=1, seed48, no random early-stopping split.
Addition requires >=.1% pooled MAE improvement and wins in >=2/3 windows.
Refit removal checks; remove features whose removal is no worse pooled and
in >=2/3 windows. Select the lower-MAE eligible model; otherwise median fallback.
All nonconstant predictions are clipped to zero. Final fits use full mature history.
Settings can be changed using --shortlist, --max-features, --screen-rows,
--min-gain and --threads; changing settings defines a new experiment.

OUTPUTS
New timestamped directory: Code/experiments/excursion_feature_search_200h_v1/
SUMMARY.txt / metrics.csv: pooled and period MAE, RMSE, Spearman IC versus
training mean and median. Constant-prediction IC is undefined, not zero.
selected_features_by_cycle.csv: selected names per target/fit.
feature_stability.csv: selection counts plus family/measurement/lookback metadata.
selections.json: full chosen lists, model and earlier validation scores.
Per target/fit: quality.csv, shortlist.json, validation_trials.csv,
removal_checks.csv, selection.json, model.joblib, predictions.csv, MODEL_HASH.json.
Removal MAE change >0 means removing that feature worsened the recorded set.
fit_windows.csv records fit/validation maturity boundaries.
CONTRACT.json contains input/code/guidance hashes, package versions and limitations.
Saved-model replay is checked before COMPLETE.json is written.
No automatic resume: an interrupted run retains partial files; rerunning creates
a separate directory. COMPLETE.json is the only completion signal.

INTERPRETATION
Selected lists are predictive candidates, not causal explanations. Empty lists
are valid: no searched set passed the baseline gate. No forced feature count.
Inner validation is adaptively reused; later periods are historically exposed.
Later observations can enter subsequent walk-forward training after maturity.
Do not choose a new pooled feature set from outer scores and call it untouched.
This bounded greedy search can miss interaction-only features.
Original masked X availability and PROVISIONAL/funding construction limitations
are inherited, not freshly certified. Predictions describe the supplied half-hour
sample and canonical reference-price extremes, not guaranteed exchange highs/lows.
Full market-data training is user-run. No promised gain or statistical significance.

200H DATA AUDIT (2026-11-13)
608 development labels are unusable: 304 INVALID_PRICE_PATH and 304 OUTSIDE_RELEASE.
The eligible population is 47488 total rows and 17873 development origins.
January has 29311 mature training rows after the strict 200h purge.
Both stages use the same eligible keys. Stage 1 saves excluded_labels.csv.
No invalid label is filled with zero; valid zero excursions remain included.
Actual-input audit and synthetic search/price-feature checks passed. No real-data
training was executed. Full follow-up reference replay awaits a completed first stage.
