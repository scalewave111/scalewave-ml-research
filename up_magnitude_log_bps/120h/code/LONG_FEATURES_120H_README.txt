120-HOUR NEW MULTIDAY FEATURE SEARCH

Run from November/Week 2:
python .\Code\discover_long_features_120h.py

Read-only real-input audit (no fitting):
python .\Code\discover_long_features_120h.py --check-inputs
Synthetic checks:
python .\Code\check_long_features_120h.py

Scope is 120 HOURS ONLY, following the user's final correction.
Predict canonical up_magnitude_log_bps/down_magnitude_log_bps on all valid
observations including zeros. No endpoint-direction filter or unit conversion.
Same 47,808 eligible inputs and 18,193 evaluation origins. Invalid labels remain
excluded (144 invalid price paths and144 outside release).

WHAT CHANGED
Start feature selection empty using ONLY104 new LONG columns. Old F4Z and
NEWPRICE columns do not enter this experiment's predictors. Old price-follow-up
forecasts remain comparison columns, not feature-selection inputs.
The aim is to test sustained price behaviour over longer periods, not assume
longer lookbacks will improve predictions. This experiment changes features;
it retains the earlier Ridge/absolute-error histogram-boosting search machinery.

FEATURES
Eight windows:24/60/120/168/240/336/720/1440hours (up to60days).
Twelve measurements per window:
 1 Net log return.
 2 Volatility per square-root hour, calculated from hourly returns.
 3 Fraction of squared movement contributed by positive hours.
 4 Signed trend efficiency: net / sum absolute hourly returns.
 5 Momentum in volatility units: net / sqrt(sum squared hourly returns).
 6 Local cumulative log-price range.
 7 Current position within that range.
 8 Distance below local high in volatility units.
 9 Distance above local low in volatility units.
10 Fraction of positive hours.
11 Largest squared hourly return as a share of total squared movement.
12 Skewness of hourly returns.
Eight cross-window features: volatility ratios and mean-hourly-drift differences
for24/120h,120/240h,168/720h,336/1440h. These describe acceleration or slowing
relative to longer history; they do not use future targets.
The output feature_dictionary.csv gives exact names and formulas.

CAUSALITY AND MISSINGNESS
Canonical five-minute returns are indexed at their endpoints. An hourly bin
labelled t contains only endpoints in[t-1h,t), excluding endpoints at t.
At a half-hour forecast origin, reuse the latest completed hourly bin. Each
hour requires12 finite returns and each rolling window requires all its hours.
No future/backfill, no silent missing-return zero imputation into features.
Cumulative levels use an arbitrary offset internally, but local range outputs
are masked if a window contains missing returns; they do not bridge missing
history. Hourly ranges describe sampled canonical paths, not exchange highs/lows.
Denominators have explicit tiny floors; zero-volatility skewness stays missing.

SELECTION AND TRAINING
Independent UP/DOWN searches atJan/Apr/Jul/Oct2024. October also predictsJan2025.
Original input keys and48h boundary gap retained; every inner/outer training
label must end strictly before the boundary (strict120h maturity).
Three earlier quarterly validation windows; training-only preprocessing.
Require80% finite coverage,1000 readings and variation in every inner train.
Single-feature depth2 median-tree screen; earliest-training redundancy at
absolute Spearman>=.90; shortlist20; greedy addition up to6 features for each
of Ridge and MAE histogram boosting, followed by removal checks.
Each addition needs>=.1% pooledMAE gain and wins in>=2/3validation windows.
Final learned model must beat median under that gate; otherwise median fallback.
Ridge alpha=.01*n uses squared-error fitting with MAE selection. HGB uses
absolute_error,80iterations,rate.05,7leaves,depth3,minleaf128,L2=1,seed48.
Learned preprocessing fits only training rows; predictions clipped nonnegative.
Screen fits use up to4096 evenly spaced training observations; validation and
final fitting use all eligible rows. Default2threads; no automatic resume.
All model/feature choices are saved before outer fitting and evaluation.

COMPARISONS AND OUTPUTS
Code/experiments/long_features_120h_v1/<timestamp>/
metrics.csv and SUMMARY.txt compare new_features, previous, training_median,
training_mean overall, by2024quarter, andJan2025 separately, in logbps.
Previous is the completed price follow-up20261113T120541578982Z, not the worse
later tuning. Its prediction hash, timestamp/issuance keys and truths are checked;
old models are not independently replayed by this runner.
Also saves predictions.csv, feature_dictionary.csv, selected_features.csv,
selections.json, per-cycle quality/shortlist/trials/removals/model/hash records,
fit_windows.csv, excluded_labels.csv, provenance/versions/source snapshots.
Saved new-model replay must pass before COMPLETE.json is written.
Coverage across all rows is diagnostic only; selection uses inner training quality.

STATUS
Synthetic direct formulas, hourly endpoint boundaries, prefix/future perturbation,
missingness, maturity, signal selection, outer isolation, median fallback,
serialization and synthetic pipeline checks passed. Real-input and causal audits
passed:47,808rows/18,193evaluation origins/104new features. No real-data training.
This remains adaptive research on exposed historical evaluation; validation wins
do not guarantee later improvements. Overlapping120h observations are dependent.
Bounded greedy search can miss predictors that work only in combination.
