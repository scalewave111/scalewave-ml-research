72-HOUR FRESH FEATURE SEARCH

Run from November/Week 2:
python .\Code\discover_long_features_72h.py

Read-only input audit: add --check-inputs
Synthetic checks: python .\Code\check_long_features_72h.py

TARGET AND POPULATION
Canonical up_magnitude_log_bps / down_magnitude_log_bps, horizon 4320 minutes,
including valid zeros. Same 48,000 eligible inputs and 18,385 evaluation origins.
48 invalid-price-path and 48 outside-release development labels excluded.
Strict origin+72h < training boundary in every inner and outer fit.
Quarterly January/April/July/October 2024 fits; October also serves January 2025.

FRESH CANDIDATES
No old F4Z or NEWPRICE columns enter selection. Search begins empty independently
for each target and quarter. These are new to this 72h search; the common
multiday formulas reuse the established long-horizon feature implementation.

120 candidates:
* 12 measurements at 24/36/72/144/168/336/720/1440 hours (96 columns): net return,
  hourly RMS volatility, up-variance share, trend efficiency, normalized momentum,
  hourly path range, range position, drawdown/rebound in volatility units,
  positive-hour share, jump-variance share, and hourly skewness.
* Volatility-rate ratios and mean-hourly-drift differences at 24/72, 72/144,
  168/720 and 336/1440 hours (8 columns).
* At 24/72/168/720 hours: P90 hourly return / RMS; -P10 hourly return / RMS;
  mean adjacent-hour return product / squared RMS; std(abs hourly returns) /
  mean(abs hourly returns) (16 columns). Adjacent movement is uncentered,
  not a correlation coefficient. Tail measurements may be negative.

All use canonical 5-minute return endpoints strictly before the forecast origin.
Hourly bin t uses endpoints [t-1h,t), requires 12 finite returns; half-hour origins
reuse the latest completed hour. Windows require complete hourly history.
No backward filling or missing-return zero imputation into predictors. Internal
relative price levels have arbitrary offsets; incomplete windows are masked.
Ranges use hourly sampled paths, not exchange intrabar highs/lows. Denominator
floors avoid division by zero; constant-window skewness may remain missing.
Exact formulas and feature names are saved in feature_dictionary.csv.

SEARCH
Existing Ridge and absolute-error histogram boosting machinery, allowing this
experiment to test a fresh feature universe. Training-only quality requires
80% finite coverage, 1,000 values, and variation in every earlier train window.
Single-feature depth-2 median-tree screen (up to 4,096 training rows), earliest
training Spearman redundancy threshold .90, shortlist 20. Independent greedy
Ridge/HGB searches add up to 6 features and perform removal checks.
Three earlier validation windows; additions require >=0.1% pooled MAE gain and
2/3 window wins. Final candidate must beat median under this gate or fall back
to median. Later evaluation is never a selection input.
Ridge alpha=.01*n uses squared-error fitting but MAE selection. HGB uses
absolute_error,80 iterations,rate .05,7 leaves,depth3,minleaf128,L2=1,seed48.
Preprocessing uses training rows only; outputs are clipped nonnegative.
Default 2 threads; no automatic resume. Prior selected features are not anchors.

COMPARISONS AND OUTPUT
Code/experiments/long_features_72h_v1/<timestamp>/
new_features versus previous tree price-followup, previous_neural, training
median, training mean; MAE/RMSE/rank IC overall and per quarter/January 2025.
Tree reference: pipeline20261113T111815030467Z/price_followup20261113T113500606206Z.
Neural reference: excursion_neural_72h_v1/20261113T123304519186Z.
Comparison forecasts are hashed and aligned by timestamp/target/issuance/truth;
old models are not independently replayed. They are not predictor columns.
Saves predictions, definitions, selected features, quality screens, trials,
removals, selections, models, fit windows, exclusions, hashes, source snapshots,
SUMMARY and COMPLETE. New model replay must pass before completion.

STATUS
Synthetic formulas, endpoint exclusion, prefix/future isolation, missingness,
purges, signal selection, fallback, outer isolation, serialization and synthetic
save/score pipeline passed. Real-input hash/truth/issuance and causal audit passed
48,000 rows / 18,385 origins / 120 candidates. No real-data fitting performed.
This is adaptive research on exposed history, not independent confirmation.
Overlapping 72h labels are dependent. Univariate screening can miss features
useful only jointly. Fresh features do not guarantee improvement.
