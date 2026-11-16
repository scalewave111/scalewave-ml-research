SECOND 4-HOUR EXCURSION EXPERIMENT: NEW HISTORICAL PRICE FEATURES

cd "C:\Users\Siddh\Desktop\ML\November\Week 2"
python .\Code\improve_excursion_features_4h.py

Optional input-only check:
python .\Code\improve_excursion_features_4h.py --check-inputs

Uses existing installed numpy/pandas/scipy/scikit-learn/duckdb/joblib/threadpoolctl.
Keep Code/discover_excursion_features_4h.py and the Week 1 canonical_return_features.py
helper available. No new package installation or download is built into the runner.

WHAT CHANGES
51 new price inputs from the checksum-verified canonical 5-minute BTC log returns:
For each 1h,4h,12h,24h,48h,168h lookback:
  realized volatility = sqrt(sum squared returns)
  up/down semivolatility = sqrt(sum squared positive/negative returns)
  net log return, past range, distance from past high/low, largest absolute5m return.
Three duration-normalized volatility ratios:1h/24h,4h/24h,24h/168h.
Ranges and distances use reconstructed relative log levels at5m closes, not
exchange intrabar highs/lows. These are historical inputs; targets stay canonical
4-hour future excursions from the minute reference-price path.
All source return END timestamps must be strictly before prediction time.
Complete rolling windows required; missing windows stay NaN. Causal prefix and
future-perturbation checks run before any fitting.

WHAT STAYS FIXED
Exact original29615 initial and18481 development origins, same asset/targets/logbps.
All valid rows including zeros enter both models; no endpoint-direction filtering.
Jan/Apr/Jul/Oct2024 expanding refits; strict4h maturity at every boundary.
October model also predicts January2025; January scored separately.
Same original80-iteration absolute-error HistGradientBoosting settings.
Same three earlier quarterly validation windows; MAE objective.
Original48h boundary gap preserved; no additional timestamps inserted.

CONTROLS AND SELECTION
Reference: excursion_feature_search_4h_v1/20261113T082016050782Z.
The unchanged original model/feature card is rebuilt and must reproduce its
saved predictions within1e-7 logbps. Model and prediction hashes are checked.
New features require80% coverage,1000readings and variation in every inner fit.
Each lookback is a feature block; the ratios form a seventh block.
Start at original features and independently test each block addition.
Greedy search may add up to2blocks (adjust --max-blocks). Each addition must
reduce weighted pooled earlier MAE by>=.1% and win>=2of3windows.
Also test original+all admitted price inputs and price inputs alone independently.
Remove each selected block and the original feature group as ablation challengers.
Final selector compares all tested variants on earlier validation only; admission
requires>=.1% pooled improvement and2of3wins against BOTH original and median.
If nothing passes, select unchanged original. No model blending.
The original F4Z inputs are retained for addition trials but price-only/removal
controls can replace them. Original feature lists are historically selected,
not claimed to arise from a new nested selection process.

OUTPUT
New timestamped Code/experiments/excursion_price_followup_4h_v1/ folder.
SUMMARY.txt compares selected strategy/original/mean/median.
metrics_all_variants.csv reports MAE/RMSE/rankIC and gains versus original,
per quarter,January and pooled. Positive gain means lower error.
All tried variants have outer predictions saved AFTER selection is frozen.
Adaptive combination variants may exist in only some cycles: full_period_coverage
and rows explicitly identify this; their gain uses original on matching rows.
Do not compare partial-coverage pooled errors against full-population errors.
selections.json / selected_features.csv: chosen features per side and refit.
Per cycle/target: new_feature_quality.csv, validation_trials.csv,
removal_checks.csv, selection.json with hashed model artifacts, predictions.csv.
fit_windows.csv records training label maturity.
Contract/source snapshots include input identities/settings/versions/limitations.
Saved models are reloaded and predictions replayed. COMPLETE.json marks success.
No automatic resume: interrupted files remain partial; rerunning makes a new folder.

INTERPRETATION
This is another historically exposed experiment. Selection reuses the same earlier
validation windows adaptively; subsequent outer scores must not be used to select
a different winner and then called independent confirmation.
Original availability/provisional source-timing limits remain.
Earlier validation gates cannot guarantee lower later-period errors.
No real-data model training was performed when preparing this runner.
