300-HOUR LONG-WINDOW FEATURE EXPERIMENT

Run from November/Week 2:
python .\Code\refine_long_features_300h.py

Input/hash checks and saved-model inference only (no fitting):
python .\Code\refine_long_features_300h.py --check-inputs

Synthetic checks:
python .\Code\check_long_features_300h.py

Only 300h is changed. Existing files and results are preserved.
Reference: excursion_pipeline_300h_v1/20261113T115754520122Z/
price_followup/20261113T121655587383Z.
Same independent canonical UP/DOWN excursions, including valid zeros, in log bps.
Same 17473 eligible development origins; 1008 invalid labels remain excluded.
Every training/validation boundary uses strict origin + 300 hours < boundary.
January/April/July/October quarterly fits; October also forecasts January2025.

54 NEW FEATURES
Four lookbacks: 168h (7d), 300h (12.5d), 720h (30d), 1440h (60d).
Each supplies 12 features from canonical five-minute log returns:
  net: sum of returns
  trend_to_vol: net / sqrt(sum squared returns)
  efficiency: net / sum absolute returns (signed trend efficiency)
  range_position: (last log level - window minimum) / window log range
  drawdown_from_high: window maximum log level - last log level
  rebound_from_low: last log level - window minimum log level
  range_to_vol: window log range / sqrt(sum squared returns)
  rv: sqrt(sum squared returns)
  up_rv: sqrt(sum squared positive returns)
  down_rv: sqrt(sum squared negative returns)
  downside_share: negative-return squared sum / total squared sum
  jump_concentration: largest squared five-minute return / squared sum
Six cross-window features, for 168/720h, 300/720h and 720/1440h:
  vol_rate_ratio: (short rv / sqrt(short hours)) / (long rv / sqrt(long hours))
  trend_rate_change: short net/hour minus long net/hour
Exact column names: LONG300__<measure>__<hours>h, or
LONG300__<cross-measure>__<short>h_<long>h.
See long_features_300h.py for exact implementation. Denominators are floored
at 1e-6 (squared-return denominators 1e-12), defining flat-price ratios as zero.
These are candidate market-state measures, not established predictive signals.

CAUSALITY / MISSINGNESS
Returns are indexed by their endpoints; only endpoints STRICTLY before the
forecast origin enter features. Full five-minute windows required. Missing
returns invalidate their window; no forward-filling. The complete five-minute
grid and aligned origins are checked. Relative path includes both window
start and end levels, using five-minute closes, not intrabar highs/lows.
drawdown_from_high is the CURRENT distance below the past high, not maximum
peak-to-trough drawdown over the window. Future target labels retain their
original one-minute reference-price definition.

DESIGN
Five independent feature packs:
  current previously selected inputs (control)
  new trend/path only (31 candidates)
  new volatility/regime only (23 candidates)
  all 54 new features, replacing all old features
  current plus all 54 new features
New features qualify using every earlier TRAINING window: >=80% available,
>=1000 nonmissing values, more than one unique value. Rejected names recorded.
No old feature is mandatory in replacement packs. No feature selection from
outer evaluation. Prior selected feature lists were already adaptively chosen.

Fixed original MAE histogram boosting for all packs: 80 iterations, rate .05,
depth3, 7 leaves, minleaf128, L2=1, seed48, no early stopping. Same training-only
imputation/scaling/missing indicators, nonnegative predictions. Expanding
training history only, to isolate feature changes from architecture/history.
Forecast weights 0/.25/.5/.75/1 blend model with expanding training median.
Three earlier quarterly windows choose minimum pooled MAE among candidates
with >=0.1% improvement over median and >=2/3 window wins. If none pass,
use median. The previous model remains a reported control; the gate does NOT
require beating that previous model. RMSE and IC are reported, not selected.
Passing earlier validation does not guarantee better later scores.

At most 120 inner candidate fits +24 inner reference fits +8 outer control
refits +8 selected outer fits; weights reuse predictions. Both targets run
sequentially. No runtime guarantee or automatic resume. Requires existing
Week 1 dependencies, canonical inputs and reference artifacts at saved paths.

OUTPUTS
Code/experiments/excursion_long_features_300h_v1/<timestamp>/
Predictions and period/pooled MAE, RMSE, IC for previous/selected/mean/median;
per-cycle feature quality, feature packs, trials, selections, inner/outer fit
windows; saved model bundles/hashes/replay; source/input/guidance hashes,
source snapshots, training/evaluation level diagnostics, SUMMARY/COMPLETE.
Only COMPLETE.json marks success. No report is generated.

Historical evaluation has already been exposed. These are adaptive research
comparisons, not independent confirmation. Strongly overlapping 300h targets
are not independent samples. Inherited original feature-availability limits
remain. No claim of improved performance before the user runs the experiment.

VALIDATION STATUS (2026-11-13)
Synthetic checks passed. Actual-input/hash and all eight saved-model replay
checks passed: 47088 input rows, 17473 development origins, 54 new features.
No real-data training performed while preparing this runner.
