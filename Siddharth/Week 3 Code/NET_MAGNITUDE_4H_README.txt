4-HOUR NET MAGNITUDE: FOUR SEPARATE MODEL FAMILIES

From the November Week 3 folder, run:
python Code/fit_net_magnitude_4h_separate.py

Runs HGB, ExtraTrees, MLP and GRU sequentially, each across all four
Jan/Apr/Jul/Oct 2024 issue periods. Within each family, earlier validation
chooses the feature pack; the primary forecast has no baseline fallback.

One family:
python Code/fit_net_magnitude_4h.py --pure-model hgb
Replace hgb with extra, mlp or gru as desired.

Input audit without fitting:
python Code/fit_net_magnitude_4h.py --check-inputs --models hgb
Optional synthetic model tests (does fit synthetic models):
python Code/fit_net_magnitude_4h.py --self-test

TARGET AND POPULATION
net_magnitude_log_bps = abs(10000 * ln(P[t+4h]/P[t])).
Reads checksum-verified canonical horizon_minutes=240 BTCUSDT partitions.
Uses the inherited playground origin grid and masked feature timeline.
Filters invalid labels and source-binding transitions using a 4-hour horizon;
never calls the 48-hour label loader. Audits canonical direction and excursion
bounds. Training labels must end strictly before each fitting cutoff.
No new raw labels are calculated or source datasets modified.

COMPARABILITY WITH THE 48H RUNNER
Same five feature packs, including the same 48 causal price features with
4/24/48/168/336-hour lookbacks and volatility ratios. Those are historical
lookbacks, not prediction horizons. Same GRU snapshots at 12/8/4/0 hours.
Same model settings, seeds, row cap (6000), training-only preprocessing,
three earlier two-month validation windows offset 9/6/3 months, pooled MAE
selection, baseline comparisons, saved-model replay and output audits.
The feature set has not been optimized specifically for the 4-hour horizon.

OUTPUTS (new timestamped folders; 48-hour results preserved)
Code/experiments/net_magnitude_4h_<family>_only_v1/<timestamp>/
Report/net_magnitude_4h_<family>_only/<timestamp>/
Includes predictions, MAE/RMSE/bias, validation trials, exact selected features,
models, input hashes, population audit and COMPLETE.json after success.
Compare the hgb/extra/mlp/gru rows respectively; guarded_selected is a separate
audit policy and may use the median baseline.

The individual script without --pure-model retains the earlier mixed-family
validation-gated workflow and writes net_magnitude_4h_v1 / net_magnitude_4h.
Use the separate launcher above for the requested independent-family workflow.

DEPENDENCIES
Same environment and read-only Week 2 code dependencies as the 48-hour runner:
numpy, pandas, scikit-learn, duckdb, joblib, threadpoolctl; torch for GRU.
--week2-code and --canonical-root override inherited local paths.
--threads and --row-cap are forwarded by the separate launcher.

STATUS
Generated and syntax checked only. No input audit, synthetic model fitting,
real-data training or inference performed by the assistant for this adaptation.
Historical data are already exposed; no untouched-test or improvement claim.
