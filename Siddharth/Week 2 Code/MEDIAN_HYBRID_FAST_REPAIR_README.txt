FAST RUNNER
Stop the slow run with Ctrl+C in its terminal, then from Week2:
python .\Code\repair_median_hybrid_fast_48h.py

This is a separate experiment. It does not alter or delete the slow run or its
cached monthly models. Do not pass the slow run directory to --resume.
Resume THIS runner with --resume "C:\full\path\to\fast\run" and identical
code,inputs,threads. Default2threads. Optional --self-test and --check-data-only.

WHY FASTER
No new median regressions or absolute-error forests. Uses the already saved,
hash-verified and replay-checked package hybrid forecasts as chronological
out-of-time forecasts. Errors on the realized side become correction labels.
Same235 causal E2 features and LightGBM L2 correction/normalization/early-stop
mechanics. Only3 configs:depth3/leaves4/rate.01,depth3/leaves4/rate.03,
depth4/leaves8/rate.02. With supplied input coverage:18 inner fits plus6 outer
fits; each has early stopping and full-history refit. No measured runtime yet.

TRADEOFFS
Saved hybrid forecasts start January2024, so Cycle1 remains unchanged.
April,July,October issues learn only from earlier completed forecast errors.
Residuals reflect the actual previous quarterly cards; unlike the slow runner,
they are not rebuilt under the current issue's card. This distribution change
can weaken transfer. This is NOT an exact substitute for monthly E2 residuals.

At each issue, last30days form config-selection window; only fully matured
48h outcomes are included. Correction training labels end strictly before
that window. Its own last20percent earlier data selects iteration count with
a48h purge,1000tree cap,50patience. Best config by side magnitudeMSE is then
refitted on all mature prior residuals. Require500 preselection and100 holdout
side rows or retain baseline. No evaluation-period outcome selects its model.

corrected uses the best correction. selected uses it only when holdoutMSE
beats the unchanged hybrid; otherwise that side stays unchanged.
Final magnitude=max(0,original hybrid+L2 correction), then fixed direction
probabilities combine UP/DOWN. Mean-oriented output, not a pure median.
Targets remain endpoint magnitudes, not upward/downward path excursions.
No new forest training; no guaranteed error decrease or measured time promise.
Historical exposed evaluation and inherited package selection remain.

OUTPUT
Code/experiments/package_median_hybrid_fast_repair_48h_v1/<timestamp>/
SUMMARY.txt,cycle_signed_metrics.csv,cycle_magnitude_metrics.csv,
cycle_2_3_4_mean_metrics.csv,eligible_period_signed_metrics.csv,
predictions.csv,prediction_vs_actual_with_direction.csv,tail_error_contributions.csv,
validation_metrics.csv,SELECTIONS_AND_FIT_AUDIT.json,CONTRACT.json,
serialized bundles,checksum cache,COMPLETE.json.
Compare hybrid,corrected,selected,E2,Ridge,zero on matching origins.
Eligible-period metrics exclude warmup to avoid diluting the measured change.

VALIDATION
Synthetic saved-error,side-mask,holdout-purge,future-invariance and warmup
fallback checks passed. Full correction fitter/composition/replay synthetic
checks passed during original runner preparation. Real input/replay checks
reported in project context. No real-data training by assistant.
