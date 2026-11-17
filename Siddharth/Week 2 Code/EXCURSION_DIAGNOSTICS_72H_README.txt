72H SAVED-FORECAST DIAGNOSTICS -- NO RETRAINING

cd "C:\Users\Siddh\Desktop\ML\November\Week 2"
python .\Code\diagnose_excursions_72h.py

Default source: Code/experiments/excursion_tuning_72h_v1/20261113T120003142401Z
Optional: --run-dir "PATH_TO_OTHER_COMPATIBLE_COMPLETED_72H_TUNING_RUN"
Outputs: Code/experiments/excursion_diagnostics_72h_v1/<timestamp>/

Requires the existing 72h tuning and follow-up helpers, original datasets,
saved reference models and their unchanged recorded source/input paths.
Uses saved-model inference only. Imported fitting entry points are disabled.
No training, feature selection, prediction changes or original-file writes.

CHECKS
Completed 4320-minute run, prediction/source/input/model hashes, all valid keys,
canonical targets, issuance, nonnegative/finite predictions, saved-model replay,
carried previous forecasts, training mean/median and every saved metric row.
Strict 72h label maturity is used when rebuilding training populations.
Any mismatch stops execution before a COMPLETE record is written.

OUTPUT TABLES
period_metrics.csv: pooled, each2024quarter and January2025. MAE/RMSE/IC,
level bias, gain versus median, helped/hurt/tied fractions and adjustment size.
training_vs_evaluation.csv: saved-model inference on candidate's training
history, expanding training population and later evaluation. Both training
populations are in-sample for their overlapping fit observations; they cannot
establish generalization. October evaluation includes January2025 here.
group_metrics.csv: same error comparisons by past72h volatility terciles and
actual excursion size (<=trainingP50, P50-P90, >P90). Thresholds come only from
expanding mature training. Missing volatility is a separate group. Ties may
produce unequal or empty groups. Actual-size groups use future outcomes and
are retrospective diagnosis ONLY, never permissible prediction gates.
total_absolute_error_gain is additive across disjoint groups; mean gains are
weighted by group sizes. Positive gain means the model beats the median.
validation_windows.csv: three saved inner-window errors, dates and row counts.
validation_transfer.csv: weighted inner gain versus later gain, window wins,
largest-window share of positive gain, chosen pack/history/blend weight.
Inner models were not saved: inner errors are recorded evidence, not replayed.
training_thresholds.csv and annotated_predictions.csv: exact group cutoffs
and row-level assignments. All price features remain strictly past-looking.
SUMMARY.txt, AUDIT.json and COMPLETE.json: readable summary and provenance.

No significance claim: 72h labels overlap, observations are dependent, and the
historical evaluation/validation have already been inspected repeatedly.
No automatic causal diagnosis, recommended gate, or new model promotion.

Preparation checks: synthetic diagnostic arithmetic, group-boundary/missing
cases and no-fitting guard; CLI/import check. Full real-data diagnostic is
left for the user to run. No full diagnostic execution during preparation.
