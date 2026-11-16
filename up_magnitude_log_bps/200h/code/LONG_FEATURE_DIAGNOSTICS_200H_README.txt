200-HOUR LONG-FEATURE DIAGNOSTIC RUNNER

cd "C:\Users\Siddh\Desktop\ML\November\Week 2"
python .\Code\diagnose_long_features_200h.py

Default reference: Code/experiments/long_features_200h_v1/20261113T123706785745Z
New output: Code/experiments/long_feature_diagnostics_200h_v1/<timestamp>/
Optional --run-dir, --output-dir, --threads 2.
No fitting or tuning. Rebuilds causal features and runs saved-model inference.
Existing models, predictions and datasets remain unchanged.

CHECKS
Input/source hashes, completed prediction hash, exact target/UTC origin/issuance
alignment, mature mean/median baseline values, each saved model hash/card,
candidate and selected inference replay, stored selection-policy consistency,
and recomputation of every saved MAE/RMSE/IC table row. Failures stop completion.

OUTPUTS
training_vs_evaluation.csv: in-sample versus later errors, level/bias, rank IC,
baseline gains and fractions helped/hurt for candidate,selected,mean,median.
validation_windows.csv: stored earlier candidate errors, recomputed baseline
errors, policy-selected errors and maturity boundaries for all three windows.
validation_transfer.csv: earlier versus later gains relative to median.
feature_relationships.csv: selected candidate features' training/later Spearman
correlations, sign changes, median levels and missingness; marks fallback cases.
feature_terciles.csv: outcome means/medians in low/medium/high feature groups,
using mature-training cutoffs for both training and later periods.
group_metrics.csv: errors by past200h volatility and by FUTURE actual size.
Volatility cutoffs use training terciles; actual-size cutoffs use training P50/P90.
Actual-size groups are hindsight diagnosis only, never usable prediction gates.
period_metrics.csv: independently reproduced metrics plus bias/help/hurt details.
overlap_counts.csv: row spacing and greedy origin count spaced more than200h
apart. This count is NOT effective sample size and is NOT a significance test.
SUMMARY.txt, AUDIT.json, source snapshot, COMPLETE.json.

INTERPRETATION
In-sample scores can be optimistic. Inner candidate errors are saved records,
not independently replayed because the earlier inner models were not saved.
The final quarterly model is not substituted into its earlier validation rows.
Correlations and group differences are exploratory, not causal importance.
Sign reversal can reflect noise. Overlap alone does not prove why a model failed.
Original source availability limitations and historically exposed adaptive
evaluation remain. No automatic feature removal, retraining or model promotion.

TEST STATUS
Synthetic end-to-end pipeline, metric arithmetic, grouping, spacing, no-fit
guard, saved-model replay and corruption rejection tested. Full real-data
diagnostics have not been executed by the assistant; run the command above.
