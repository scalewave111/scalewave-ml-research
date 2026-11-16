72H EXPANDING VERSUS RECENT MEDIANS

cd "C:\Users\Siddh\Desktop\ML\November\Week 2"
python .\Code\compare_rolling_medians_72h.py

Default reference:
Code/experiments/excursion_tuning_72h_v1/20261113T120003142401Z
Optional --reference-run "OTHER_COMPATIBLE_COMPLETED_RUN"
Output: Code/experiments/rolling_medians_72h_v1/<timestamp>/

Same 18385 eligible origins and separate canonical 72h UP/DOWN log-bps targets,
including valid zero labels. No tree or neural fitting. Existing predictions,
datasets and model files are not changed. Existing helpers/dependencies and
original source paths are required; input/source/prediction hashes are checked.

COMPARISONS
1. Original expanding median frozen at each quarterly issue.
2. Original quarterly mean and saved previous/tuned tree predictions.
3. Expanding median updated at EACH forecast time.
4. Rolling medians updated at EACH forecast, using 30/90/180/365 days.
5. Earlier-validation-selected method, or frozen quarterly median fallback.

CAUSAL WINDOW DEFINITION
At time t, only targets with origin+72h STRICTLY LESS THAN t may enter.
A D-day window covers maturity times [t-D days,t), equivalently target origins
[t-D days-72h,t-72h). The latest three days of origins cannot enter.
Minimum48 eligible matured observations; missing/invalid labels are excluded,
never zero-filled. Insufficient required history stops the run.
Uses the original eligible30-minute origin inventory and inherited48h gap.

Online medians may use earlier evaluation outcomes AFTER those outcomes mature;
this is an explicitly online forecast simulation, not a frozen quarterly model.
The online expanding control distinguishes update frequency from recent-history
effects. All window widths are prespecified. Each side's choice is selected at
Jan/Apr/Jul/Oct issuance using only three earlier quarterly validation windows.
Requires >=0.1% pooled MAE improvement and wins in >=2/3 windows versus frozen
quarterly median, otherwise retains that baseline. October choice remains for
January2025; online median values still update using matured data.

OUTPUTS
predictions.csv: all prespecified forecasts, actuals, selected method and counts.
metrics.csv: MAE/RMSE/rankIC by quarter, January2025 and pooled, plus MAE gains.
validation_trials.csv, selections.json, availability_audit.csv.
CONTRACT.json: source/input/guidance hashes, rules and versions.
SUMMARY.txt and COMPLETE.json (written only on successful completion).
Prefix replay checks ensure future labels cannot affect forecasts at checked times.

The best fixed window on outer scores is hindsight, not an earlier-selected policy.
Repeated historical evaluation and overlapping labels prevent fresh-confirmation
or independence claims. No automatic promotion, no guaranteed gain, no resume.

Preparation: synthetic maturity boundaries, rolling membership/counts, irregular
keys, prefix/future isolation, insufficient history, gate and fallback checks;
CLI/import check. Real-data experiment is left for user execution.
