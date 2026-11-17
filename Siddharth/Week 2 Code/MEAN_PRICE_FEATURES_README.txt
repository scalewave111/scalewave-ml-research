Run from Week 2:
  python .\Code\discover_mean_magnitude_features_48h.py

Purpose: investigate new price-history predictors of UP/DOWN conditional mean
48-hour endpoint magnitudes. No signed-return forecasts or probabilities.
Both side outputs are nonnegative ordinary bps at every reference origin.

Five predefined feature blocks, tested individually and together:
1. Volatility levels: sqrt(sum of squared 5m log returns), 4/24/48/168 hours.
2. Side volatility: positive/negative-return squared sums, 24/48 hours.
   Both are past-history inputs to BOTH models, not future-side routing.
3. Recent moves: past net and absolute net log returns, 4/24/48 hours.
4. Jumps: maximum absolute 5m log return and its squared share of total squared
   returns, 24/48 hours. These are proxies, not formal jump classification.
5. Volatility change: hourly-normalized 24h/168h and 48h/168h volatility ratios.

Feature log-bps units do NOT change the target: each future 48h label is
converted to ordinary-bps magnitude BEFORE squared-error fitting.
Every derived feature uses only five-minute return endpoints STRICTLY BEFORE
the forecast time. Complete rolling windows required; gaps remain missing and
are imputed using training-only medians with missingness flags. No future fill.
Canonical source hash and exact reproduction of existing targets are checked.
Prefix and future-perturbation checks test causal feature construction.

Fixed Ridge alpha=0.01*training rows; no hyperparameter search in this runner.
Expanding history. UP anchor=core4; DOWN anchor=core4+basis. Fixed model lets
comparisons measure the effect of adding feature blocks to those anchors.
Blocks are tested individually and all together; no exhaustive subset search.

Selection uses only 3 earlier validation windows per side and outer quarter.
An added block must beat both anchor and training mean in pooled validation
MSE and beat anchor in at least 2/3 windows. Select the eligible block with
lowest pooled MSE; if none qualify, retain anchor. This is a predeclared
screening rule, not a significance test or improvement guarantee.
All validation and final fit labels end strictly before their fit cutoff.
Offsets 9/6/3 months; validation origins cover first 2 months after each cutoff.
168 inner +56 final Ridge fits. --threads defaults to 2.

Outputs: Code/experiments/mean_magnitude_price_features_48h_v1/<timestamp>/
SUMMARY.txt, magnitude_metrics.csv: all blocks, selected policy, averages,
and the earlier best fixed Ridge comparator. Pooled and per-quarter errors.
feature_block_effects.csv: positive MSE reduction means improvement over anchor.
tail_errors.csv: ordinary/large errors; large means above training-side P90.
Tail errors are diagnostics only; training/selection do not overweight extremes.
validation_metrics.csv, SELECTIONS.json: exact earlier-window evidence/choices.
magnitude_predictions.csv: both side predictions, actuals and fold identifiers.
*.joblib: all saved per-side/per-quarter models, usable through base.predict
with the required past-only feature columns. No target needed for inference.
CONTRACT.json, FIT_AUDIT.json, source snapshots, COMPLETE.json: provenance.

Dependencies: existing tuning helper beside runner; canonical_return_features.py
in Week 1/Code; numpy,pandas,scipy,scikit-learn,joblib,threadpoolctl,duckdb.
Defaults use the existing prepared.joblib, canonical_returns_5m.parquet and
completed controlled-ablation run 20261112T082233470776Z. Input paths can be
overridden via --prepared, --returns and --reference-run.

Evaluation remains the same previously exposed 18,481 origins, Jan2024-Jan2025.
It is not fresh-data confirmation. Comparisons are conditional on the fixed
Ridge model/anchor; they are not universal or causal feature-importance claims.
The original F4Z features' live publication latency is not re-established here.
No newly trained performance is claimed until the user executes the runner.
