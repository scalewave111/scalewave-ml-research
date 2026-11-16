72-HOUR MEDIAN-ANCHORED REFINEMENT

From November/Week 2:
python .\Code\tune_excursions_72h.py

Input audit and saved-model replay only, no fitting:
python .\Code\tune_excursions_72h.py --check-inputs

Reference: excursion_pipeline_72h_v1/20261113T111815030467Z/
price_followup/20261113T113500606206Z. Requires existing Week 1 dependencies,
canonical targets/returns and original reference artifacts at their saved paths.

Same independent UP/DOWN canonical 4320-minute excursions in log bps,
including valid zeros. Same 18385 eligible evaluation timestamps. The 96
invalid labels remain excluded. Quarterly expanding issuance schedule and
October model retained for January 2025. Every fit purges origins until
origin + 72h is STRICTLY earlier than its prediction/validation boundary.

MODEL AND EXPERIMENTS
Original MAE histogram boosting: 80 iterations, rate .05, depth3, 7 leaves,
minleaf128, L2=1, seed48, no early stopping. Original training-only median
imputation, scaling and missing indicators. Predictions clipped nonnegative.
No architecture tuning: experiments isolate feature/history/blend changes.

Two feature packs: previously selected inputs; those same inputs plus eight
price measures each at 72h, 168h, 336h, 720h (3d/7d/14d/30d).
Measures: total/up/down realized volatility (sqrt sum squared 5m log returns),
net return, past range, distance from high/low, maximum absolute 5m return.
Only return endpoints strictly before prediction time; full lookback required,
otherwise missing. Price ranges use 5m closes, not intrabar highs/lows.
Existing 7d columns are deduplicated. Other existing lookbacks unchanged.

Training histories: expanding, last180days, last365days. Recency affects model
training; the baseline anchor ALWAYS uses the expanding mature training median.
Forecast = w * model + (1-w) * expanding training median.
Weights: 0, .25, .5, .75, 1. Zero weight is the baseline/fallback.
Six feature/history candidates per side/issue, each with five blend weights.
Independent feature-only, history-only and blend-only controls are identified
in validation_trials.csv; combined cells are prespecified factorial tests,
not modifications inherited from earlier winning branches.

Three earlier quarterly windows choose minimum pooled MAE among candidates
beating expanding median MAE by >=0.1% and in >=2/3 windows. Otherwise use
the expanding median, even if the previous model is worse. Current model is
always carried as a comparison, not forced into the final policy. RMSE/IC are
reported but do not select this MAE policy. No guarantee of later improvement.
Reference lists were already selected on these earlier windows: this is adaptive
historical research, not nested or fresh confirmation.

OUTPUTS
Code/experiments/excursion_tuning_72h_v1/<timestamp>/
predictions.csv: actual, previous, selected, training_mean, training_median.
metrics.csv: pooled, 2024 quarters and January2025; MAE, RMSE, Spearman IC.
level_and_fit_diagnostics.csv: training/evaluation errors and predicted/actual
mean/median levels. Training diagnostics use selected candidate history (or
expanding history for fallback); they are in-sample, not validation evidence.
Per-cycle validation trials, chosen feature lists, window audits, model bundles
and hashes; CONTRACT/source/input/guidance hashes; SUMMARY and COMPLETE.
Selections are saved before the outer fit/evaluation. Full model replay required.
No automatic resume. New timestamped folders preserve prior outputs.
At most144 inner candidate fits +24 inner control fits +8 control refits
+8 selected outer fits; blends reuse predictions. No runtime guarantee.

VALIDATION STATUS
Synthetic purge, recency, long-window formula/causality, selection/gate,
outer-isolation, zero fallback, missingness and serialized-replay checks passed.
Actual input/hash audit and all8 saved-model replays passed:48000 input rows,
18385 evaluation origins,75 available price columns. No real-data fitting was
executed while preparing this runner. Inherited source-availability limitations
and exposed evaluation remain. Full runner completion awaits user execution.
