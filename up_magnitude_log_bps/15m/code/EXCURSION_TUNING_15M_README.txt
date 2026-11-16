15-MINUTE UP/DOWN EXCURSION MAE TUNING

Run from November/Week 2:
python .\Code\tune_excursions_15m.py

Read-only input audit and saved-reference inference, without fitting:
python .\Code\tune_excursions_15m.py --check-inputs

Resume an interrupted tuning run:
python .\Code\tune_excursions_15m.py --resume "PATH_TO_TUNING_TIMESTAMP_FOLDER"
Completed target/quarter blocks are hash-checked and reused. An incomplete block
is recomputed. Code, inputs, versions, threads and selection settings must match.

TARGET / COMPARISON
Same canonical 15-minute up_magnitude_log_bps and down_magnitude_log_bps,
including zero excursions. Same 29615 initial and 18481 development origins,
30-minute sampling, Jan/Apr/Jul/Oct2024 expanding fits, original48h boundary gap.
Strict 15-minute maturity at all fit and validation boundaries. January2025
retains the October model and is separately scored. Primary objective is MAE.

Reference: completed excursion_pipeline_15m_v1/20261113T104418654180Z,
price_followup/20261113T110533531198Z. Both previous selected and all-price
models are hash-checked and replayed. The current selected model is the fallback.
Unchanged settings must reproduce both earlier-validation and outer predictions.

INDEPENDENT EXPERIMENTS
Three feature packs crossed with eight configurations (24 direct candidates):
1. Current selected per-quarter features.
2. Previous all_price_plus_original per-quarter features.
3. Pack2 plus16 new short-price features (15m and30m).
Each short lookback adds rv,up_rv,down_rv,net,past_range,distance_from_high,
distance_from_low,max_abs_return using only5m return endpoints strictly BEFORE
the forecast origin. Same formulas as prior1/4/12/24/48/168h features, now with
3 or6 returns. Missing windows stayNaN. Training-only quality>=80%coverage,
>=1000readings and variation in each inner fit applies to the new16 inputs.

All models: sklearn HistGradientBoostingRegressor absolute_error, seed48,
early_stopping=False. Training-median imputation, appended missing indicators,
training centering/scaling, nonnegative output clipping, as in previous run.
Configuration             depth leaves minleaf iterations rate   L2
unchanged                 3     7      128      80        .05     1
shallow_strong            2     4      256      80        .05    10
shallow_flexible          2     4       64     160        .05     5
smaller_leaf              3     7       64      80        .05     1
more_iterations           3     7      128     160        .05     1
slow_regularized          3     7      128     240        .025   10
deeper_regularized        4    12      128     160        .05    10
deeper_flexible           4    12       64     160        .025    5

SELECTION
Three earlier quarterly windows, same as the original search. For each direct
candidate test weights0.5 and1 against the current model:
new_forecast=(1-weight)*current_forecast+weight*candidate_forecast.
Require>=0.1%pooled MAE gain AND wins in>=2of3 windows versus BOTH current and
training median. Select eligible lowest pooled validationMAE; otherwise weight0
retains current exactly. Weight0.5 is an empirical MAE blend; it is not guaranteed
to equal a conditional median. No endpoint-direction probabilities are involved.
Earlier-selected feature sets and validation windows are reused; not fully nested.

Up to576 inner and192 outer model fits (768total), before identical-pack cache
sharing, plus saved-model inference. No random early stopping. The runtime is
not benchmarked; larger iteration counts can make this longer than earlier runs.

OUTPUT
Code/experiments/excursion_tuning_15m_v1/<timestamp>/
SUMMARY.txt, metrics.csv, predictions.csv, selections.json, CONTRACT.json,
COMPLETE.json, per-target/quarter validation trials, exact feature packs,
model artifacts, fit-window audit and hash-checked completion receipts.
All24 direct candidates receive pooled/quarter/January metrics. Current,
previous_all_price, median/mean and selected policies are included. Positive
mae_gain_vs_current means improvement. Direct candidate scores are diagnostic;
they do not change selection. A model bundle has features,config,model; use
base.predict(bundle['model'], X[bundle['features']].to_numpy(float)). For the
selected blend use selection.json candidate/current artifacts and weight.

STATUS / LIMITS
Prepared for user-run training. Historical periods have already been examined.
All-price pack was included following exposed results; no independent confirmation
or guaranteed improvement. Original feature-availability limitations remain.
No real-data model fitting during preparation. Existing results are preserved.
