Run from Week 2 for the full comparison:
  python .\Code\tree_signed_return_search_48h.py --trials-per-family 8

Eight families: Decision Tree, Random Forest, Extra Trees, Gradient Boosting,
Histogram Gradient Boosting, XGBoost, LightGBM and CatBoost. This is a finite,
predeclared search, not every possible tree algorithm/configuration.

Two capacities per each of four side-specific feature sets:
  anchor: UP core4 / DOWN core4+basis
  reduced: UP3 / DOWN7 previously inspected fixed Ridge feature sets
  eight: original eight features
  expanded: original eight +20 causal price-history inputs (28 total)
Expanded inputs: past volatility levels, upside/downside semivolatility,
net/absolute recent moves, jump proxies and volatility ratios. Trees learn
nested splits/interactions directly; no future-direction regime routing.

All models predict conditional mean positive ordinary-bps magnitudes using
squared-error objectives; no log-target fitting, outlier removal or tail weights.
Separate UP/DOWN training on matured historical outcomes. Both models predict
every evaluation origin. Existing v45-style probabilities remain unchanged.
Signed return=p_up*UP_mean-p_down*DOWN_mean. Final forecast sign defines its
predicted direction; it can differ from simply thresholding p_up at 0.5.

Selection: minimum row-weighted conditional-magnitude validation MSE, per side
and family, over 3 earlier windows. This is NOT tuning on final signed-return
test error. Also report an overall earlier-validation side selector; it may
choose different families for UP and DOWN. No evaluation-selected winner is
automatically promoted. Identical rows and labels versus saved Ridge reference.

Expanding quarterly fits Jan/Apr/Jul/Oct2024. Only labels ending strictly before
each fit cutoff. Validation starts9/6/3months earlier; first2months of origins.
Every preprocessing statistic uses fit-training data only. Past-price windows
exclude endpoints at/after forecast time. Source hashes/canonical targets and
causality checks run before fitting. Saved models are replay checked.

With all8 families and8 trials:1536 validation fits +64 final fits.
This is a substantial CPU search. Faster default4 trials uses one capacity per
feature set:768 validation fits +64 final fits. No runtime guarantee or resume
mode. Every run writes a new folder; partial progress is saved after trials.
--threads 2 controls model threading. No GPU required.

Dependencies: installed XGBoost/LightGBM/CatBoost locally in Code/tree_model_deps;
runner adds that folder to its import path. Also requires existing sklearn,
numpy,pandas,scipy,joblib,duckdb,threadpoolctl and helper files in Code/Week1.
On another machine install optional libraries with:
  python -m pip install xgboost lightgbm catboost
Missing optional packages stop the run unless --allow-missing is explicitly set.

Results: Code/experiments/tree_signed_return_search_48h_v1/<UTC timestamp>/
  SUMMARY.txt: all-family signed-return metrics versus Ridge, side means, zero
  signed_metrics.csv: pooled/per-period RMSE,MAE,bias,IC,direction accuracy,UP rate
  prediction_vs_actual_with_direction.csv: each model/time, forecast/actual percent,
      predicted/actual direction, correct flag and error bps
  signed_return_predictions.csv: both magnitudes, probabilities, combined forecasts
  conditional_mean_metrics_by_fold.csv: UP/DOWN errors
  validation_metrics.csv, SELECTIONS.json, FIT_AUDIT.json, CONTRACT.json
  *.joblib and source snapshots; COMPLETE.json only on successful completion

All rows remain historically exposed. Features and model plans were influenced
by previous experiments. Beating Ridge here is not untouched-test confirmation.
Large expected-return errors can reflect outcome uncertainty; do not scale up
forecasts to fit particular examples. Results may be worse than Ridge.
