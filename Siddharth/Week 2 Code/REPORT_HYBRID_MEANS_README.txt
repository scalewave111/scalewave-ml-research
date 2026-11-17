Run from November/Week 2:
  python .\Code\fit_report_hybrid_means_48h.py

Optional input-only check (no fitting):
  python .\Code\fit_report_hybrid_means_48h.py --check-data-only
Synthetic checks: --self-test. CPU thread limit: --threads 2 (default).

MODEL
Separate UP and DOWN conditional mean models in ordinary basis points.
Each is a blend of two-liquidation-input ordinary least squares regression and
an average of three 128-tree Random Forest regressors using all seven inputs.
All fits use squared error, affine target scaling, and positive magnitudes.
No target log transform, target trimming, tail weighting or direction training.
Both side models predict every origin; signed return = p_up*UP - p_down*DOWN.
Existing v45-style direction probabilities are reused from the Ridge reference.

This adapts the report architecture to means. It does not reproduce the PDF's
MAE model or its log-bps score. The PDF omits exact column identifiers and some
hyperparameters. These explicit choices are also recorded in CONTRACT.json:
  forest depth3, minimum leaf128, bootstrap up to4096 rows, max_features1.0;
  seeds48/148/248; three forests of128 trees, squared_error criterion;
  OLS replaces the original two-input median-style component;
  flow activity uses its percentile representation (not robust z-score).

EXACT SEVEN INPUTS
F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__15m__value
F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__5m__value
F4Z__flow__flow_activity_percentile__1h__value
F4Z__flow__flow_activity_percentile__15m__value
F4Z__flow__flow_activity_percentile__5m__value
F4Z__liquidation__venue_fragmentation_percentile__1h__value
F4Z__liquidation__liq_temporal_concentration_percentile__1h__value
The first two feed OLS; all seven feed the forests. Missingness flags accompany
the inputs. Train-only median imputation/scaling and8-sigma feature clipping
match the existing mean experiment helper. No new market features are derived.

VALIDATION
Four expanding quarterly fits, Jan/Apr/Jul/Oct2024. Training label_end must be
strictly before each cutoff. Separately for each side, three earlier windows
(starting9/6/3months earlier, first2months) select forest weight from
[0,.1,.25,.5,.75,1] by pooled row-weighted conditional-magnitude MSE.
Outer outcomes never choose the weight. Weights are newly selected for means;
the report's historical MAE-selected weights are not copied.
32 component-pair fits:24 inner and8 final; each includes1 OLS and3 forests.
No resume mode. Each invocation creates a new timestamped output directory.

OUTPUT
Code/experiments/report_hybrid_means_48h_v1/<UTC timestamp>/
  SUMMARY.txt: pooled signed and conditional-magnitude results
  signed_metrics.csv: pooled/per-period MAE,RMSE,MSE,bias,Spearman IC,
      forecast/actual means, prediction spread and direction accuracy
  conditional_mean_metrics.csv: same magnitude metrics by side and period
  prediction_vs_actual_with_direction.csv: rowwise bps/percent, predicted and
      actual direction, correct flag and error for every comparator
  signed_return_predictions.csv: both sides' means, probabilities and returns
  validation_metrics.csv, SELECTIONS.json, FIT_AUDIT.json, CONTRACT.json
  eight model bundles, source snapshots and per-period predictions
  COMPLETE.json only after all outputs and saved-model replay checks succeed

Comparators: hybrid, two-input linear, seven-input forest, training side means,
saved best Ridge signed forecast, zero. Constant IC is undefined, not zero.
The hybrid is the prespecified primary; outer rankings do not promote a model.
predict_bundle(bundle, seven_feature_dataframe) supports label-free magnitude
inference; future signed inference also needs valid direction probabilities.

Dependencies: existing Week1 prepared.joblib and package X tables, saved Week2
Ridge reference, tune_conditional_mean_magnitudes_48h.py, numpy,pandas,scipy,
scikit-learn,joblib,duckdb,threadpoolctl. No optional boosted-tree libraries.
Historical evaluation is already exposed. Source F4Z availability is inherited;
this runner does not redo its original feature-availability audit.
No guaranteed improvement and no real-data training performed during creation.
