RUN
cd "C:\Users\Siddh\Desktop\ML\November\Week 2"
python .\Code\tune_e2_features_48h.py

Optional: --threads 2 (default), --check-data-only, --self-test.
Resume an interrupted run with the same settings:
python .\Code\tune_e2_features_48h.py --resume "C:\full\path\to\run"
Cache hashes and code/input/config signature must match. Completed runs cannot
be resumed. Do not edit scripts or inputs during an active run.

BASELINE
Frozen E2 from lightgbm_repair_experiments_48h_v1/20261113T053731575845Z.
E2 is expanding Ridge plus a full-strength LightGBM correction trained on
monthly chronological out-of-fold Ridge residuals. Separate UP/DOWN mean
magnitudes in ordinary bps, fixed v45-style direction probabilities.
Ridge, direction, training history and evaluation dates remain unchanged.
The saved E2 prediction hash and8bundle hashes are checked. Bundles are replayed;
the rebuilt unchanged E2 must also reproduce the saved predictions within1e-6.

A: INDIVIDUAL REMOVALS
Each compared with unchanged235-input E2, retaining E2's side/issue parameters:
1 calendar flags
2 short-window rolling transformations (std/range/IQR/zscore/MA ratios), keeping
  lagged market values and longer price features
3 long-horizon total volatility
4 long-horizon upside/downside volatility together
5 long-horizon absolute net return and sum-absolute-return together
6 long-horizon price range
7-10 all six measures at one horizon:24h,48h,7days,30days
Removals are individual, not arbitrary combinations. Select A using earlier
validation MSE per side; unchanged remains eligible.

B: INDIVIDUAL HORIZON ADDITIONS
Start from selected A features. Add4h,12h or72h price features individually,
or add none. Each addition has realized volatility,up/down volatility,
absolute net return,sum of absolute returns,and cumulative log-price range.
Sources: canonical5m returns, endpoints strictly before origin, full-window
coverage required. Earlier validation chooses B per side; no outer selection.

C: CAPACITY
Start from selected B features. Test depth/leaves3/4,3/6,4/8,4/12 crossed
with min_child_samples20/30/50 (12settings), plus inherited E2 settings.
Learning rate remains E2's earlier-selected value. All other penalties,
sampling settings,1000maximum iterations and50round early stopping remain.
Select C on earlier validation; inherited configuration stays eligible.

D: SEED AVERAGE
Selected C at seed48 versus average of seeds48/148/248. Each seed independently
selects iterations. Average nonnegative side-magnitude predictions (rather
than averaging raw corrections before the nonnegative projection).
Report the three-seed policy and the earlier-validation-selected D policy.

VALIDATION AND LIMITS
All stages use the same three earlier windows, starting9/6/3months before
each outer issue and evaluating first2months of origins. Labels fully mature
before issue. Selection uses pooled conditional-magnitude MSE, separately per
side, matching E2. Final signed-return metrics use unchanged probabilities.
Sequential reuse of validation windows can overfit validation; this is not
nested independent confirmation. Nothing chooses variants from outer scores.
First-stage parameters are inherited from E2's earlier-validation search.
Tree training uses matured monthly OOF Ridge residuals, latest20percent early
stopping with48h purge, train-only preprocessing and affine residual scaling,
then full matured residual refit at selected tree count. No target clipping,
tail weighting or history tuning. Raw market availability is inherited.
Tie breaks: lower validation MSE, fewer columns, fewer seeds, experiment name.

OUTPUT
Code/experiments/e2_feature_tuning_48h_v1/<timestamp>/
SUMMARY.txt: all variants pooled, selected A/B/C/D by cycle, cycle2-4 means.
cycle_signed_metrics.csv: every variant,RMSE,MAE,IC,bias,direction accuracy,
  gains versus BOTH E2 and Ridge.
cycle_magnitude_metrics.csv: separate UP/DOWN metrics for every variant.
cycle_2_3_4_mean_metrics.csv: unweighted mean of per-cycle metrics.
tail_error_contributions.csv: excess squared error versus E2, split by
  per-issue training-only90th-percentile absolute magnitude threshold.
prediction_vs_actual_with_direction.csv: all variants, rowwise bps,percent,
  predicted/actual directions,error.
predictions.csv: both side magnitudes,probabilities,actuals,signed forecasts.
validation_metrics.csv,SELECTIONS.json,FIT_AUDIT.json,OOF_AUDIT.json,
CONTRACT.json, source snapshots, model bundles and hash-verified cache.
COMPLETE.json only after successful replay/output checks.
Cycle1=Jan-Mar2024,2=Apr-Jun,3=Jul-Sep,4=Oct-Dec;Jan2025 separately.
ALL includes all18481evaluation origins. Counterfactual group removals are
predictive experiments, not proof of a feature's causal influence.

COMPUTE
This is a substantial CPU/disk experiment. Roughly32 variant/seed fits per
side per cutoff before cache reuse, each with early-stop and final refit.
Four outer issues and three prior windows; duplicate configurations reuse
cache. Every declared variant gets saved outer predictions, not just winners.
The rowwise comparison file may be large. No runtime or improvement guarantee.
Requires existing local LightGBM dependencies, prior helper scripts, Week1
datasets, frozen direction artifacts and the completed E2 source directory.

TEST STATUS
Synthetic causal horizons, exact removal masks, early-stop purge, seed-average
semantics and saved-bundle replay passed. Real-data fitting is left to user.
