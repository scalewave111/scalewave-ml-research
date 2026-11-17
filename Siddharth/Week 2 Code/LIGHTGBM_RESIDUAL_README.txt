RUN IN POWERSHELL
cd "C:\Users\Siddh\Desktop\ML\November\Week 2"
python .\Code\tune_lightgbm_residual_48h.py

Optional: --threads 2 (default), --check-data-only, --self-test.
The default trains and evaluates; input-only check does not fit models.
No real-data training was performed while preparing this runner.

TASK
Continue the UP/DOWN conditional mean magnitude experiment, ordinary bps,48h.
Direction probabilities remain those of the saved v45-style reference.
Signed forecast = p_up*UP_mean - p_down*DOWN_mean.
Four policies reported: existing Ridge, tuned direct LightGBM, tuned
Ridge+LightGBM residual, and earlier-validation-selected fallback policy.
No switch is made based on Cycle2/3/4 evaluation outcomes.

FEATURE ENGINEERING
Use the union of the existing8 magnitude inputs and report7 market inputs
(13 unique raw inputs). Tree inputs use one-period lag plus rolling standard
deviation, min-max range, IQR and z-score over7/14/30periods; moving-average
ratios7/30 and14/30. One period=30minutes, not one day.
Reindex onto an elapsed half-hour grid, then shift(1) BEFORE every rolling
operation. Require full window observations. Zero/near-zero denominators
produce missing values; no infinities. Do not bridge gaps by compressing rows.
Calendar flags: weekday0-6, week-of-month1-5 (days1-7,etc), quarter1-4.
These are known calendar indicators, not discovered market-regime labels.
Total211 tree columns plus training-derived missing indicators.
Tree preprocessing: train-only medians, standardization and8-sigma feature
clipping, matching the existing helper. Raw input availability is inherited
from the original package, not re-audited here. Causal perturbation and prefix
checks run before training.

LIGHTGBM GRID
Depth/leaves: (3,4),(3,6),(4,4),(4,8),(4,12).
Learning rates .01,.02,.03:15 configurations per direct/residual variant.
Depth2 cannot satisfy both num_leaves>=4 and num_leaves<2**max_depth, so is
excluded. Leaf counts obey the strict inequality. Min child samples30;
there are thousands of eligible rows, so the under500-row exception is unused.
Subsample .75, subsample_freq1, colsample_bytree .75, reg_alpha2, reg_lambda5.
Up to1000trees, early stopping patience50 using RMSE on the most recent20%
of eligible side-specific training origins. Training labels must end strictly
before this early-stop slice begins. Preprocessing and initial Ridge see only
the earlier fit subset. Refit on the full matured training window with the
chosen iteration count, never using outer evaluation for early stopping.
Targets/residuals receive affine standardization; L1/L2 apply on that normalized
scale. This preserves squared-error means but is an explicit regularization
scale choice. No target clipping/log transformation/tail weighting.

RESIDUAL MODEL AND FALLBACK
Ridge uses existing UP3/DOWN7 inputs and alpha=.01*training_rows, as the saved
Ridge benchmark. Fit Ridge first; train LightGBM on y-Ridge(x).
Early-stop residuals use the inner-fit Ridge for BOTH fitting and validation.
Final refit residuals use full-training Ridge; final = Ridge+LightGBM residual.
Residuals are in-sample training residuals, not cross-fitted residuals.
Final magnitudes projected nonnegative. Both sides predict all origins.
Three preceding purged validation windows choose the best configuration for
each variant and the fallback among Ridge/direct/residual, separately per side,
by pooled conditional-magnitude MSE. Final signed metrics are also reported.
The final refitted Ridge signed predictions must reproduce the existing saved
benchmark within1e-6bps; mismatch stops the run.

COMPUTE
720 inner candidate fits plus16 outer selected fits. Each includes an
early-stop LightGBM and full-history refit, totaling1472 LightGBM fits, plus
Ridge fits. Substantial CPU work; no runtime guarantee or resume mode.
Existing local Code/tree_model_deps LightGBM installation is reused.

OUTPUT
Code/experiments/lightgbm_residual_48h_v1/<timestamp>/
SUMMARY.txt and cycle_signed_metrics.csv: every model's RMSE,MAE,IC,bias,
direction accuracy, and RMSE/MAE gain versus Ridge (positive means better).
Cycle1=Jan-Mar2024, Cycle2=Apr-Jun, Cycle3=Jul-Sep, Cycle4=Oct-Dec.
Jan2025 tail reported separately; ALL includes all18481 origins.
cycle_2_3_4_mean_metrics.csv: unweighted arithmetic mean of the three cycle
RMSEs/MAEs, distinct from pooled ALL errors.
cycle_magnitude_metrics.csv: UP/DOWN metrics for every cycle and overall.
prediction_vs_actual_with_direction.csv: predicted/actual bps,percent,direction.
predictions.csv: side means, probabilities, signed returns and actuals.
validation_metrics.csv, SELECTIONS.json, FIT_AUDIT.json, CONTRACT.json,
16 replay-checked model bundles and source snapshots.
COMPLETE.json written only after successful checks/output.

Same historically exposed evaluation; improvement must be measured, not
assumed or guaranteed. Family/features were informed by earlier research.
