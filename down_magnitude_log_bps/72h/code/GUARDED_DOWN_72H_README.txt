GUARDED DOWN72h: RESIDUALS, SIZING, STABILITY AND FOUR-WINDOW VALIDATION

Run from November/Week 2:
python .\Code\fit_guarded_down_72h.py

Input/stability-only audit: add --check-inputs
Synthetic tests: python .\Code\check_guarded_down_72h.py

SCOPE AND BASELINES
DOWN only; UP models/results are untouched. Same canonical72h excursion target,
valid zeros,48000eligible inputs,18385evaluation origins and original exclusions.
No raw dataset edits. Quarterly Jan/Apr/Jul/Oct2024 fits; October also servesJan2025.
The primary reported benchmark remains the original quarterly training median.
The adjustment anchor is an online180day median: historical label MATURITY times
in[t-180days,t),strictly excluding maturity at t. At least48labels required;
online expanding median used if the180day window has fewer. First48-history
rows cannot supply residual training targets and are omitted from learned fits;
the quarterly control still uses ALL eligible mature labels.
Early evaluation labels can enter this online baseline only after72h maturity.
This online information policy differs from a frozen quarterly median; both
baselines are reported and both must be beaten in earlier learned-model selection.

TRAINING-ONLY FEATURE STABILITY
Start from all120fresh features; no fixed old selected subsets. For EACH fit:
* Form three trailing nonoverlapping3-calendar-month blocks ending at the latest
  training timestamp+30minutes. These are trailing blocks, not necessarily Q1/Q2.
* Each feature needs >=300finite observations and >=80%coverage in each block.
* Max minus min block coverage must be <=10percentagepoints.
* Spearman correlation with actual DOWN must be all>.01 or all<-.01 across blocks.
* Rank survivors by minimum absolute block correlation; take up to6.
Strict sign filtering is a user-requested heuristic; it can remove useful nonlinear
or regime-dependent features. This is not proof of stability on future data.
For quantile mapping, this screen uses only the older core training subset, not
the held-out calibration outcomes. Missing inputs are train-median imputed with
missing indicators. Empirical percentile ranks are fitted on training inputs only.
No survivor means that family cannot compete in that validation fit.

FIVE MODEL FAMILIES
1 residual_linear: predict (actual-causal rolling median)/training target median.
  SGDRegressor epsilon_insensitive with epsilon0 = absolute-error data loss;
  elasticnet alpha.01,l1_ratio.5; max_iter3000,tol1e-5,early_stoppingFalse,
  invscaling rate eta0.001,power.25,seed48,shuffleFalse. Constant target scale
  floor1; restored to original logbps before scoring. L1/L2 penalize coefficients;
  intercept is not penalized by sklearn. Regularization does not force zero output.
2 residual_tree: same residual target with absolute-error histogram boosting,
  depth2,4leaves,minleaf256,80iterations,rate.05,L2=10,seed48,no early stopping.
3 quantile_map: train residual_tree on older core. Hold out the last90days of
  mature training observations for calibration; core labels must mature before
  calibration starts. Need1000core/300calibration rows. Core model stays frozen
  after calibration. Rank baseline+predicted residual against held-out prediction
  scores; map percentile to empirical calibration DOWN quantile. Clip percentile
  to[.05,.95] to limit extremes. This is MARGINAL quantile mapping, not a calibrated
  conditional median, and can worsen MAE; held-out validation must justify it.
4 bins3 and5 bins5: training actual-target quantile cutoffs define3or5size bins;
  duplicate cuts collapse. Histogram boosting classifier with depth2,4leaves,
  minleaf256,80iterations,rate.05,L2=10,seed48. Predict most likely class and assign
  that class's training target median. Single-class training returns its median.
  Classification uses log loss; final selection still uses original-unit MAE.

SHRINKAGE, DEADZONE, DYNAMIC GATING
Raw forecast minus current rolling median is the candidate adjustment.
Test weights .25/.5/1 and deadzones0/.05/.10 of training target IQR(floor1logbps).
Adjust only when abs(raw adjustment)>=deadzone*IQR. A large adjustment is NOT
calibrated confidence; these thresholds are hypotheses checked in validation.
Dynamic quality multiplier:
* zero if any selected input is missing/nonfinite now;
* zero if any selected input's past7day missing rate exceeds training rate by
  >10percentagepoints, or fewer than48 historical feature rows are available;
* otherwise 1-.75*fraction of inputs outside their training P1/P99 range.
All recent windows exclude current/future feature rows. Current values themselves
are available and checked separately for missingness/outside-range status.
Forecast=max(0,rolling median + weight*quality*adjustment),subject to deadzone.
Static-weight controls are tested alongside dynamic gating; earlier validation
chooses whether gating/deadzone helps. Therefore a selected strategy may use
deadzone0 or static weighting. Trial records distinguish those cases explicitly.

FOUR EARLIER PURGED WINDOWS
Each issue uses four previous3month validation windows, rather than the old3.
All training labels mature strictly before validation starts, and validation
labels strictly before their window ends. Selection uses no outer-period outcome.
Require >=.1% pooled MAE gain AND beat/tie in >=3of4windows AND no window >2%worse
against BOTH quarterly median and online rolling median. Choose best eligible
pooled score. Rolling median alone can win if it passes against quarterly median;
otherwise quarterly median is fallback. Four windows are not independent because
72h labels overlap. This gate is stricter but cannot guarantee later superiority.
If a selected family has no eligible features at its final fit, its forecast
falls back to the rolling anchor; this unavailability is recorded explicitly.

OUTPUTS AND CONTROLS
Code/experiments/guarded_down_72h_v1/<timestamp>/
Predictions contain selected, rolling/quarterly median, mean, previous neural,
fresh tree, compact tree and transformed-model controls; effective adjustment
weight and rolling sample count. MAE/RMSE/IC pooled/byquarter/January2025.
Each cycle saves feature_stability.json, all validation_trials.json, selection,
final model bundle and hash. Bundle contains transforms,feature audit,parameters,
calibration references when relevant. Saved replay must match before COMPLETE.
Root includes fit windows,definitions,exclusions,source snapshots,input hashes,
instruction hashes,versions,contract,SUMMARY andCOMPLETE. Earlier model artifacts
are not saved individually; earlier feature/calibration audit and scores are saved.
Controls are prediction-hash/key/truth/issuance checked, not old-model replayed.
Latest reference transformed_models_72h_v1/20261114T085824921737Z.
No automatic resume; up to80inner fits+4finalfits, fewer if features unavailable.
Requires existing local helper modules and datasets; not standalone.

STATUS
Synthetic causal rolling/boundary,4windowpurge,sign/coverage stability,all five
families,held-out calibration,missingness/shift/deadzone/blend,replay,gate,
outer isolation,fallback andfullsave-scorepipeline checks passed. Real-input-only
audit passed48000rows/18385origins; no real-data fitting executed by assistant.
Historical adaptive research; unused later data still needed for confirmation.
The median itself has forecast error; it merely avoids feature-driven adjustments.
