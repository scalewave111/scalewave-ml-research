200H JOINT MAE AND RMSE REFINEMENT

Run in PowerShell:
  cd "C:\Users\Siddh\Desktop\ML\November\Week 2"
  python .\Code\fit_dual_objective_200h.py
No-training input audit and existing-model replay:
  python .\Code\fit_dual_objective_200h.py --check-inputs
Real-input exclusion-metadata/export regression (no training):
  python .\Code\check_dual_objective_200h.py --check-real-inputs
Synthetic tests:
  python .\Code\check_dual_objective_200h.py

OBJECTIVE
One forecast per side must beat BOTH training MEAN and MEDIAN on MAE,
and the training MEAN on RMSE. These are two separate criteria, not interchangeable scores.
No sign forecast, target logarithm, tail trimming or target reweighting.
Canonical200h UP/DOWN excursions remain in log bps, including valid zeros.

STARTING MODELS
UP: saved price-followup20261113T121536996657Z, historical MAE411.963448.
All quarters retain original selected market features, with MAE HGB depth3,
7leaves,80iterations,leaf128,learningrate.05,L2=1.
DOWN: saved magnitude_models200h20261114T131905974842Z selected policy,
historical MAE380.375598. January median; April/July25%GRU8+75%median;
October/January2025 10%GRU16+90%median. GRUs retain38leverage features at
60/40/20/0h snapshots,24epochs,4096row estimator cap and original settings.
All eight quarterly artifacts are hash checked and replayed before search.
These references beat median MAE but did not beat mean RMSE.
The exact saved forecast is retained whenever replacement fails validation.

CHANGES
Fit paired histogram boosting models using absolute_error and squared_error.
Both use ALL mature training rows; no6000-row cap or outcome clipping.
Two capacity settings: depth2/4leaves/leaf256/60iterations and
depth3/7leaves/leaf128/80iterations; rate.05,L2=10,maxbins63,seed48,
no random early stopping. Training-only imputation/scaling/missing flags.
Features need80%finite coverage,1000finite rows,and variation in each fit.

Three update packs: UP that quarter's existing inputs / DOWN38leverage
features;12 compact long-price
features; those12 plus the focused combined price/market pack (69 unique
columns). Existing features are explicit anchors, not a new from-scratch
selection claim. Complete names and formulas are recorded in outputs/code.
Long-price pack covers volatility200/720h,jump share336h,net168h,range1440h,
upside variance share200/1440h,drift changes200vs400h and336vs1440h,
range400h,normalized momentum200h,and range position200h.

For each pair, test mean-model weights0/.25/.5/.75/1; then blend that
forecast with the existing anchor at update weights.25/.5/.75/1.
All20blend combinations reuse fitted predictions. Missing inputs below80%
finite at inference revert to the anchor; an empty admitted pack also does.

EARLIER-ONLY SELECTION
Four previous3month windows, strict200h maturity in training and validation.
Refit the corresponding HGB, GRU or median anchor inside each training window; never
use its outer fitted model to score earlier validation rows.
Choose by the worst relative MAE/RMSE ratio against both the two baselines
and the anchor. A replacement requires:
 - at least0.1%pooled MAE improvement versus the SMALLER mean/median
   baseline MAE in each window, pooled by row count (a stricter benchmark
   than separately comparing the two pooled baselines);
 - at least0.1%pooled RMSE improvement versus training mean;
 - no pooled MAE or RMSE degradation versus the refitted anchor;
 - win/tie at least3of4windows for EACH baseline criterion;
 - no window worse by more than2% on either criterion.
Otherwise selected remains the saved anchor, NOT a newly substituted constant.
Best validation candidate is still saved separately as a diagnostic.
Only selected/candidate are evaluated later; no outer-result winner switching.

QuarterlyJan/Apr/Jul/Oct2024 fits,October retainedJanuary2025;
17873evaluation origins/target. Original48h input gap plus200h purges.
384inner challenger fits +32inner anchor fits; up to32final challenger fits.
No automatic resume. Full-row squared-error fitting may take longer than
the prior capped experiment. Default2CPUthreads; --threads can change this.

OUTPUT
Code/experiments/dual_objective_200h_v1/<timestamp>/
Predictions,MAE/RMSE/IC by period,dual_baseline_checks.csv with an explicit
beats_median_mae,beats_mean_mae,beats_mean_rmse and beats_both flags,validationtrials,selections,actual fitted feature lists,
quality audits,fit ledger,saved models and replay,hashes,versions,SUMMARY,
and COMPLETE only after successful end-to-end checks.
The selected_features.csv lists the union of anchor and correction inputs;
final_quality files identify admitted correction features specifically.

STATUS/LIMITS
Prepared for user-run training. Input-only audits and synthetic model tests
do not establish improved real-data performance. Prior chosen feature lists, GRU configurations and blend weights
and historical evaluation have been repeatedly examined; this remains
adaptive research. Earlier dual acceptance does not guarantee later wins.
Source availability/masking and overlapping-label limitations remain.
