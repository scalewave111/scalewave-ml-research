120H JOINT MAE AND RMSE REFINEMENT

Run:
  python .\Code\fit_dual_objective_120h.py
No-training input audit and existing-model replay:
  python .\Code\fit_dual_objective_120h.py --check-inputs
Real-input exclusion-metadata/export regression (no training):
  python .\Code\check_dual_objective_120h.py --check-real-inputs
Synthetic tests:
  python .\Code\check_dual_objective_120h.py

OBJECTIVE
One forecast per side must beat the training MEDIAN on MAE and the training
MEAN on RMSE. These are two separate criteria, not interchangeable scores.
No sign forecast, target logarithm, tail trimming or target reweighting.
Canonical120h UP/DOWN excursions remain in log bps, including valid zeros.

STARTING MODELS
UP: saved price-followup20261113T120541578982Z, historical MAE317.631061.
DOWN: saved long-feature20261113T123515172594Z, historical MAE301.106182.
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

Three packs: that quarter's existing winning inputs;12 compact long-price
features; those12 plus the focused combined price/market pack (69 unique
columns). Existing features are explicit anchors, not a new from-scratch
selection claim. Complete names and formulas are recorded in outputs/code.
Long-price pack covers volatility120/720h,jump share336h,net168h,range1440h,
upside variance share120/1440h,drift changes120vs240h and336vs1440h,
range240h,normalized momentum120h,and range position120h.

For each pair, test mean-model weights0/.25/.5/.75/1; then blend that
forecast with the existing anchor at update weights.25/.5/.75/1.
All20blend combinations reuse fitted predictions. Missing inputs below80%
finite at inference revert to the anchor; an empty admitted pack also does.

EARLIER-ONLY SELECTION
Four previous3month windows, strict120h maturity in training and validation.
Refit the anchor architecture/features inside each training window; never
use its outer fitted model to score earlier validation rows.
Choose by the worst relative MAE/RMSE ratio against both the two baselines
and the anchor. A replacement requires:
 - at least0.1%pooled MAE improvement versus training median;
 - at least0.1%pooled RMSE improvement versus training mean;
 - no pooled MAE or RMSE degradation versus the refitted anchor;
 - win/tie at least3of4windows for EACH baseline criterion;
 - no window worse by more than2% on either criterion.
Otherwise selected remains the saved anchor, NOT a newly substituted constant.
Best validation candidate is still saved separately as a diagnostic.
Only selected/candidate are evaluated later; no outer-result winner switching.

QuarterlyJan/Apr/Jul/Oct2024 fits,October retainedJanuary2025;
18193evaluation origins/target. Original48h input gap plus120h purges.
384inner challenger fits +32inner anchor fits; up to32final challenger fits.
No automatic resume. Full-row squared-error fitting may take longer than
the prior capped experiment. Default2CPUthreads; --threads can change this.

OUTPUT
Code/experiments/dual_objective_120h_v1/<timestamp>/
Predictions,MAE/RMSE/IC by period,dual_baseline_checks.csv with an explicit
beats_both flag,validationtrials,selections,actual fitted feature lists,
quality audits,fit ledger,saved models and replay,hashes,versions,SUMMARY,
and COMPLETE only after successful end-to-end checks.
The selected_features.csv lists the union of anchor and correction inputs;
final_quality files identify admitted correction features specifically.

STATUS/LIMITS
Prepared for user-run training. Input-only audits and synthetic model tests
do not establish improved real-data performance. Prior chosen feature lists
and historical evaluation have been repeatedly examined; this remains
adaptive research. Earlier dual acceptance does not guarantee later wins.
Source availability/masking and overlapping-label limitations remain.
