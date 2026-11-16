120H TRANSFORMATIONS: JOINT MAE AND RMSE SEARCH

Run from C:\Users\Siddh\Desktop\ML\November\Week 2:
  python .\Code\fit_transformed_dual_120h.py
Audit and existing-model replay only:
  python .\Code\fit_transformed_dual_120h.py --check-inputs
Synthetic tests:
  python .\Code\check_transformed_dual_120h.py
Real input/export regression (no training):
  python .\Code\check_transformed_dual_120h.py --check-real-inputs

TARGET
Same120h UP/DOWN canonical magnitudes, in log bps, all valid zeros.
One prediction must beat median MAE and mean RMSE. No target log transform,
rank transform, winsorization or tail deletion. Targets are divided by a
positive training-median constant for numerical scaling and predictions
are restored before evaluation; this preserves each data loss up to scale.

INPUT TRANSFORMATIONS
All medians,IQRs,rank reference arrays,basis means and standard deviations
are fitted inside EACH training window and saved for inference.
Missing raw inputs are replaced by training medians; missing flags retained.
1. raw: (x-training median)/training IQR, clipped[-8,8]. Affine control.
2. signed_log: sign(z)*log(1+abs(z)), where z is median/IQR standardized.
3. rank:2*training empirical midrank CDF-1; ties use left/right insertion
   midpoint, future extremes map to[-1,1]. No evaluation-batch fitting.
4. quadratic: signed_log values plus their individual squared terms.
   This is an additive nonlinear basis, not all pairwise interactions.
Signed-log bases are clipped[-8,8] before squaring; all resulting bases
and missing flags receive training mean/std scaling and final[-8,8]clip.
Input clipping affects predictors only. Zero IQR/std uses scale1.

MODELS
Four linear pairs (one per transform):
 - L1 side: SGDRegressor epsilon_insensitive with epsilon0 (absolute loss),
   elasticnet alpha.001,l1_ratio.1,adaptive step.005,max1500iterations,
   tol1e-5,n_iter_no_change15,averaged weights,seed48,no validation split.
 - L2 side: Ridge squared error,alpha.01*training-row-count.
An additional HGB pair uses the quadratic basis,absolute/squared losses,
depth2,4leaves,leaf256,60iterations,rate.05,L2=10,63bins,seed48.
All mature rows are used; no estimator row cap. These model labels identify
training objectives; calibrated conditional means/medians are not assumed.
Monotone input transforms can leave tree splits largely unchanged; linear
and quadratic models supply alternative relationships for this experiment.

FEATURES AND SEARCH
Same3packs as the preceding dual-objective run: existing per-quarter
anchor inputs,12long-price features,69long-price+market features.
Training-only quality:80%finite,1000observations,more than one value.
Mean-side weights0/.25/.5/.75/1 and update weights.25/.5/.75/1 reuse fits.
Blend correction with the existing saved anchor; insufficient prediction
coverage(<80%) falls back to anchor. Separate UP/DOWN and quarterly choice.
Four previous3month validation windows,strict120h maturity throughout.
Same joint gate:0.1%pooled gain vsmedianMAE andmeanRMSE,no degradation
vsrefittedanchor on either pooled metric,3/4wins each,no>2%windowloss.
Rank candidates by worst relative MAE/RMSE versus baselines and anchor.
If no candidate qualifies,retain the saved original anchor exactly.
Best validation candidate is also saved diagnostically; no outer-score
promotion or gate relaxation. Original48h input gap remains plus120hpurges.
QuarterlyJan/Apr/Jul/Oct2024;October held throughJanuary2025.

COMPARISONS/OUTPUT
18193evaluation origins/target. Old price-UP/long-feature-DOWN anchors are
hash checked/replayed; previous dual run20261114T132002835184Z selected
and diagnostic forecasts are hash/key/truth aligned for comparison only.
Previous candidates are not used as trained anchors or new training labels.
960inner challenger fits+32inner anchor fits;up to32final challenger fits.
Most challenger fits are linear; no runtime estimate or automatic resume.
Output: Code/experiments/transformed_dual_120h_v1/<timestamp>/
Saves metrics,predictions,dual_baseline_checks.csv,validationtrials,
selections,raw feature lists,admission quality,serialized transforms and
models,fit ledger,source/input/guidance hashes,versions,SUMMARY,COMPLETE.
Quadratic basis ordering:transformed values,squares,missing flags.
Other basis ordering:transformed values,missing flags.

STATUS
User-run real training. Synthetic tests and input/replay audits do not
establish a baseline win. Historically adaptive feature anchors and exposed
evaluation remain; transformations may fail to improve either criterion.
Previous code/results are preserved; no full experiment launched by assistant.
