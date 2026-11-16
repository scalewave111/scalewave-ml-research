200H TRANSFORMED-INPUT REFINEMENT FOR MEAN RMSE AND MEAN/MEDIAN MAE

PowerShell:
  cd "C:\Users\Siddh\Desktop\ML\November\Week 2"
  python .\Code\fit_transformed_dual_200h.py
Input/replay only:
  python .\Code\fit_transformed_dual_200h.py --check-inputs
Synthetic tests:
  python .\Code\check_transformed_dual_200h.py

STARTING FORECASTS
Latest diagnostic candidates from dual_objective_200h_v1/20261114T135118300960Z:
UP MAE398.096148/RMSE551.274734; DOWN MAE378.914575/RMSE551.301760.
They beat both mean/median MAE, but not mean RMSE539.383149UP/525.553711DOWN.
These are exploratory starting candidates, not previously guard-approved
models. Original guarded controls and previous dual selected forecasts remain
separate output columns. Fallback preserves the latest candidate exactly.
Both the8original and8candidate bundles are hash checked and replayed.

DIAGNOSTIC MOTIVATION
Read-only recomputation on saved candidate predictions found mean forecast
errors -170.771985UP/-160.856149DOWN logbps. Highest10%actual magnitudes
account62.6471%UP/67.2950%DOWN of candidate squared error. Their contribution
to candidate-minus-mean MSE is16244.548392UP/42323.624263DOWN versus total
12969.651438UP/27726.927977DOWN; lower90% partly offsets that disadvantage.
Groups and errors are hindsight descriptions, not selection rules or proof
of a cause. No global evaluation-period bias correction is fitted.

MODELS AND INPUTS
Five paired configurations: raw, signed-log, rank, signed-log+squares
linear models; plus shallow HGB on signed-log+squares inputs.
Mean head: Ridge alpha=.01*n; absolute-loss head: SGD epsilon=0,
elasticnet alpha=.001,l1ratio=.1,adaptive eta=.005,1500iterations,
averaged weights,seed48. HGB heads:absolute/squared loss,depth2,4leaves,
leaf256,60iterations,rate.05,L2=10,bins63,seed48,no early stopping.
Every challenger uses ALL mature training rows. Train-only median/IQR
imputation/scaling; empirical midrank references fitted on training only;
quadratic basis adds individual squares of signed-log inputs. Standardize
basis on training and clip transformed inputs to[-8,8]; append missing flags.
Targets only positively constant-scaled; no targetlog, tail trimming or
outcome clipping. Final forecasts constrained nonnegative.
Three update packs: original UP anchor inputs / DOWN38leverage features;
12long-price features;69long-price+market features. Admission per fit requires
80%finite coverage,1000finite observations and variation. Inference with
less80%finite required values reverts to starting candidate.
Mean-head weights0/.25/.5/.75/1; update weights.25/.5/.75/1 versus candidate.

VALIDATION
Four earlier3month windows; strict200h maturity for training and validation.
The complete starting candidate is refitted inside each earlier window:
original HGB/GRU/median plus old L1/L2 update and its fixed blend settings.
Saved outer fitted estimators never generate earlier validation predictions.
Historical choices of features, architecture and blends remain adaptive.
An update requires .1%pooledMAEgain against the smaller mean/median MAE in
each window pooled by row count, .1%RMSEgain against mean, no pooled MAE
or RMSE degradation versus refitted starting candidate,3/4wins each, and
no window more than2%worse. This MAE gate is stricter than comparing two
pooled baseline scores separately. Rank by worst relative MAE/RMSE ratio.
Selected fails closed to the starting candidate. Best validation challenger
also saved separately. Neither is switched based on evaluation scores.

Same47488eligible rows/17873evaluation origins pertarget;608exclusions.
QuarterlyJan/Apr/Jul/Oct2024;October retainedJanuary2025. Original48h input
gap plus200h maturity purges. Up to960inner challenger fits,96inner fits
for nested starting models,32final challenger fits. No automatic resume;
full-run runtime unmeasured. Two CPUthreads by default.

OUTPUT
Code/experiments/transformed_dual_200h_v1/<timestamp>/
Predictions,metrics,three explicit baseline-win flags,selections,trials,
quality,feature dictionary/actual input union,fitledger,saved models/replay,
input/source/guidance hashes,versions,SUMMARY and COMPLETE.

STATUS
Synthetic allfive model-pair fits,transform formulas/ties/missingness/batch
isolation,strictpurges,gates/outerisolation/fallback,8model pipeline/replay
passed. Nested starting-model refits proven independent of saved outer
estimators on synthetic data. Actual input audit and all16replays passed.
No real-data fitting by assistant. Historical adaptive research; another
experiment does not guarantee simultaneous baseline wins.
