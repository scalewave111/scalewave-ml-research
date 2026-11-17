120-HOUR UP/DOWN FEATURE AND MODEL SEARCH

Run from November/Week 2:
  python .\Code\fit_magnitude_models_120h.py

Input/causality audit only (no training):
  python .\Code\fit_magnitude_models_120h.py --check-inputs
Synthetic checks:
  python .\Code\check_magnitude_models_120h.py

This extends the later 72h market-feature/model experiments to 120h.
It does not reuse fitted 72h weights or their selected winners.
Targets: canonical up_magnitude_log_bps and down_magnitude_log_bps,
7200-minute horizon, all valid rows including zero magnitudes. No signed
return forecast and no second log transform of the targets.

FEATURE SEARCH
Three prespecified packs, chosen independently per target and quarter:
  leverage: 38 price/funding/basis/open-interest features
  flow_liquidation: 30 price/flow/historical-liquidation-activity features
  combined: 62 features
Price inputs: 120h volatility, normalized momentum, range position,
up-variance share; 168h drawdown and 336h jump-variance share.
Market means/deviations and interactions use past24h and120h windows,
at least80% half-hour snapshot coverage, excluding the origin snapshot.
Liquidation inputs describe historical activity rates, not current size.
The feature dictionary records all generated columns, including unused ones;
the contract lists actual candidate packs and selections list fitted features.
No univariate ranking or forced reuse of previously selected features.
HGB retains training-eligible columns (80% coverage,1000finite,varied);
neural packs require every input to have80% coverage and1000finite values.
All imputation/scaling uses the current training window only.

MODELS
HGB: absolute_error, depth3,7leaves,leaf128,60iterations,rate.05,L2=10,
63bins,seed48,no random early stopping;6000 evenly spaced estimator rows.
MLP16: ReLU,learningrate.002,AdamW decay.1.
MLP32/16: ReLU,learningrate.001,decay.03.
GRU8: learningrate.002,decay.1.
GRU16: learningrate.001,decay.03.
Neural models:24epochs,batch512,gradientclip1,seed48,4096 evenly spaced
estimator rows; full training rows supply preprocessing and median.
Neural output is a bounded residual around the training median,
0.75*max(median,1)*tanh(head), with original-log-bps MAE objective up to
a constant training scale. MLP uses current features; GRU uses exact
snapshots30/20/10/0h ago. Sequence construction uses the full X timeline
before filtering invalid labels; missing timestamps stay missing.
Predictions with less than80% finite required inputs fall back to median.

SELECTION AND EVALUATION
Each of5configurations crosses all3packs, with median-blend weights
.1/.25/.5/1 (reuses predictions). Four earlier3month validation windows.
Every training label ends STRICTLY before its validation/issuance boundary;
validation labels also mature before the window ends:120h purges throughout.
Lowest validation MAE is the diagnostic candidate. Guarded selected model
must improve pooled median MAE by0.1%, win/tie3/4windows, and lose no window
by more than2%; otherwise selected is the training median. A diagnostic
candidate is not automatically promoted based on evaluation scores.
Jan/Apr/Jul/Oct2024 expanding fits; October model retained for January2025.
Original48h input gap remains, supplemented by120h training-label maturity.
18193 evaluation origins per target;47808 valid total rows. The288 excluded
development labels comprise144invalid-price-path and144outside-release.
Up to480inner fits plus16final fits. Threads default2. No runtime promise
or automatic resume; a failed run retains its partial timestamp folder.

COMPARISONS AND OUTPUTS
Frozen 120h price-followup20261113T120541578982Z and long-feature
20261113T123515172594Z predictions are hashed and aligned for comparison
only; their fitted models are not refitted/replayed by this runner.
Also compares training median and mean; reports MAE,RMSE,Spearman IC by
quarter/January and pooled, separately for UP and DOWN.
Output: Code/experiments/magnitude_models_120h_v1/<timestamp>/
Saves predictions,metrics,selections,feature dictionary,validation trials,
quality audits,purge ledger,models with replay checks,source/input/guidance
hashes,package versions,CONTRACT,SUMMARY and final COMPLETE marker.
Requires existing Week1/October data and prior120h runs, sklearn and PyTorch.
Old files/results remain unchanged. This is repeatedly exposed historical
research; neither the model search nor validation guards guarantee gains.

STATUS
Prepared for user-run real training. Synthetic tests and actual input-only
audits are separate from fitting/evaluating models on real data.
