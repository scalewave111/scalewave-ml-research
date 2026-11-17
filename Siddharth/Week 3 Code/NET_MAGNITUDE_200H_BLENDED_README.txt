200H NET MAGNITUDE: UP200H APPROACH FOR FOUR INDEPENDENT FAMILIES

Run all four from PowerShell:
cd "C:\Users\Siddh\Desktop\ML\November\Week 3"
python Code/fit_net_magnitude_200h_blended.py

This supersedes the plain horizon adaptation as the requested approach.
HGB, ExtraTrees, MLP and GRU run sequentially, independently across every
Jan/Apr/Jul/Oct issue period. Model families are never blended together.
No previous net200h results or saved UP model weights are required.

One family: add --families hgb (or extra, mlp, gru).
Input audit, no fitting: add --check-inputs.
Non-fitting synthetic checks: python Code/check_net_magnitude_200h_blended.py

TARGET / DATA
Canonical net_magnitude_log_bps = abs(10000*ln(P[t+200h]/P[t])).
Reads checksum-verified horizon_minutes=12000 BTCUSDT labels via the existing
net200h loader. Keeps invalid-label/source-binding crossing exclusions and
strict 200h training maturity. No raw data are written or labels recalculated.
Defaults to Datasets/canonical_receipts/OUTPUT_FILES.json; override with
--canonical-manifest PATH. --canonical-root and --week2-code remain supported.
Actual input availability/access has not been checked for this new runner.

UP200H FEATURE APPROACH
Copies the established pure feature builders into local modules:
net_200h_long_features.py and net_200h_market_features.py.
Three candidate packs, selected independently by family and quarter:
leverage38 / flow_liquidation30 / combined62.
Six shared price inputs: 200h volatility per sqrt hour, normalized momentum,
range position and up-variance share; 168h drawdown in volatility units;
336h jump-variance share. Market means/deviations and interactions use
24h/200h history: funding, basis, open interest, flow, historical liquidation
activity rates/breadth. These inputs retain their original source meanings.
Full source X timeline is used before outcome filtering. Historical windows
exclude the origin. GRU snapshots are 60/40/20/0h ago, with exact timestamps.
The saved feature dictionary includes unused generated features; candidate
packs and fitted feature names are saved separately in contract/selections.

Preprocessing uses all mature training rows: >=80% finite coverage, >=1000
finite values and variation; training median/IQR, clipping +/-8, missing flags.
Default input transform is signedlog: sign(z)*log1p(abs(z)); flags unchanged.
--transforms raw signedlog quadratic searches all three; quadratic adds each
signedlog value squared, with no interaction expansion or target transform.
Inference requires >=80% observed retained inputs at every relevant snapshot;
otherwise that head returns the training median. Outputs are nonnegative.

PAIRED OBJECTIVES AND BASELINE BLENDS
Every candidate fits both L1 and L2 heads in the SAME family.
head = (1-q)*L1 + q*L2, q in [0,.5,1].
forecast = w*head + (1-w)*training_anchor,
w in [.1,.25,.5,.75,1], anchor mean or median.
Predictions are reused across all weights; blends do not require extra fits.
Targets are median-centered/std-scaled using full mature training rows;
there is no second target log, evaluation bias correction or tail trimming.

Fast budgets are default: estimator cap2000; HGB40rounds/depth3/7leaves/
minleaf128/rate.05/L2=10/bins63; ExtraTrees24/depth4/minleaf32/maxfeatures.75/
bootstrap1024; MLP32 ReLU16epochs/AdamWlr.001/decay.1; GRU16hidden12epochs/
AdamWlr.001/decay.03. Neural batch256/gradientclip1; seed48; CPUthreads2.
--full-search restores cap6000 (or --row-cap), HGB60rounds, ExtraTrees64/
depth6/bootstrap4096, GRU24epochs, and three MLP capacities:
32/16hidden40epochs decay.01;64/32hidden60epochs decay.01;
32hidden60epochs decay.1. Full and fast use the same three feature packs.
Neither speed nor predictive improvement is guaranteed.

VALIDATION / SELECTION
Four earlier three-month windows beginning 15/12/9/6 months before issuance.
Training labels end strictly before each fitting cutoff. Validation labels
end strictly before their own window end and before issuance.
Rank the worst pooled relative score among MAE/median, MAE/mean, RMSE/mean.
Acceptance requires >=.1% pooled improvement for ALL three comparisons,
>=3/4 window wins/ties and no >2% window loss for each comparison.
If none qualifies, selected is the training mean. Best unrestricted
validation candidate is diagnostic and saved separately. No later outcomes
choose features, models, transforms, weights or whether a guard is relaxed.
This extends the UP method; it does not reproduce its nested saved-model
anchors or rank/Ridge challengers. Each family is freshly fitted to net Y.

OUTPUTS / COST
Code/experiments/net_magnitude_200h_<family>_upstyle_fast_v1/<timestamp>/
Report/net_magnitude_200h_<family>_upstyle_fast_v1/<timestamp>/
Full-budget roots use upstyle_v1. Old outputs remain unchanged.
Includes contracts/settings/input hashes, feature dictionary, population
audit, validation trials, frozen choices, selected features, paired saved
models, serialization replay, predictions, MAE/RMSE/bias and COMPLETE.
Default:384 inner head fits across all4 families, up to64 final head fits.
Full budget with one transform:576 inner head fits. Each extra transform
multiplies this search cost. No automatic resume. Progress prints each head.
Selected may be a baseline policy, not a pure-family forecast.

STATUS
Generated and syntax checked; non-fitting synthetic feature causality/prefix,
transform, blend, dual-metric guard and strict-boundary checks passed.
No model fitting, real-data input audit, training or inference executed by
the assistant. End-to-end fitting and serialization replay remain untested
until a user run. Historical exposed evaluation; no fresh-test/gain claim.

MARKET SOURCE FIX (2026-11-16)
The legacy loader provides a reduced inventory. Required11market sources
now load directly from original masked X_train/X_development files, aligned
to that loader timeline. Actual source-only read verified48096timestamps
and all11columns. Full input audit/fitting remain unexecuted. Missing values
are preserved. Temporary-Parquet source regression and causality checks pass.
