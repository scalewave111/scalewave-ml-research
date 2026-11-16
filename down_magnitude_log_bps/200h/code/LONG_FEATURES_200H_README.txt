NEW LONG-HORIZON FEATURES: 200 HOURS ONLY

PowerShell:
cd "C:\Users\Siddh\Desktop\ML\November\Week 2"
python .\Code\discover_long_features_200h.py

Input audit without training:
python .\Code\discover_long_features_200h.py --check-inputs
Synthetic checks:
python .\Code\check_long_features_200h.py

This is a fresh feature experiment, not an established improvement. Existing
models, source datasets and results are preserved. Real-data fitting is user-run.

WHAT CHANGED
No old selected features are mandatory. The candidate universe is replaced by
116 derived features designed to describe slower price and market conditions:

1. 84 price features: 12 measurements at 24,72,168,200,336,720,1440 hours.
   Net log move, realized volatility, upside/downside semivolatility, past range,
   distance from the past high (current drawdown), distance from past low (rally),
   position in range, signed trend efficiency, trend/volatility, upside-minus-
   downside variance balance, and largest five-minute squared-return share.
2. 12 regime features: volatility-rate ratios and recent-vs-earlier trend-rate
   differences for 24/200,72/200,168/720,200/720,336/1440,720/1440-hour pairs.
3. 20 market features: past 7-day and 30-day mean/std of existing masked 4-hour
   basis-level percentile, funding-level percentile, flow-pressure percentile,
   OI-range-position percentile, and active-liquidation-severity percentile.
   These summarize percentile histories; they are not raw funding rates, dollar
   liquidation amounts, or signed flow. Exact source columns/formulas are saved.

Price windows end strictly before the forecast origin and require a complete
five-minute return window. Relative log paths use five-minute endpoints, not
intrabar highs/lows. Unknown returns invalidate affected windows; no fabricated
prices. For a flat path, normalized features use a small denominator floor.
Market histories use the original half-hour X inventory independently of label
validity, exclude the origin, and require >=80% expected observations. No filling.
Missing features remain NaN; earlier-training quality checks can reject them.

TARGET AND COMPARISON
Same canonical BTCUSDT horizon_minutes=12000 UP/DOWN excursions in log bps,
including valid zeros. Same 17873 eligible evaluation origins. The 608 invalid
development labels remain excluded and are saved in excluded_labels.csv.
Independent UP/DOWN fits in Jan/Apr/Jul/Oct2024; October also predicts Jan2025.
Every inner/outer training label ends strictly before issuance. The inherited
48-hour input gap remains, with the additional required 200-hour maturity purge.

Same model families as the original feature search, to isolate a feature change:
Ridge alpha=.01*n; MAE HGB80 iterations,rate.05,leaves7,depth3,minleaf128,L2=1,
seed48,no early stopping. Training-only median imputation, missing indicators,
scaling and feature quality (80% finite,1000 valid readings,variation per fold).
Three earlier quarterly validation windows. Single-feature absolute-error trees
screen; redundancy uses earliest-training Spearman .9; shortlist20; greedy
forward/removal search up to8 features per family. No feature is forced in.

The learned candidate must beat BOTH mean and median on earlier pooled MAE by
>=.1% and win >=2/3 windows against each. Otherwise selected uses whichever
constant (training mean or median) has lower earlier pooled MAE. This validation
gate cannot guarantee future improvement. No outer-outcome tuning or selection.
The raw candidate is saved separately, including when search returns a median.

Reference: excursion_pipeline_200h_v1/20261113T115534387046Z/
price_followup/20261113T121536996657Z. Source hashes, predictions, truths, origins,
issuance and baselines are checked. Old models are not replayed in this runner;
their saved forecasts are compared. All new model artifacts are replay-checked.

OUTPUTS
Code/experiments/long_features_200h_v1/<timestamp>/
predictions.csv: candidate,selected,previous,training_mean,training_median,actual.
metrics.csv: pooled and quarterly/Jan MAE,RMSE,Spearman IC for all five policies.
feature_definitions.csv, selections.json, per-cycle quality/shortlist/trials/
removals/policy/model files, fit_windows.csv, CONTRACT.json, SUMMARY.txt.
COMPLETE.json is written only after all eight fits and new-model replays finish.
No automatic resume; each invocation writes a new run directory.

LIMITS
Historical evaluation and validation have already been examined. Overlapping
200-hour targets reduce effective independent observations. Bounded greedy
search can miss interactions. Original F4Z availability/funding limitations
remain. No independent confirmation, guaranteed baseline win, or significance
claim. No external macro/on-chain data or new source-availability certification.

VERIFIED PREPARATION STATUS (2026-11-13)
Synthetic formula/flat-path/missingness/future-isolation/purge/selection/replay checks passed.
Actual-input-only audit passed:47488eligible rows,17873development origins.
96price candidates pass earlier-training quality in every quarterly fit; all20market
summary candidates fail coverage and will be excluded, not filled or forced in.
No real-data fitting executed; new-model performance remains unknown.

