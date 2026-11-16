300-hour UP/DOWN magnitude feature and model search

FAST MODE (added 2026-11-14):
Stop an existing full run with Ctrl+C, then restart:
cd "C:\Users\Siddh\Desktop\ML\November\Week 2"
python .\Code\fit_magnitude_models_300h.py --fast

Fast mode: combined62 pack only; HGB, MLP16, GRU8, each with L1 and L2 loss.
192 inner fits instead of960 (80% fewer), plus up to16 final fits.
HGB40 iterations/3000 estimator rows; neural8 epochs/2048 estimator rows.
All four validation windows, all evaluation origins, preprocessing/baseline
training rows, strict300h purges and joint MAE/RMSE gates remain unchanged.
Smaller search/training budget can change accuracy. Wall-clock speedup is not
benchmarked. Input audits still run. Existing processes do not adopt changes.
Separate output: Code/experiments/magnitude_models_300h_fast_v1/<timestamp>.
No resume; earlier output folders are preserved. Without --fast, full mode
below remains available. Test: python Code/check_magnitude_models_300h.py --fast

Run in PowerShell:
cd "C:\Users\Siddh\Desktop\ML\November\Week 2"
python .\Code\fit_magnitude_models_300h.py

Read-only input audit:
python .\Code\fit_magnitude_models_300h.py --check-inputs
Synthetic checks (no real-data fitting):
python .\Code\check_magnitude_models_300h.py

Targets: canonical BTCUSDT up_magnitude_log_bps and down_magnitude_log_bps
over 18000 minutes. All valid outcomes, including zeros. No additional log
transform. Independent choices for each target and quarterly issue.

Feature search: three prespecified packs, selected jointly with the model:
leverage (38), flow/liquidation history (30), combined (62). Training-only
quality admission. Six price inputs measure 300h volatility, normalized
momentum, range position and upside variance share, 168h drawdown in volatility
units and 336h jump share. Market statistics/interactions use 24h/300h past
windows; >=80% expected snapshot coverage. Liquidation measures are historical
activity rates, not liquidation size. Feature formulas saved in dictionary.
GRUs use exact snapshots at t-90h, t-60h, t-30h, t; no filling. Sequences are
assembled on the full X inventory BEFORE filtering future-label validity.

Models: HGB depth3/7 leaves/min leaf128/60 iterations/lr .05/L2=10;
MLP16 lr .002/decay .1, MLP32->16 lr .001/decay .03;
GRU8 lr .002/decay .1, GRU16 lr .001/decay .03.
Each model has absolute-error and squared-error variants: ten configurations.
Neural: 24 epochs, batch512, AdamW, gradient clip1, deterministic seed48;
bounded residual .75*max(training median,1)*tanh around training median.
Training-only median/IQR preprocessing, missing indicators, neural input clip8.
HGB uses at most6000 evenly spaced estimator rows; neural at most4096.
Preprocessing, quality checks and baselines use ALL mature training rows.
These caps/bounded neural outputs can limit performance on large tails.

Selection: four earlier three-month windows, strict origin+300h < next boundary
for BOTH training and validation. Model weights .1/.25/.5/1; baseline anchor
is median, half mean/half median, or mean. Rank by worst relative MAE/RMSE
against the stronger pooled baseline for each metric. MSE pooled by row count
BEFORE taking square root. Acceptance requires >=.1% gain on BOTH metrics
against BOTH constants, >=3/4 wins-or-ties per metric against both, and no
window >2% worse on either metric against either baseline. Else use median.
Missing-input coverage below80% returns the chosen constant anchor. Diagnostic
candidate remains separate; it is not automatically promoted if gates fail.

At most960 inner fits and16 final fits, CPU threads default2. No automatic
resume. Runtime is not benchmarked. Each run creates a fresh timestamp folder:
Code/experiments/magnitude_models_300h_v1/<timestamp>

Outputs: predictions.csv, metrics.csv (pooled and periods),
dual_baseline_checks.csv (four comparisons plus beats_both_on_both), selections,
validation trials, actual admitted features/quality, serialized model bundles,
replay verification, split ledger, exclusions, source/input hashes and versions.
Prior price and long-feature predictions are verified comparison-only controls.
Original datasets and historical outputs are unchanged.

The inherited X train/development inventory includes the original48h gap;
strict300h outcome maturity is additionally enforced at every fit boundary.
17473 eligible development origins per target; 504 invalid-path and504
outside-release development labels excluded. Historical evaluation has already
been examined repeatedly; it is not a fresh independent confirmation set.
Validation acceptance does not guarantee later improvement. No model can be
claimed to beat the baselines until the real run's scores demonstrate it.
