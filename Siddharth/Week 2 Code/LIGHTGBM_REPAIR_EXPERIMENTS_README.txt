RUN
cd "C:\Users\Siddh\Desktop\ML\November\Week 2"
python .\Code\lightgbm_repair_experiments_48h.py

Optional --threads 2 (default). --check-data-only performs input/causality checks
without fitting. --self-test exercises synthetic models only.
Resume after interruption with the exact printed/created run directory:
python .\Code\lightgbm_repair_experiments_48h.py --resume "C:\full\path\to\run"
Use the same --threads value. Code/input/config hashes must match; completed
runs cannot be resumed. Cache receipts verify hashes; invalid cache entries
are recomputed. Do not edit scripts/data while a run is active.

FOUR CONTROLLED STAGES
E1: Current211 tree inputs + monthly chronological out-of-fold Ridge residuals.
    Expanding residual history; conditional-magnitude validation MSE selection.
E2: E1 plus24 longer-horizon price features; still expanding history.
E3: E2 plus correction-history selection:180days,365days,expanding.
E4: E3 candidate pool, jointly select UP/DOWN configurations and correction
    weights [0,.1,.25,.5,1] by signed-return validation MSE. Require pooled
    improvement and improvement in at least2 of3 earlier windows. Otherwise
    keep Ridge for both sides. This is an exploratory stability gate, not a
    statistical significance test. E4 changes objective, shrinkage and gating
    together; it does not isolate those three changes individually.
E1-3 use full-strength corrections without the E4 gate to expose each stage's
measured effect. All stages train separate UP/DOWN means in ordinary bps and
retain fixed direction probabilities. Prior direct/residual results included.

RIDGE AND OUT-OF-FOLD RESIDUALS
The baseline remains expanding-history UP3/DOWN7 Ridge, alpha=.01*n, exactly
the existing benchmark. History selection changes the LightGBM correction
training window only; it does not change Ridge. This keeps weight0 identical
to the benchmark and separates history changes from a changed baseline.
At each calendar-month start, fit side-specific Ridge using labels ending
strictly before that start, then predict that month's side-specific origins.
Residual = observed positive magnitude - raw Ridge prediction. Require1024
matured Ridge training rows; earlier warmup rows have no residual and are not
used. Every residual becomes trainable only after its48h outcome matures.
Original all-history OOF records are filtered by cutoff before every tree fit.
Monthly updates are a declared approximation to forward errors; final
evaluation Ridge still follows the four original quarterly issue dates.
No outer outcomes choose hyperparameters or correction weights.

FEATURES
E1 preserves the prior211 inputs: lagged13 market inputs,7/14/30half-hour
rolling signals, calendar flags. E2-4 add, at24h/48h/7days/30days:
  realized volatility = sqrt(sum of squared5m log returns)
  separate upside and downside realized volatility
  absolute net log return and sum of absolute log returns
  range of cumulative log-price path, in log bps (not dollar high-low range)
All windows exclude return endpoints at/after origin, require full history,
and preserve missing windows. Canonical return hash and48h label checks,
causal prefix/future-perturbation checks run before fitting. Total235 columns
for E2-4, plus train-only missingness indicators. Original feature availability
is inherited from the package, not independently re-audited here.

MODELS AND SELECTION
Same15 constrained configurations as the prior runner: depth3/leaves4or6,
depth4/leaves4or8or12, learning rates.01/.02/.03. Min child30, subsample.75,
freq1,colsample.75,L1=2,L2=5. Depth2 is incompatible with4<=leaves<2**depth.
Latest20percent of eligible residual origins select iterations, max1000,
patience50. Purge labels before that validation slice. Preprocessing is fitted
only on the earlier fit rows. Refit on all eligible matured residual rows at
the selected tree count. Affine residual normalization; penalties apply on
that normalized scale. No tail weighting, target trimming or log target fit.
Final magnitude=max(0, raw_Ridge + weight*tree_correction).
Three earlier validation windows start9/6/3months before issue and use first
2months of origins. Saved purged direction probabilities must cover every
origin; their model hashes and label cutoffs are checked. Both side forecasts
are generated for EVERY validation/evaluation origin, no realized-side routing.
E4 considers all UP/DOWN pairs on identical rows, never chooses by outer scores.

COMPUTE AND STORAGE
60 feature/history/config candidates, two sides, reused across four outer
issues. Each tree fit includes early stopping and a full-history refit.
There are up to720 unique inner tree configurations/cutoffs, plus selected
outer fits not already cached. Substantial CPU and disk usage; no runtime
guarantee. Hash-verified cache saves models and validation predictions for
resume and avoids repeating identical fits across overlapping windows.
Local LightGBM dependency installation is reused. Existing Week1 data,
Week2 helper scripts, frozen direction artifacts and previous completed run
20261113T050448624486Z must remain available. Not a standalone script.

OUTPUT
Code/experiments/lightgbm_repair_experiments_48h_v1/<timestamp>/
SUMMARY.txt: Ridge, old direct/residual, E1/E2/E3/E4 pooled and cycle results.
cycle_signed_metrics.csv: RMSE,MAE,IC,bias,direction accuracy,gains versus Ridge.
cycle_magnitude_metrics.csv: separate UP/DOWN magnitude metrics.
cycle_2_3_4_mean_metrics.csv: unweighted mean of the three cycle RMSEs/MAEs.
Cycle2=Apr-Jun2024,3=Jul-Sep,4=Oct-Dec; January2025 is separate, ALL includes it.
tail_error_contributions.csv: tail uses training-only90th-percentile magnitude
threshold at each issue; reports tail/non-tail contributions to excess MSE.
prediction_vs_actual_with_direction.csv: actual/forecast bps,percent,direction.
predictions.csv, validation_metrics.csv, OOF_AUDIT.json,
SELECTIONS_AND_FIT_AUDIT.json, CONTRACT.json, replay-checked bundles and cache.
COMPLETE.json appears only after all checks and output generation succeed.

TEST STATUS
Real-input-only checks passed48096rows,235features,18481evaluation origins.
Synthetic OOF maturity/prefix, causal price windows, joint selection/fallback,
early stopping and model replay passed. No real-data model training performed
during preparation. This remains exposed historical research, not fresh-data
confirmation. Improvements are to be measured, never guaranteed.
