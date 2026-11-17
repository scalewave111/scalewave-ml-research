RUN
cd "C:\Users\Siddh\Desktop\ML\November\Week 2"
python .\Code\tune_e2_direct_controls_48h.py

INDEPENDENT EXPERIMENTS: EACH STARTS FROM UNCHANGED E2
Control: all235 original inputs, original side/issue settings, seed48.
Horizon branch: separately add six price measures at4h,12h or72h; original
settings and seed unchanged. Each variant has241 columns. No combined additions.
Capacity branch: original235inputs, depth/leaves3/4,3/6,4/8,4/12 crossed with
min_child_samples20/30/50; original learning rate and other settings unchanged.
Seed branch: original235inputs AND original settings; average48/148/248 versus48.
No feature removal, and no branch uses another branch's selected configuration.

17 declared variants: unchanged +3horizon +12capacity +1seed ensemble.
Three separately selected policies H_selected,C_selected,S_selected also saved.
Each selector includes unchanged E2, uses per-side pooled magnitude MSE across
three earlier purged validation windows. No global outer-selected winner.

Fixed: expanding UP3/DOWN7 Ridge, monthly chronological OOF residual targets,
expanding correction history, full-strength nonnegative side predictions,
direction probabilities, rows, quarterly issue dates, target units ordinarybps.
Targets remain conditional48h ENDPOINT magnitudes, not path excursions.
Horizon features: realized/upside/downside volatility,absolute net return,
sum absolute returns,cumulative log-price range. Strictly prior5m endpoints.
Early stopping: most recent20% residual origins,48h label purge,patience50,
max1000,full-history refit at selected iteration count. Three seeds each
choose iterations independently; ensemble averages nonnegative side means.
Training-only imputation/scaling and affine residual normalization retained.

E2 source: lightgbm_repair_experiments_48h_v1/20261113T053731575845Z.
Hashes and saved E2 replay verified. Rebuilt unchanged control must reproduce
saved E2 forecasts within1e-6bps. Source data/helper scripts are required.

OUTPUT
Code/experiments/e2_direct_controls_48h_v1/<timestamp>/
SUMMARY.txt: all variants overall and selected policies by cycle.
cycle_signed_metrics.csv: every variant,RMSE/MAE/IC/bias/direction accuracy,
gains vsE2 and Ridge. cycle_magnitude_metrics.csv: UP/DOWN metrics.
cycle_2_3_4_mean_metrics.csv: unweighted mean across those cycles.
tail_error_contributions.csv: training-P90 tail/nontail excess MSE versusE2.
prediction_vs_actual_with_direction.csv: rowwise predictions,actuals,directions.
predictions.csv,validation_metrics.csv,SELECTIONS.json,FIT_AUDIT.json,
OOF_AUDIT.json,CONTRACT.json,model bundles,source snapshots,cache.
Cycle2=Apr-Jun2024,3=Jul-Sep,4=Oct-Dec;Jan2025 separately;ALL includes it.
COMPLETE.json only after successful output/replay checks.

RESUME
python .\Code\tune_e2_direct_controls_48h.py --resume "C:\full\path\to\run"
Same --threads value (default2) and matching code/input/config signatures needed.
Hash-verified caches avoid repeating identical fits. Do not modify inputs or
scripts while running. Substantial CPU/disk use; no runtime guarantee.

CHECKS
--check-data-only checks actual input alignment/hashes/causal features, no fitting.
--self-test checks synthetic branch independence, feature retention, causality,
early-stop purge, seed averaging and bundle replay.
Both checks passed during preparation; no real-data training was executed.
Historically exposed evaluation. Any improvement remains exploratory until
confirmed on newer data; no guaranteed improvement.
