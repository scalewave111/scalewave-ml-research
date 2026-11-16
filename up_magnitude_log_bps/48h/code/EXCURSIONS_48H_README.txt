RUN FROM WEEK 2
python .\Code\fit_excursions_48h.py

Optional --threads 2 (default), --check-data-only, --self-test.
Resume with --resume "C:\full\path\to\excursion\run" and identical code,
inputs,versions,threads. Outputs are separate from all endpoint experiments.

TARGETS
up_magnitude_log_bps =10000*ln(maximum future48h price/starting price).
down_magnitude_log_bps =-10000*ln(minimum future48h price/starting price).
Both nonnegative, both can be positive for the SAME row. Train BOTH targets
on all valid rows. No UP/DOWN endpoint filtering or direction probabilities.
Metrics are LOG BPS; do not compare them to previous ordinary-bps signed errors.

DATA
Canonical Handoff2 y/horizon_minutes=2880/symbol=BTCUSDT partitions are read
with DuckDB and verified against OUTPUT_FILES hashes. Exact timestamps join
the existing masked playground X plus prior causal price features. No Y rebuild.
Only VALID,finite,nonnegative labels; missing outcomes never imputed.
Canonical source-binding policy excludes paths crossing2024-10-29T00:01Z.
Input check:48096original rows ->48000valid;18481evaluation ->18385.
All96 exclusions are source-binding-crossing paths, saved in population audit.
Uses existing historical half-hour playground origins, not every canonical
minute and not a claim of all rows in the full source release. Future-derived
labels and direction probabilities are never features. Raw data unchanged.

MODELS AND SELECTION
1.Training mean and median constants for each target/fold.
2.Fixed Ridge alpha=.01*n, retaining prior UP3/DOWN7 input lists as candidate
  designs, now fitting ALL valid rows for each new target.
3.Tuned Ridge: original8,package7,E2engineered235 features x four alpha
  multipliers .001/.01/.1/1 (alpha=multiplier*n). Training-only imputation/scaling.
4.Direct LightGBM squared-error target models, three constrained settings.
5.E2-style LightGBM correction of monthly out-of-fold fixed-Ridge errors,
  using the same235engineered features, expanding residual history,3settings.
  This adapts E2 to new targets/population and narrows its original15setting
  search. No new monthly forests; monthly Ridge refits are inexpensive.

LightGBM settings:depth3/leaves4/rate.01;depth3/leaves4/rate.03;
depth4/leaves8/rate.02. Seed48,minchild30,sampling.75,L1=2,L2=5.
Each fit uses latest20percent earlier data for early stopping with48h label
purge,1000tree maximum,patience50,then refits full mature history at that count.
Direct target/residual normalization follows existing E2 fitter.
Predictions clipped at0; targets are not clipped or winsorized.

Three earlier two-month validation windows start9/6/3months before each outer
issue. Strict label_end<fit cutoff. Pooled magnitude MSE selects best config
per family/target; selected policy compares all candidates including mean.
Median is reported as an MAE baseline, not selected using a conflicting loss.
RMSE is primary for these mean-oriented models; MAE,IC,bias also reported.
Outer issues Jan/Apr/Jul/Oct2024. October model also predicts Jan2025 tail.
Feature sets inherited from endpoint work are hypotheses, not validated
excursion feature rankings. Exposed historical evaluation, no guaranteed gain.

OUTPUT
Code/experiments/excursion_models_48h_v1/<timestamp>/
SUMMARY.txt,metrics.csv,cycle_2_3_4_mean_metrics.csv,predictions.csv,
POPULATION_AUDIT.csv,validation_metrics.csv,SELECTIONS.json,
OOF_UP_AUDIT.json,OOF_DOWN_AUDIT.json,CONTRACT.json,
checksum-verified cache,model bundles,source snapshots,COMPLETE.json.
Reports each target separately:pooled,cycles1-4,Jan2025,unweightedcycles2-4.
Saved models are replay-checked before completion. No expected-return or
trading output is constructed from the two excursion predictions.

COMPUTE
At most456inner model fits before cache reuse (312Ridge,144LightGBM), plus selected
outer models and cheap monthly Ridge fits. Each LightGBM job has early stopping
and refit. This is much smaller than monthly median-forest retraining, but
runtime has not been benchmarked; no promised completion time.
No real-data training was executed while preparing this runner.
