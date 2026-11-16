72h FRESH-FEATURE DIAGNOSTICS AND COMPACT TREE EXPERIMENT

From C:\Users\Siddh\Desktop\ML\November\Week 2:
1. python .\Code\diagnose_fresh_features_72h.py
2. python .\Code\fit_compact_trees_72h.py

Input-only audit: python .\Code\fit_compact_trees_72h.py --check-inputs
Synthetic tests: python .\Code\check_compact_trees_72h.py

The diagnostic performs saved-model inference, not fitting. It verifies input,
source, prediction and model hashes; keys, truths, issuance, baselines, model
replay and every saved period metric. It reports prediction bias (prediction
minus actual), mean/median/standard deviation, training versus later errors,
feature missingness/P10/P50/P90/rank correlations, and stored earlier validation
gain versus later gain. Q2 is reported alongside every other period. Inner
models were not saved, so inner errors are stored records, not independently
replayed. In-sample errors and feature shifts do not establish causation.
Diagnostic output: Code/experiments/fresh_diagnostics_72h_v1/<timestamp>/.

The experiment does NOT consume diagnostic output or use Q2 to choose settings.
It uses only the same120 fresh causal hourly-price features. No old F4Z/NEWPRICE
inputs or previously selected subsets are forced. Search starts empty for each
target/quarter. See LONG_FEATURES_72H_README.txt for formulas and missingness.

TREE CONFIGURATIONS
Absolute-error histogram gradient boosting, 80 iterations, learning rate .05,
L2=5, seed48, early stopping disabled:
  depth1: max2 leaves, minimum256 rows/leaf
  depth2: max4 leaves, minimum256 rows/leaf
  depth3: max8 leaves, minimum512 rows/leaf
Preprocessing learned on each training window: median imputation, missingness
flags and standardization. Nonnegative tree predictions. Blends with that
window's training median at model weights .25/.5/.75/1. Median fallback is weight0.
No Ridge or neural fitting in this experiment.

SELECTION
Same strict72h label maturity, same18385 evaluation origins, quarterly expanding
Jan/Apr/Jul/Oct2024 fits, October retained for January2025. Three earlier purged
validation windows. Quality requires80% finite coverage,1000 readings and
variation in each earlier train. All3 tree settings screen each single feature
using up to4096 evenly spaced training rows. Earliest-training absolute Spearman
>=.95 removes redundancy; shortlist12. For each configuration/blend separately,
greedy forward search visits up to3features, with no forced minimum.
Each addition needs >=.1% pooledMAE gain and2/3window wins. A visited subset
is eligible only if it also beats the median by those rules and no earlier
window loses >2% MAE. Choose lowest earlier pooledMAE among eligible subsets;
otherwise median fallback. This guard does not guarantee later-period wins.
Full earlier/final training rows used after screening; fitted predictions cached
across blend weights when subsets match. Default2threads; no automatic resume.
The bounded search may miss interactions. The screen itself is another adaptive
selection step. Up to about18,200 fits at defaults before cache savings.

COMPARISONS
Fresh-feature run long_features_72h_v1/20261114T074315743848Z, old tree
price-followup and previous neural, plus training mean/median. Old predictions
are hashed and aligned as controls, not selection inputs; compact runner does
not replay old models (the diagnostic replays the fresh reference models).
All baseline columns retain the original quarterly schedules.

OUTPUT
Code/experiments/compact_trees_72h_v1/<timestamp>/
Predictions and MAE/RMSE/rankIC by quarter,January2025,pooled; selections,
search paths, quality, validation trials, feature dictionary, chosen features,
models with replay checks/hashes, fit windows, source snapshots, contract,
SUMMARY and COMPLETE. Model strings such as depth2@0.5 include blend weight.
All errors in logbps; targets remain canonical UP/DOWN excursions including zeros.

This is adaptive research on previously inspected history. No guaranteed gain
or independent confirmation. Historical72h labels overlap. Preserve old outputs.

STATUS: Synthetic gate/purge/signal/outer-isolation/fallback/blending/configuration,
serialization, save-score pipeline and inference-only diagnostic replay/metric
reproduction/corruption rejection passed. Actual-input-only compact audit passed
48000rows/18385origins/120features. No real-data fitting or full real diagnostic
execution performed by the assistant.
