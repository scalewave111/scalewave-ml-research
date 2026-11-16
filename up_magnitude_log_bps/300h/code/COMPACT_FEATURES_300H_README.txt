300H: FRESH SMALL FEATURE SETS AND SIMPLE MODELS

From November/Week 2:
python .\Code\search_compact_features_300h.py
Input audit only (no fitting): append --check-inputs
Synthetic checks: python .\Code\check_compact_features_300h.py

SCOPE
First proposed experiment only: fresh feature selection and simpler models,
keeping quarterly expanding refits. Monthly updates are not mixed into it.
Same canonical300h BTCUSDT UP/DOWN excursions in log bps, including valid zeros;
17473 eligible evaluation origins;1008 invalid development labels excluded.
Every inner/outer fit uses origin+300h STRICTLY before prediction boundary.
Jan/Apr/Jul/Oct2024 refits; October retained for January2025.

FEATURES
Same54 LONG300 features from long_features_300h.py: trend/path, total/up/down
volatility, relative position and cross-window changes at168/300/720/1440h.
Exact formulas: LONG_FEATURES_300H_README.txt. Strictly past return endpoints;
complete five-minute windows; missing stays missing. No old F4Z input or prior
selected feature is mandatory. This changes selection and models, not the
measurement universe from the previous54-feature experiment.

SEARCH (independent per target and issue)
1. Start empty. Features require >=80% coverage, >=1000 nonmissing values and
   >1 unique value in every earlier TRAINING window.
2. Screen single features with all3 models on chronological evenly spaced
   training samples capped at4096 per window; full validation windows. Rank
   by minimum pooled MAE across models. Reject absolute Spearman>=.95 with
   retained features, calculated on earliest-training rows (>=1000 pairs).
   Shortlist at most12.
3. Each model greedily adds up to6 features using full mature training rows.
   Each addition needs >=.1% pooled MAE gain and >=2/3 window wins versus its
   preceding step. All visited nonempty subsets considered for final choice.
4. Eligible subsets must beat the median by >=.1% pooled MAE, win >=2/3
   windows, and lose no more than2% against median in ANY earlier window.
   Lowest pooled MAE wins; ties prefer fewer features then model name.
   Otherwise use training median. No forced minimum: 1-6 features or fallback.
Bounded greedy search is not exhaustive and may miss interaction-only signals.
Earlier-period guard does not guarantee future improvement.

MODELS
ridge_strong: sklearn Ridge alpha=1*training_row_count.
ridge_stronger: Ridge alpha=10*training_row_count.
shallow_tree: DecisionTreeRegressor criterion=absolute_error, max_depth2,
min_samples_leaf256, random_state48.
Ridge FITS squared error; all SELECTION uses MAE. Original training-only median
imputation/scaling/missing flags. Nonnegative predictions. No boosting, neural
network, history-window tuning or blending; median is explicit fallback.

OUTPUTS / CONTROLS
Code/experiments/compact_features_300h_v1/<timestamp>/
new_features = selected compact model or median fallback
previous = original price-follow-up selected prediction
previous_long_features = previous54-feature-pack selected prediction
training_mean / training_median = same mature quarterly baseline populations
Frozen references: pipeline20261113T115754520122Z/followup20261113T121655587383Z;
excursion_long_features_300h_v1/20261113T123508595159Z.
Prediction hashes/keys/truths/issuance verified. Original mean/median values
recomputed from mature targets and checked. Old models NOT refit/replayed here.
Saves predictions, period/pooled MAE/RMSE/IC, feature dictionary/quality/shortlist,
validation trials/search paths/selections, inner/outer fit windows, models and
replay/hash checks, source/input/guidance hashes, versions and source snapshots,
SUMMARY/COMPLETE. Only COMPLETE marks success. No report or automatic resume.
Every run has a new folder. Old data/results/models preserved.

BUDGET
At most486 screening fits plus513 forward fits per target/quarter, then final
fit; <=8000 fits total with defaults before cache savings. Two threads default.
No runtime guarantee. Existing Week1 dependencies and canonical inputs required.

LIMITS
All evaluation dates have already been reviewed. Reused validation and highly
overlapping300h labels limit evidence. This is adaptive historical research.
No unused later dataset was supplied; independent confirmation remains future
work. No promised improvement.
