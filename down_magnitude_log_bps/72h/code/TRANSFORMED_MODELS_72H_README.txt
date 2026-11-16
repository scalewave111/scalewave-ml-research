72h FEATURE TRANSFORMATIONS AND ALTERNATIVE MODELS

Run from November/Week 2:
python .\Code\fit_transformed_models_72h.py

Input-only audit: add --check-inputs
Synthetic tests: python .\Code\check_transformed_models_72h.py

PURPOSE
Test feature representations and different model families after fresh-feature
and compact boosting failed to beat pooled median MAE. Both UP and DOWN are
evaluated independently; DOWN is the unresolved baseline target. Existing neural
UP predictions remain an unchanged comparison. No automatic replacement or
promotion is performed, and a baseline win cannot be promised.

INPUT GROUPS
Three prespecified groups from the new120-feature universe; no old F4Z/NEWPRICE
features, no inherited selected subsets, no greedy feature search in this run.
The short/multiday groups avoid dependence on the problematic60-day window,
but that research choice is informed by already observed diagnostics.
Volatility: hourly RMS24h, hourly RMS72h, volatility ratio24/72h, up variance share72h.
Path: range position72h, drawdown/vol168h, rebound/vol72h, trend efficiency168h.
Mixed: hourly RMS168h, normalized momentum72h, range position168h, jump share336h.
Each group has4features. All120 are built for compatibility; only these groups
are offered to models. Full exact names are in transformed_models_72h.PACKS and
the saved CONTRACT.json. Source formulas remain in feature_dictionary.csv.

FEATURE TRANSFORMS (CROSSED WITH EVERY GROUP AND MODEL)
All statistics are fitted ONLY on the relevant training window:
1. raw: (x-training median)/training IQR. This is an affine control, no nonlinear
   transform or clipping. IQR<=1e-8 or nonfinite uses scale1.
2. signed_log: sign(z)*log(1+abs(z)), z=the same median/IQR standardized value.
3. rank: 2*training empirical midrank CDF(x)-1. Ties use midpoint of counts below
   and at/below x; beyond-training-range values saturate at-1/+1. No recomputation
   of ranks using validation/evaluation batches.
Missing/nonfinite values use training median and explicit missing indicators.
Transforms are saved inside each model bundle. No target logarithm is applied.

MODELS
A. Additive spline median regression: sklearn SplineTransformer,4training-quantile
   knots per input,degree2,constant extrapolation,include_bias=False. Missing flags
   enter linearly. QuantileRegressor quantile=.5,L1 penalty alpha=.01,HiGHS solver.
   This learns smooth additive effects, without cross-feature interactions.
B. ExtraTrees:64 randomized trees,absolute_error criterion,max_depth5,
   min_samples_leaf64,max_features1,bootstrap4096rows/tree (or fewer if necessary),
   seed48,n_jobs1. Average of leaf-median tree predictions; not itself guaranteed
   to be a conditional median. Supports interactions and nonlinear responses.
Both fit y/max(training target median,1) for numerical scale, then restore logbps.
This constant scaling preserves the data-loss MAE objective. Spline regularization
is specified in these normalized target units. Predictions clipped nonnegative.

SELECTION
3groups x3transforms x2models =18 candidates per target/quarter; each trained in
3earlier purged windows. Each candidate tests model weights .25/.5/.75/1 against
the expanding training median, with no refit needed for blend weights.
Admission requires every input >=80% finite,>=1000 values and variation in all
3earlier training windows. An accepted model/blend must improve pooled earlier
MAE >=.1%, win2of3windows, and lose no window by more than2% versus median.
Choose lowest earlier pooledMAE among accepted candidates; else median fallback.
Comparison controls are never used as predictors or to select models. Selection
is saved before outer fitting. Candidate trials show transformation effects on
earlier validation; only the selected policy is fitted/scored on the outer period.
Do not treat the validation-best transform as independently confirmed.

SAME POPULATION AND EVALUATION
Canonical72h UP/DOWN logbps excursions,valid zeros included,48000eligible inputs,
18385evaluation origins. Strict origin+72h<boundary in inner/outer training;
quarterly Jan/Apr/Jul/Oct2024 expanding fits,October also serves January2025.
Earlier choices do not see later outcomes. Inherited initial48h boundary gap is
supplemented by the strict72h maturity rule. Original96 invalid labels excluded.
No changes to source datasets or previously saved experiments.

COMPARISONS AND OUTPUTS
Code/experiments/transformed_models_72h_v1/<timestamp>/
Compares selected new_features with previous_compact, previous_fresh, previous
price-followup tree, previous_neural, training_median and training_mean. Metrics
are MAE,RMSE,rankIC overall/perquarter/January2025. Controls checked for prediction
hashes,keys,truths andissuance; old models are not independently replayed here.
Reference fresh20261114T074315743848Z,compact20261114T080702116226Z,
neural20261113T123304519186Z,price-followup20261113T113500606206Z.
Saves trials,quality,selections,features,transform/model bundles,replay hashes,
fit windows,exclusions,source snapshots,contract,versions,SUMMARY andCOMPLETE.
Requires existing local helper modules; not a standalone portable package.
Up to432inner fits +8final fits. Default2threads,no automatic resume. Quantile
regression runs on full eligible rows; ExtraTrees caps bootstrap per tree only.

STATUS AND LIMITS
Synthetic transform formulas/ties/missingness/batch isolation, both families with
all transforms, blends, serialization, selection gates/signal/zero fallback,
outer isolation,72h maturity and fullsave-score-replay pipeline passed.
Real-input-only audit passed48000rows/18385origins/120generated features and
required group columns. No real-data fitting executed by assistant.
Historically exposed adaptive research,not fresh confirmation. Missing information
cannot be recovered by ranks/logs; transformations may lose useful absolute-scale
information. Overlapping72h labels are dependent. No guaranteed baseline gain.
