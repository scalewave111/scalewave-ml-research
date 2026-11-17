RUN FROM WEEK 2:
  python .\Code\tree_interaction_signed_search_48h.py --trials 96

Optional complete input/feature preflight, without fitting models:
  python .\Code\tree_interaction_signed_search_48h.py --check-data-only
DuckDB microsecond and prepared nanosecond UTC indexes are normalized to
nanoseconds before exact comparison; no timestamp rounding or row dropping.

This supersedes the earlier search strategy. All eight tree families remain:
Decision Tree, Random Forest, Extra Trees, Gradient Boosting, Histogram Gradient
Boosting, XGBoost, LightGBM and CatBoost. Optional packages use Code/tree_model_deps.
No fitting happens on import. User executes real-data training.

WHAT CHANGES
1. New source inputs: 14 existing4h F4Z percentiles covering liquidation severity,
   side pressure, concentration, venue breadth, cross-asset severity, flow pressure,
   flow activity, CVD efficiency, basis level/change, funding level/change and OI
   leverage unwind/range position. Exact names are in the source and contract.
2. Original8 inputs +20 causal price features +58 explicit interaction/change/
   regime inputs, plus14newmarket inputs =100 candidate columns.
   Exact-time1h/4h/24h changes; centered pair products; market-feature times past
   volatility; liquidation threshold flags at.75/.90/.99 and their interactions
   with past24h volatility/net move. Missing exact lag timestamps stay missing.
   These are features supplied to trees, not guaranteed interpretable regime gates.
3. Seeded feature-subset search across original,market,price,interaction,all pools.
   Subsets retain40/65/85/100percent of a pool, minimum3 columns. No outcome-based
   feature selection before splitting. First candidate per family uses original8.
4. Independently sampled model parameters: depth2/3/4/6/8, leaf32/64/128/256/512,
   tree count96/192/320, learning rate.01/.03/.06/.1, L2 1/5/20/100, L1 0/.1/1,
   row sampling.6/.8/1, feature sampling.5/.75/1, leaves7/15/31, CatBoost random
   strength.1/1/5. Only applicable parameters are used by each family; CatBoost
   uses symmetric depth controls, XGBoost uses min_child_weight, LightGBM uses
   min_child_samples. No automatic early stopping or hidden evaluation fit.
5. Most important: joint UP/DOWN model selection minimizes SIGNED RETURN MSE
   on identical earlier validation timestamps, not separate magnitude scores.

LEAKAGE / OBJECTIVE
Each candidate still trains mean magnitudes separately on UP/DOWN historical
rows with squared-error loss, affine target normalization, and no target clipping
or tail reweighting. Both side predictions are computed on ALL validation and
evaluation origins. Final return=p_up*UP_mean-(1-p_up)*DOWN_mean in ordinarybps.
No realized-side branch selection. Negative estimated magnitudes are projected0.
Train-only median imputation, standardization, feature clipping8sigma and flags.

Three validation origins windows per outer quarter, starting9/6/3months earlier
and lasting2months. Training label_end strictly precedes each fit cutoff; all
validation labels mature before outer model issue. Validated historical
direction probabilities are loaded from saved purged2023folds and the frozen
2024development model. Pipeline hashes and max training-label timestamps are
checked. Missing validation probabilities cause a failure, never a substitution
with realized directions or an in-sample direction forecast. Direction itself
is not retrained and remains v45-style, not exact reportedv45.

Within each family, choose its best UP/DOWN pair on pooled signed validation MSE.
The joint_selected policy also permits different families on the two sides.
All candidate pair MSEs and selections are saved BEFORE outer fitting/scoring.
Model choices may change per quarterly refit. No outer-best automatic promotion.
Ridge remains an unchanged saved benchmark on the identical18481origins.

COMPUTE AND RESUME
--trials96 means96 TOTAL configurations balanced over8families, not96perfamily.
That means2304inner fits and at most72outer fits; pair comparisons reuse cached
validation predictions, without fitting a model for each pair. This can take
many hours depending on hardware. --trials48 reduces the search. --threads2
is default. Trial counts must be at least the number of selected families.
Partial per-candidate validation results are checkpointed with hashes.
To resume, use the SAME original settings plus the output directory:
  python .\Code\tree_interaction_signed_search_48h.py --trials 96 --resume "Code\experiments\tree_interaction_signed_search_48h_v1\<timestamp>"
Completed validation candidates are reused; interrupted candidates and final
fits may be repeated. Code/data/config changes are rejected on resume.

OUTPUTS: Code/experiments/tree_interaction_signed_search_48h_v1/<timestamp>/
SUMMARY.txt / signed_metrics.csv: return errors,rankIC,direction accuracy andUP rate
prediction_vs_actual_with_direction.csv: forecast/actual percent,directions,error
signed_return_predictions.csv: side means,probabilities,combined forecasts
SELECTIONS.json: chosen pairs,features,hyperparameters and signed validation RMSE
<quarter>/signed_validation_pair_mse.csv: full UP-candidate x DOWN-candidate table
<quarter>/*validation.joblib: cached predictions/keys/training-cutoff audit
<quarter>/*final.joblib: saved selected models with preprocessing and cutoffs
FIT_AUDIT.json / CONTRACT.json / snapshots / COMPLETE.json: provenance and checks

LIMITS
This is the same historically exposed population. Broad tuning/pair search can
overfit validation. Improved historical scores are not independent confirmation
and do not guarantee beating Ridge or future results. F4Z publication latency is
inherited from the source package; new price features are causality checked.
Mean return predictions need not match each realized return. Sign correctness is
reported for the final return forecast, not merely the direction classifier.
