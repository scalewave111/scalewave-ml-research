RUN
cd "C:\Users\Siddh\Desktop\ML\November\Week 2"
python .\Code\fit_package_median_hybrid_signed_48h.py

Optional: --threads 2 (default), --check-data-only, --self-test.
Resume after interruption:
python .\Code\fit_package_median_hybrid_signed_48h.py --resume "C:\full\path\to\run"
Use identical threads/code/inputs. Completed runs cannot resume. Artifacts are
reused only with matching run signature and per-model checksum receipt.

MODEL
Separate UP/DOWN models using the exact seven columns and order from
Code/ScaleWave_Magnitude_Model/features.csv. First two inputs feed
QuantileRegressor(quantile=.5,alpha=0,solver='highs'). All seven feed three
RandomForestRegressors,128trees each,absolute_error,seeds48/17/91,
bootstrap=True,max_features=1.0.
Training-only median imputation/mean/std scaling and interleaved missing flags
match the package. No8-sigma clipping or target standardization.
Clip pair and each forest forecast at zero, average forests, then blend.

Fixed package quarterly settings, applied to BOTH sides:
Jan2024:depth5,leaf64,nominal bootstrap16384,forest weight.5
Apr2024:depth5,leaf64,nominal bootstrap16384,forest weight.5
Jul2024:depth5,leaf128,bootstrap4096,forest weight.25
Oct2024:depth3,leaf128,bootstrap4096,forest weight0
Settings read from package selection cards. No new tuning/weight selection.
October forest is fitted and reported as a component, but hybrid equals pair.

NECESSARY ADAPTATIONS
Original package: overall absolute endpoint log-bps magnitude.
This runner: conditional positive ordinary-bps endpoint magnitude, per side,
as explicitly requested for comparison with E2 signed forecasts.
Convert signed_log_bps to10000*expm1(signed_log_bps/10000),then take absolute.
Train UP on positive endpoint outcomes,DOWN on negative. Both predict EVERY
evaluation origin. This is not canonical upward/downward path excursion.
Existing v45-style probabilities yield p_up*UP_hybrid-p_down*DOWN_hybrid.
These are weighted median-oriented forecasts, NOT calibrated expected returns.

sklearn requires integer max_samples<=training_rows. Side splitting yields
fewer than16384rows in JanUP14979,JanDOWN14636,AprDOWN16235. Bootstrap draws
are capped to those counts; other fits retain the exact requested counts.
Nominal/effective counts are explicit in FIT_AUDIT and saved artifacts.
No claim of exact original-package numerical reproduction.

DATA AND EVALUATION
Week1 Playground seven F4Z columns and prepared48h labels, same48096input rows
and18481evaluation origins. Strict label_end<quarterly fit cutoff. Expanding
side-specific history, no training on current/future fold outcomes.
Reference E2:lightgbm_repair_experiments_48h_v1/20261113T053731575845Z.
Input hashes, E2 prediction hash, exact origins/actuals/probabilities checked.
Package historical hyperparameters were development-selected; evaluation
remains historically exposed. No guaranteed win against E2 or Ridge.
8median fits +24forests of128trees. Absolute-error forests can be slow.

OUTPUT
Code/experiments/package_median_hybrid_signed_48h_v1/<timestamp>/
SUMMARY.txt;cycle_signed_metrics.csv;cycle_magnitude_metrics.csv;
cycle_2_3_4_mean_metrics.csv;predictions.csv;
prediction_vs_actual_with_direction.csv;FIT_AUDIT.json;CONTRACT.json;
eight replay-checked model bundles and receipt files;source snapshots.
Comparators:hybrid,pair,forest,training-side medians,training-side means,E2,
Ridge,zero. Reports RMSE,MAE,IC,bias,direction accuracy,gains versusE2/Ridge.
Cycle1Jan-Mar2024,2Apr-Jun,3Jul-Sep,4Oct-Dec;Jan2025separately;ALL includes it.
COMPLETE.json written only after successful output and replay checks.

STATUS
Actual-input-only check passed. Synthetic median/forest objectives,seeds,
preprocessing,bootstrap cap,blend,zero-weight and saved replay passed.
No real-data training performed during preparation.
