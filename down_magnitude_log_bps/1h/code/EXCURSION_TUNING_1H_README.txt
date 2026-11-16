1-HOUR EXCURSION ERROR-REDUCTION EXPERIMENT

cd "C:\Users\Siddh\Desktop\ML\November\Week 2"
python .\Code\tune_excursions_1h.py

Input/reference replay only (no training):
python .\Code\tune_excursions_1h.py --check-inputs
Synthetic tests: python .\Code\tune_excursions_1h.py --self-test

Uses the completed 1h price follow-up from pipeline20261113T101316741479Z.
Both targets retain all18481 development origins, including zero excursions,
and original quarterly expanding training with strict1h label-maturity purges.
October model predicts January2025; original41h boundary gap remains.

Tests current selected features, current plus all51price features, and price-only.
No new price formulas or future inputs. Existing reference cards are checked
against prediction/model hashes and replayed on the original evaluation inputs.
All candidate preprocessing/binning uses training rows only; native NaN handling.

18 candidate fits per target/cycle/window: 3packs x2losses x3settings.
Losses absolute_error and squared_error; learning_rate.04,max_bins127,seed48,
no random early-stopping split. Settings(iterations,depth,leaves,minleaf,L2):
 (120,2,4,128,10), (160,3,7,64,5), (160,4,12,128,10).
Each candidate is also tested as a50/50 blend with the current model.
Current model is refitted on each inner training window to supply fair controls.
Up to432inner candidate fits,24inner reference fits and16outer candidate fits.
No measured runtime estimate. Defaults to2threads; --threads may be changed.

Three earlier quarterly validation windows select two separate policies:
 mae_selected: >=0.1% pooledMAE reduction and>=2/3window wins versus current,
               with pooled validationMSE no worse than current.
 rmse_selected: >=0.1% pooledMSE reduction and>=2/3window wins versus current;
               this policy can sacrifice MAE.
If no candidate qualifies, that policy retains current predictions unchanged.
Choices are written before outer fitting/scoring. Outer data never selects a model.
Validation guards do NOT guarantee later error improvements.

Outputs: Code/experiments/excursion_tuning_1h_v1/<timestamp>/
metrics.csv and SUMMARY.txt compare current,mae_selected,rmse_selected,mean,median.
All errors are logbps; periods include each2024quarter and January2025 separately.
Also saves predictionCSV,all inner trials,selection/specs,selected model bundles,
model hashes,fit windows,input/source hashes,source snapshots and COMPLETE.json.
Blended replay requires the unchanged reference model as well as new bundle;
reference path and hashes are recorded. Only COMPLETE.json marks full success.
No automatic resume; rerunning starts a fresh output directory.

Prepared and tested on synthetic data plus real-input/reference-replay checks.
No real-market training performed during preparation.
This remains adaptive historical research on exposed evaluation periods.
Previously selected feature packs reuse earlier validation; no nested or fresh
confirmation claim. Original X availability/provisional-source limits remain.

