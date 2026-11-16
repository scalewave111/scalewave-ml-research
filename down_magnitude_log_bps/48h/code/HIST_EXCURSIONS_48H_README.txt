RUN FROM WEEK 2
python .\Code\fit_hist_excursions_48h.py

Optional: --threads 2 (default), --check-data-only, --self-test.
Resume an interrupted histogram run with --resume "C:\full\path\to\run".
Same code,input hashes,versions and threads required. Do not resume another
runner's directory. Completed fits are cached with checksum receipts.

SCOPE
48h canonical up_magnitude_log_bps and down_magnitude_log_bps, each on all
valid rows. Same48000input/18385evaluation origins as completed excursion
run20261113T093614902840Z;96binding-crossing exclusions preserved. Both can
be positive. No endpoint-direction filtering or probability weighting.
Targets and errors in LOG BPS. Actual label/origin/issuance equality and
reference prediction hash verified before any training.

MODELS
sklearn HistGradientBoostingRegressor, separate squared_error and absolute_error
models. Squared_error configurations select by earlier pooled MSE; absolute_error
by earlier pooled MAE. Never choose a loss retrospectively from outer scores.
Two settings per loss/feature pack:
small:depth2,4leaves,minleaf128,80iterations,rate.05,L2=10.
medium:depth3,7leaves,minleaf64,120iterations,rate.05,L2=5.
Both seed48,max_bins127,nativeNaN handling,training-only binning.
early_stopping=False:fixed iterations avoid sklearn random validation splits.
Targets are not clipped/scaled; forecasts clipped at0.
No LightGBM or monthly residual/forest fits in this new experiment.

FOUR FEATURE PACKS
raw13:unique union of original8 and package7 prior inputs.
raw13_price51:raw13 plus51past-price features (64total).
engineered235:existing causal E2 feature pack.
engineered235_price51:235 plus51 (286total).
Some51 columns overlap or correlate with E2 information; these are additional
representations, not51 wholly independent market measurements.

NEW PRICE FEATURES
At1h/4h/12h/24h/48h/168h:RV,upRV,downRV,signed netreturn,past5mclose-path range,
distance from that high,distance from that low,largest absolute5mreturn.
Plus time-normalized RV ratios1h/24h,4h/24h,24h/168h:48+3=51.
Strictly earlier return endpoints; no same-origin/future return use. Incomplete
windows stayNaN. Ranges are5mclose-based, not1m/intrabar highs or lows.
Reuses tested feature generator from4h follow-up, with new48h targets/splits.

SELECTION AND OUTPUT POLICIES
Four expanding quarterly outer fits Jan/Apr/Jul/Oct2024;October also covers
Jan2025. Three earlier2month validation windows at offsets9/6/3months.
All training labels end strictly before forecast window;48h purge.
hgb_rmse:best L2 HGB config/pack by pooled earlierMSE.
hgb_mae:best L1 HGB config/pack by pooled earlierMAE.
selected_rmse:retain trainingmean if best HGB fails to beat earliermeanMSE.
selected_mae:retain trainingmedian if best HGB fails to beat earliermedianMAE.
These fallback rules do not guarantee improvements on later data.
Also report earlier-selected best setting per pack/loss; this compares old
versus added features without selecting packs using outer outcomes.
Pack differences combine feature-block and selected-setting changes; they do
not prove individual feature importance or causality.

COMPARISONS
All previous excursion mean,median,fixedRidge,tunedRidge,LightGBM,E2,selected
forecasts are carried through unchanged on matching rows. Previous best
observed pooledRMSE:UPselected248.740640;DOWNfixedRidge280.029555.
Report pooled,cycles1-4,Jan2025,unweightedcycle2-4means,MAE,RMSE,IC,bias.
No fresh-data/significance claim; repeated historical discovery remains exposed.

OUTPUT
Code/experiments/hist_excursion_48h_v1/<timestamp>/
SUMMARY.txt,metrics.csv,cycle_2_3_4_mean_metrics.csv,predictions.csv,
POPULATION_AUDIT.csv,validation_metrics.csv,SELECTIONS.json,CONTRACT.json,
per-quarter forecasts,model bundles,checksum cache,source snapshots,COMPLETE.json.
All new nonconstant delivered models replayed after saving before completion.
At most384inner fits and64distinct outer fits before cache sharing; policies
reuse fitted candidates. No measured runtime or guaranteed error reduction.

PREPARATION STATUS
Synthetic L1/L2nonlinear-signal,purge,future-isolation,selection/fallback,
price directformula/causality/missingness and savedreplay checks passed.
--check-data-only validates source hashes,actual targets and causal features;
it does not fit models. No real-data training performed during preparation.
