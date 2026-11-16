FAST DOWN72h MARKET-STATE RUN

python .\Code\fit_market_state_down_72h.py --fast

If the previous full run is active in your terminal, Ctrl+C first. The fast
command starts a separate run; it does not change or resume an existing process.

32inner fits+up to4final fits versus144+4 in full mode. Extra Trees removed.
One MAE HGB:60iterations,depth3,7leaves,minleaf64,rate.05,L2=10,max_bins63.
Two packs:36price control and241market+price predictors before quality filtering.
Estimator fitting uses up to6000evenly-spaced chronological training rows.
Feature quality,imputation/scaling and baseline medians still use ALL mature
training rows. All four earlier validation windows and18385evaluation origins
remain intact; no validation/evaluation subsampling. Weights.5/1 and baseline
fallback unchanged. Smaller search/training sample can change accuracy.

New output: Code/experiments/market_state_down_72h_fast_v1/<timestamp>/.
No automatic resume. Omit --fast to retain the original full configuration.
Synthetic241feature fit+serialization replay took about3seconds on this machine;
this is not a full real-data runtime measurement. Loading,feature construction,
quality checks and validation add time. No real-data training run by assistant.

Test: python .\Code\check_market_state_fast_72h.py
