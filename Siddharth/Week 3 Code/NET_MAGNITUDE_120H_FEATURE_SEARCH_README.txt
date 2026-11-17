120H NET: NEW HORIZON-SPECIFIC FEATURES, MLP AND GRU ONLY

cd "C:\Users\Siddh\Desktop\ML\November\Week 3"
python Code/fit_net_magnitude_120h_feature_search.py

One family: --families mlp or gru. Input audit only: --check-inputs.
Fast default cap2000,MLP32hidden16epochs,GRU16hidden12epochs.
Three packs: volatility_endpoint16;trend_range22;slow_market40.
54 distinct causal base features total. Counts and exact formulas saved by run.
Past60/120/240/720h 5min RV and Gaussian absolute-endpoint proxy combine past
drift and variance extrapolated to120h. Proxy is a distributional hypothesis,
not a future label. Other volatility lookbacks24/60/120/240/168/336/720/1440h.
Trend/range uses60/120/240h momentum,range position,efficiency,upvariance plus
168hdrawdown and336hjump concentration. Slow market uses past24/120h funding,
basis,openinterest,unwind,flow and liquidation state. Original masked X columns
projected directly; missing values preserved. Full feature timeline before
outcome filtering; GRU36/24/12/0h snapshots. Canonical7200minute target unchanged.

Paired L1/L2 neural heads,head mixture0/.5/1 and baseline weights
.05/.1/.25/.5/.75/1 toward training median/mean. Earlier-only four3month windows
offset15/12/9/6months;strict120h maturity. Rank worst ratio MAE/median,
MAE/mean,RMSE/mean. Accept .1%improvement each,3/4wins/ties,no>2%windowloss
on each criterion. Otherwise meanfallback; best candidate diagnostic separately.
No baseline win guaranteed. Both metrics reported,history already exposed.

192inner head fits across both families,up to32outer head fits with default
one transform. Additional --transforms multiplies search cost. No runtime promise.
Outputs Code/experiments/net_magnitude_120h_<family>_feature_search_fast_v1/
and matching Report root; timestamped. No previous model required or overwritten.
Default local canonical manifest receipt; --canonical-manifest overrides.
Exact features/formulas,input hashes,validation ledgers,selected blends,models,
replay checks,predictions and pooled/period scores saved upon successful run.

Status: syntax and synthetic non-fitting feature prefix/future-isolation,
finite-input,direct RV formula,guard/blend checks passed. No actual input audit,
model fitting,training or inference executed by assistant. No performance result.
