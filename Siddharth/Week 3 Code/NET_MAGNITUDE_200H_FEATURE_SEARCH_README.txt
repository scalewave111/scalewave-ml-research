200H NET MAGNITUDE: NEW ENDPOINT-SPECIFIC FEATURE SEARCH, ALL FOUR FAMILIES

cd "C:\Users\Siddh\Desktop\ML\November\Week 3"
python Code/fit_net_magnitude_200h_feature_search.py

HGB, ExtraTrees, MLP and GRU run independently and sequentially. Fast budgets
are default. One family: --families hgb (or extra/mlp/gru).
Optional --full-search; optional --transforms raw signedlog quadratic.
Input audit only: --check-inputs. No automatic resume.
Synthetic non-fitting checks: python Code/check_net_magnitude_200h_feature_search.py

WHY NEW FEATURES
Previous UP-style candidate forecasts lost to the mean baseline. UP excursion
and absolute endpoint magnitude have different targets. This search tests
features designed for absolute 200-hour endpoint returns, keeping canonical
net_magnitude_log_bps and strict 200h maturity. A baseline win is an objective,
not a guaranteed property of the code. Exposed historical results remain
exposed; no final-period feature ranking or evaluation bias correction.

THREE PRESPECIFIED PACKS: 68 DISTINCT INPUTS
volatility_endpoint30: hourly volatility at24/100/168/200/336/400/720/1440h,
plus all22new endpoint features.
trend_range36: momentum,range position,efficiency,up-variance share at100/200/
400h;168h drawdown and336h jump share; plus22new endpoint features.
slow_market54: volatility_endpoint30 plus24past market-state means/deviations
at24/200h for funding,basis,OIposition,OIunwind,flowpressure and historical
liquidation activity. Exact feature names and formulas saved during runs.

NEW ENDPOINT FEATURES (net_200h_endpoint_features.py)
For each h=100/200/400/720h, use exactly12*h finite 5min returns in[t-h,t):
N=sum(r),Q=sum(r^2),mu=.25*N*200/h,
sigma=sqrt(max(Q-N^2/(12*h),0)*200/h), floored1e-8.
Four measurements per h:
- RV5m=sqrt(Q)
- zero-drift magnitude proxy=sigma*sqrt(2/pi)
- shrunk-drift magnitude proxy=Eabs(Normal(mu,sigma^2))
- drift-to-noise=mu/sigma
The25% drift shrink is fixed before validation, not fitted on later outcomes.
These16features are historical hypotheses, not forecasts treated as truth.
Six more inputs: mean,RMS,coefficient of variation of past200h absolute
endpoint returns, aggregated over168/720completed hourly endpoints. These
blocks overlap; they are known historical measurements, not future labels.
Hourly aggregation uses only 5min return endpoints strictly before its hour
timestamp. Half-hour origins use latest completed hour with59m59s tolerance.
Missing returns invalidate affected windows; no missing outcome imputation.

MODELS / SELECTION
Retains independently fitted paired L1/L2 heads in each of all4families.
Head weights0/.5/1; model weights .05/.1/.25/.5/.75/1 versus mean/median.
Default input signedlog; optional raw/quadratic transform search.
Full training rows supply coverage filters,median/IQR,missing flags,target
centering/scaling and baselines. At least80% finite/1000finite/variation.
GRU uses full X timeline snapshots60/40/20/0h; no future-label eligibility
mask in its sequence. Inference with<80%finite required inputs at any
snapshot returns training median for the paired heads.
Four earlier3month validation windows offset15/12/9/6months; strict200h
training and validation label maturity. Choices frozen before outer scoring.
Rank worst pooled relative MAE/median,MAE/mean,RMSE/mean. Require .1%gain
for each comparison,3/4windowwins/ties,no>2%windowloss. Otherwise selected
is mean. Unrestricted validation-best candidate saved separately.
No family mixing. No guard relaxation to manufacture a win.
Fast cap2000,HGB40rounds,ExtraTrees24depth4bootstrap1024,MLP32/16epochs,
GRU16/12epochs. See NET_MAGNITUDE_200H_BLENDED_README.txt for full settings.
384inner head fits total across4families,up to64outer head fits with one
transform; full budgets576inner. Extra transforms multiply search cost.

OUTPUTS (prior runs preserved)
Code/experiments/net_magnitude_200h_<family>_feature_search_fast_v1/<timestamp>/
Report/net_magnitude_200h_<family>_feature_search_fast_v1/<timestamp>/
Full budgets use feature_search_v1. Includes feature formulas/packs,
input/source hashes,validation ledger,choices,exact fitted features,paired
model bundles/replay,predictions,MAE/RMSE/bias,COMPLETE after success.
Same canonical-root/week2-code/canonical-manifest overrides as prior runner.
Default local receipt: Datasets/canonical_receipts/OUTPUT_FILES.json.

STATUS
Syntax checked; synthetic exact RV formulas,22feature finiteness,prefix and
future isolation,missing-return invalidation,source projection,transforms,
blending,dual-metric guards and strict200h boundaries passed.
Actual --check-inputs PASSED:47088eligible rows,17473evaluation timestamps;
608missing/invalidlabels and400source-binding crossings excluded. Three
pack memberships and causal prefix/future checks passed on actual inputs;
all16earlier validation histories have sufficient mature training/validation.
No fitting,training or inference performed by assistant. Predictive benefit,
end-to-end training and serialization replay await user run. No baseline
win or untouched-test claim.
