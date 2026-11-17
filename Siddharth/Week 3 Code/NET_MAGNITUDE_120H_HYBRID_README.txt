120H NET: TREE-NEURAL HYBRIDS AND ENDPOINT-MEMORY ABLATION

cd "C:\Users\Siddh\Desktop\ML\November\Week 3"
python Code/fit_net_magnitude_120h_hybrid.py

Changed approach: jointly validate HGB,MLP,GRU standalone controls plus 50/50
HGB+MLP,HGB+GRU,MLP+GRU and equal three-way hybrids. Fixed convex mixture grid;
not learned stacking on evaluation labels. No repeated fits per blend weight.
Each component has L1/L2 heads; within-family objective mixture0/.5/1.
Then weight .05/.1/.25/.5/.75/1 toward mature training mean or median.

Paired feature packs with identical data/model budgets:
control_trend22: previous trend/range and endpoint-volatility controls.
endpoint_memory40: adds18 features to the control pack. Past60/120/240/720h
zero-drift Gaussian absolute-move proxies,25%-shrunk drift proxies and drift/noise;
past168/720h mean,RMS,CV of completed historical120h absolute endpoint returns.
Overlapping past blocks are historical features,not independent observations.
Features use only return endpoints strictly before the decision timestamp.
Existing four prior5minRV controls and full-drift proxies retained for ablation.
Exact formulas and membership saved during run; upstream F4Z semantics inherited.

Validation: four earlier3month windows offset15/12/9/6months,strict120h maturity.
All imputation/scaling,coverage filtering and target normalization training-only.
GRU uses full feature timeline snapshots36/24/12/0h,not label-eligible rows only.
Rank worst pooled MAE/median,MAE/mean,RMSE/mean ratio. Require .1%gain each,
3/4wins/ties,no>2%windowloss each; else meanfallback. Diagnostic candidate separate.
feature_group_ablation.csv records best per-pack validation scores for every mix;
positive relative_score_reduction indicates added-group improvement on validation.
This is conditional group evidence,not individual causal feature importance.
Quarterly selection freezes before outer scoring. Historical evaluation exposed;
no promised baseline gain and no untouched confirmation.

Fast default:2000estimator rows,HGB40rounds,MLP32hidden16epochs,GRU16hidden12epochs.
192inner component-head fits,up to48outer component-head fits,one transform.
Full search increases neural budgets/config behavior; extra transforms multiply cost.
No measured runtime or resume. No training is executed by creation of this file.
Run --check-inputs for a no-fit actual input/feature audit first if desired.
Local canonical manifest receipt is default; --canonical-manifest overrides.

Outputs Code/experiments/net_magnitude_120h_hybrid_endpoint_v1/<timestamp>/
and Report/net_magnitude_120h_hybrid_endpoint_v1/<timestamp>/.
Saves all validation trials,feature ablations,chosen model/head/baseline weights,
exact features/formulas,source hashes,models,replay,predictions and metrics.
Previous runs are preserved; no previous trained model is required.

Status: syntax plus synthetic non-fitting exact40feature prefix/future isolation,
completed-endpoint directformula,finite coverage,convexmix and dualmetricguard
checks passed. No actual input audit,fitting,training,inference or replay executed.
