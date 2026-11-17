72H NET MAGNITUDE: FOCUSED MLP REFINEMENT

cd "C:\Users\Siddh\Desktop\ML\November\Week 3"
python Code/fit_net_magnitude_72h_mlp_refined.py

User-run only. Input audit: add --check-inputs.
Focuses on the winning guarded MLP policy (MAE226.793372) without rerunning the
other families. Preserves that saved result; new outcomes may be worse.

3MLP configs: control32hidden/16epochs/decay.1; longer32hidden/32epochs/decay.1;
lighter_decay32hidden/16epochs/decay.03. ReLU/L1/AdamWlr.001,seed48,batch256,
clipgrad1,deterministic shuffled minibatches. Same two packs market_new/all,
2000estimator-rowcap,training-only IQR/missingflags/clip8,full mature targetmedian
scaling. One model family; no model-family mixing.

Blends weights .1/.25/.4/.5/.6/.75/.9/1 with training median,mean,or midpoint
(mean+median)/2. Predictions reused across blends;72innerfits,up to8outerfits.
Same3earlier2month windows at9/6/3months,strict72h label maturity.
Guard remains >=.1%pooled MAEgain,>=2/3windowwins/ties,no>2%windowloss versus
EACH mean/median. Fallback mean; diagnostic validation-best candidate separate.
RMSE reported,not optimized. Choices frozen before outer errors are scored.

The prior pureMLP and winning fastMLP predictions are comparison columns
previous and previous_fast. Hash/key/truth/baseline checks mandatory; neither
reference selects new settings. Current historical evaluation is exposed and
motivated this refinement; no independent confirmation or guaranteed gain.

New output roots (existing results untouched):
Code/experiments/net_magnitude_72h_mlp_refined_v1/<timestamp>/
Report/net_magnitude_72h_mlp_refined_v1/<timestamp>/
Includes config/input hashes,versions,validation ledger,exact fitted features,
selection records before outerfits,model replay,predictions,period metrics,
comparisons and COMPLETE. Same dependencies/local inputs as72h blended runner.
--row-cap/--threads/--week2-code/--canonical-root supported; no resume.

STATUS: syntax and non-fitting control/config/blend/selection checks passed.
No input audit,synthetic model fits,training or inference run by assistant.
