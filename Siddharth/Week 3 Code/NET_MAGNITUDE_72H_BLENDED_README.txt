72H NET MAGNITUDE: FOUR INDEPENDENT BLENDED MODEL SEARCHES

cd "C:\Users\Siddh\Desktop\ML\November\Week 3"
python Code/fit_net_magnitude_72h_blended.py

One family: add --families mlp (or hgb/extra/gru).
Input audit without fitting: add --check-inputs.

Five existing feature packs per family/quarter. HGB/GRU retain existing MAE
settings. ExtraTrees uses64absolute-error trees,depth6,minleaf32,maxfeatures.75,
bootstrap up to4096rows/tree. MLP uses earlier UP72h PyTorch ReLU/L1 settings:
small32/16,40epochs,decay.01;medium64/32,60epochs,decay.01;
regularized32,60epochs,decay.1. AdamWlr.001,batch256,clipgrad1,seed48.
MLP/ExtraTrees target divided by max(full-trainingmedian,1); HGB/GRU retain
existing target scaling. Existing net training-only median/IQR preprocessing,
missing flags,clipping +/-8 and6000estimator-rowcap retained. Not exact UP fit
reproduction. Preprocessing/baselines use all mature training rows.

Blend=weight*model+(1-weight)*anchor. Weights .25/.5/.75/1;
anchors training median/mean. No family mixing. Fits reused across weights.
Three earlier2month windows offset9/6/3months; strict72h maturity.
Rank pooled MAE; require>=.1%gain versus BOTH baselines,>=2/3windowwins/ties,
no>2%windowloss against EACH. Otherwise selected falls back to training mean.
Candidate is validation-best blend regardless of guard, diagnostic only.
RMSE reported, not optimized. No guarantee of later improvement.
Selections frozen before outer scoring. Previous pure forecasts are hash/key/
truth/baseline aligned controls only, never used to choose settings.

Outputs separately per family:
Code/experiments/net_magnitude_72h_<family>_blended_v1/<timestamp>/
Report/net_magnitude_72h_<family>_blended_v1/<timestamp>/
Contracts/hashes,versions,validation_trials.json,selections.json,exact features,
models/replay,predictions,period/pooled metrics,previous forecasts,COMPLETE.
Up to360innerfits (60HGB+60ExtraTrees+180MLP+60GRU),32outerfits.
Same loader/data dependencies as prior net72h runner; PyTorch required.
--week2-code/--canonical-root/--threads/--row-cap supported. No resume.

Generated/syntax checked; non-fitting blend/selection/fallback and dualbaseline
checks passed. No input audit,model fitting,training or inference executed.
Historical exposed adaptive evaluation; no untouched-test or improvement claim.

FAST MODE (2026-11-16)
Stop the existing slow process with Ctrl+C in its terminal, then run:
cd "C:\Users\Siddh\Desktop\ML\November\Week 3"
python Code/fit_net_magnitude_72h_blended.py --fast

Four families retained; packs market_new and all only; one regularized32 MLP
with16epochs; HGB40rounds; ExtraTrees24absolute-error trees/depth4/1024bootstrap;
GRU16hidden/12epochs; estimator cap2000. Up to96inner+32outerfits.
Full mature training data still determine preprocessing/scales/baselines.
Same72h purge,3validation windows,blend weights,anchors and dualbaseline guard.
Each window prints progress. A smaller --row-cap is respected.
Outputs use net_magnitude_72h_<family>_blended_fast_v1 timestamped roots.
Full mode remains available without --fast. Existing completed/partial runs
are preserved; no resume or automatic termination of a running process.
Reduced budget changes the experiment and may affect accuracy. No measured
runtime or performance claim. Fast settings syntax/configuration checked only;
no input audit or training executed by assistant.
