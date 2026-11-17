120H NET MAGNITUDE: FOUR INDEPENDENT BLENDED MODEL SEARCHES

cd "C:\Users\Siddh\Desktop\ML\November\Week 3"
python Code/fit_net_magnitude_120h_blended.py

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
Three earlier2month windows offset9/6/3months; strict120h maturity.
Rank pooled MAE; require>=.1%gain versus BOTH baselines,>=2/3windowwins/ties,
no>2%windowloss against EACH. Otherwise selected falls back to training mean.
Candidate is validation-best blend regardless of guard, diagnostic only.
RMSE reported, not optimized. No guarantee of later improvement.
Selections frozen before outer scoring. Previous pure forecasts are hash/key/
truth/baseline aligned controls only, never used to choose settings.

Outputs separately per family:
Code/experiments/net_magnitude_120h_<family>_blended_v1/<timestamp>/
Report/net_magnitude_120h_<family>_blended_v1/<timestamp>/
Contracts/hashes,versions,validation_trials.json,selections.json,exact features,
models/replay,predictions,period/pooled metrics,previous forecasts,COMPLETE.
Up to360innerfits (60HGB+60ExtraTrees+180MLP+60GRU),32outerfits.
Same loader/data dependencies as prior net120h runner; PyTorch required.
--week2-code/--canonical-root/--threads/--row-cap supported. No resume.

Generated/syntax checked; non-fitting blend/selection/fallback and dualbaseline
checks passed. No input audit,model fitting,training or inference executed.
Historical exposed adaptive evaluation; no untouched-test or improvement claim.

MANIFEST ACCESS
Defaults to Datasets/canonical_receipts/OUTPUT_FILES.json, the local receipt used successfully for the pure120h runs.
Override with --canonical-manifest PATH if needed. Every canonical label checksum remains mandatory.
Target is abs(10000*ln(P[t+120h]/P[t])); canonical horizon_minutes=7200.
Earlier UP72h method adapted to net120h, not an exact reproduction or guaranteed gain.
Previous pure120h results are preserved and used only for comparisons.

FAST DEFAULT (2026-11-16)
Same command now defaults to fast mode. All four families and three validation
windows retained; packs market_new/all only; estimator row cap2000.
HGB40iterations; ExtraTrees24trees/depth4/bootstrap1024; regularized MLP32hidden,
16epochs; GRU12epochs. Same blend weights and both anchors; same dual-baseline
acceptance and mean fallback. 96 inner fits instead of360; up to32 final fits.
Output roots use _blended_fast_v1 to distinguish altered experiments.
Use --full-search for original budgets; --fast is also accepted explicitly.
Original full script preserved as fit_net_magnitude_120h_blended_full_original.py.
No measured runtime or speedup; narrower search may change results.
