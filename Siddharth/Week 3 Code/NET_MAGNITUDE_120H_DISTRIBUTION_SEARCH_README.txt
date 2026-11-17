120H NET: ORIGINAL FOUR MODELS, NEW DISTRIBUTION FEATURES

cd "C:\Users\Siddh\Desktop\ML\November\Week 3"
python Code/fit_net_magnitude_120h_distribution_search.py

Independent HGB,ExtraTrees,MLP,GRU only. No cross-family mixtures,LSTM or RF.
Within each family compare36feature control vs48feature augmentation.
12new inputs: historical120h endpoint-magnitude Q50/Q75/Q90 over past120/720h;
hourly-to5min squared-return-energy ratios60/120/240/720h; volatility regime
ratios60/720 and240/720. All inputs use completed past returns,not future labels.
Control includes trend/range,volatility,zero/shrunkdrift Gaussian proxies and
completed pastendpoint mean/RMS/CV. Formulas and exact membership saved.
feature_group_ablation.csv measures added12feature group validation MAE/RMSE
changes separately for each family/issue. Contributions conditional,not causal.

Each family uses paired L1/L2heads,headweights0/.25/.5/.75/1,baselineweights
.05/.1/.2/.25/.35/.5/.65/.75/.9/1 toward training mean/median.
Rank by earlier pooled MAE; acceptance still requires .1%MAEgain vs bothbaselines
AND .1%RMSEgain vsmean,3/4windowwins/ties,no>2%windowloss for each criterion.
Mean fallback explicit,candidate diagnostic separate; no guaranteed later win.
Four earlier3monthwindows offsets15/12/9/6months,strict120hmaturity.
Train-only preprocessing;fulltimeline GRU36/24/12/0h snapshots.
Fastdefault cap2000,HGB40rounds,Extra24treesdepth4bootstrap1024,
MLP32hidden16epochs,GRU16hidden12epochs.256innerheadfits across4families,
up to64outerheadfits. --families mlp gru restricts run if desired.
No resume/runtime guarantee; additional transforms increase fits.

Outputs Code/experiments/net_magnitude_120h_<family>_distribution_search_fast_v1/
and matching Report timestamped roots. Previous runs preserved.
--check-inputs audits without fitting. --canonical-manifest overrides localreceipt.
Feature/formula inventories,ablation,allvalidationtrials,selection,models/replay,
sourcehashes,predictions and pooled/periodmetrics saved by successful run.

Synthetic syntax,48feature prefix/future isolation,finitecoverage and direct
completedendpointquantile checks passed. No training or inference by assistant.
Actual input-audit outcome recorded separately in PROJECT_CONTEXT.md.
Historical adaptive exposed evaluation,no independent confirmation.
