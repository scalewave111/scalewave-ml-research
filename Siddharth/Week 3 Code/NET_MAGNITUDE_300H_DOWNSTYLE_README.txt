300H NET MAGNITUDE: EARLIER DOWN300H MAE TECHNIQUE

cd "C:\Users\Siddh\Desktop\ML\November\Week 3"
python Code/fit_net_magnitude_300h_downstyle.py

Input audit without fitting: add --check-inputs.
Canonical net target abs(10000*ln(P[t+300h]/P[t])); horizon_minutes18000.
Checksum verification/local receipt; --canonical-manifest/--canonical-root/
--week2-code/--threads supported. Same dependencies plus PyTorch.

Reuses original read-only Week2 mae_both_300h model/search backend:
3packs leverage38,flow/liquidation30,combined62. HGB L1 depth2,4leaves,
minleaf256,60rounds,lr.05,L2=10,bins63,3000estimatorcap. GRU16 L1,
12epochs,AdamWlr.001/decay.03,batch512,clipgrad1,2048cap. Neural residual
bounded .75*tanh(head) about training median; original preprocessing retained.
90/60/30/0h sequences assembled BEFORE future-label eligibility filtering.
Four earlier3month windows offset12/9/6/3months;strict300h maturity.
Weights .05/.1/.25/.5/.75/1; anchors median,midpoint(mean/median),mean.
Joint quarterly HGB/GRU16 selection,not four independent families.
MAEguard .1%gain against BOTH baselines,3/4windowwins/ties against stronger
windowbaseline,no>2%windowloss. Medianfallback; bestcandidate diagnostic only.
RMSE,medianabsoluteerror,bias reported,not optimized.96inner+8outerpolicyfits;
no resume/runtime guarantee. No old DOWN-winning choices forced on net target.

Outputs: Code/experiments/net_magnitude_300h_downstyle_v1/<timestamp>/
and Report/net_magnitude_300h_downstyle_v1/<timestamp>/.
Includes source/input/guidance hashes,versions,feature dictionary,population,
validation/quality/fitboundaries,selections,savedmodels/replay,predictions,
periodmetrics and COMPLETE. No prior net300h run required. Oldresults preserved.

Status: syntax and non-fitting synthetic checks passed for62used causalfeature
prefix/future isolation and dualbaselineMAEguard. An initial broad diagnostic
included unused rolling-skew columns with extreme-future numerical sensitivity;
production checks now cover only actual pack columns. No used skew input.
No actualinput audit,training or inference executed; end-to-end adaptation
unexecuted. Exposed historical evaluation; no guaranteed gain.

2026-11-16 FIX: required11market sources now projected from original masked X files,not reduced legacy inventory. Actual --check-inputs passed46,488eligible rows/16,873evaluation origins and all causal/window checks. No model fitting. Rerun original command.
