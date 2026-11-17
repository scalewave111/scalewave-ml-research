120H BOUNDED NEURAL REFINEMENT

cd "C:\Users\Siddh\Desktop\ML\November\Week 3"
python Code/fit_net_magnitude_120h_bounded_refined.py

Independent MLP/GRU;fixed16features and2000cap. Retains tiny8 andstrong16
controls from previous smallcorrection run. Alternatives strong16bound10%,
bound40%,or24epochs. Othercontrolsettings retained:zero readout,AdamWlr.0005,
weightdecay.5tiny8/1strong16,gradientclip.5,12epochs exceptlong16.
Bounds based only on maturetrainingmean; no targetclipping/future scale.

Samefour earlier3monthwindows15/12/9/6months,strict120hmaturity,
training-only preprocessing,GRU36/24/12/0h. Nine original shrinkageweights.
Newstabilityselector ranks mean(windowMAE/meanbaselineMAE)+.25std(ratios).
Acceptance requires .1%pooledMAEgain,3/4wins/ties,no>1%windowMAEloss;
otherwise meanfallback. RMSEreported only. Candidate separately labeled.
Thischanges selector from previous pooledMAErank/2of4/no>2%loss and narrows
modelgrid;previous fixedconfiggains motivate exploratory search only.

160innerfits+40outer/all2families. All5fixedconfigforecasts saved,plus previous
tiny8/strong16 controls,selected/candidate/baselines andtrain/validation/tail
 diagnostics. Previousreferencekeys/truth/baselines/checksums verified,not used
for withinissue selection. No guarantee of later gain or runtime,resume absent.
New net_magnitude_120h_<family>_bounded_refined_v1 timestamped output roots.
Optional --families mlp or gru;--check-inputs nofit audit.

Syntax/import,synthetic zerooutputforward,guard,stabilityselection/fallback
checks passed. No model fitting,actualinputaudit or realdata inference executed.
Historical adaptive exposed evaluation,not independent confirmation.
