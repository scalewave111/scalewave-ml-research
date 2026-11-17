120H SMALL REGULARIZED NEURAL CORRECTIONS

cd "C:\Users\Siddh\Desktop\ML\November\Week 3"
python Code/fit_net_magnitude_120h_small_correction.py

Independent MLP/GRU,three configurations each:
control: exact previous small mean-residual fitting settings,MLP32hidden16epochs,
GRU16hidden12epochs; original optimizer/decay/unbounded residual.
tiny8:8hidden,12epochs,AdamWlr.0005decay.5,gradientclip.5.
strong16:16hidden,12epochs,AdamWlr.0005decay1,gradientclip.5.
New networks output mean + .25*mean*tanh(net),zero-initialized readout.
Maximum raw correction25%trainingmean;further validation shrinkage retained.
Original-unit L1 objective up to fixed training scale; actual target not clipped.
Architecture,bound and regularization changed jointly;control retained.

Same16fixed features as previous controlled run,2000estimatorcap,
train-onlypreprocessing,120h maturity,GRU36/24/12/0h fulltimeline snapshots.
Four earlier3monthwindows15/12/9/6months. Weights
.025/.05/.1/.2/.25/.35/.5/.75/1 toward mature trainingmean.
Select MAE; .1%pooledmeanbaselinegain,2/4wins/ties,no>2%windowloss;
else meanfallback. RMSEdiagnostic only. Candidate separately labeled.
All three raw outer configs,selected,candidate,mean/median saved for comparison;
training/validation and training-P90tail diagnostics retained.
96innerfits+24outer across both families;runtime unmeasured,noresume.
Optional --families mlp or gru. --check-inputs audits without fitting.
Local manifest receipt and read-only inputs retained.

New Code/experiments/net_magnitude_120h_<family>_small_correction_v1/<timestamp>/
and matching Report folders;prior runs unchanged.
Syntax/import,synthetic zero-output forward,bound arithmetic and MAEguard
checked. No fitting or realdata inference by assistant; no baseline win yet.
Historical exposed adaptive evaluation,not independent confirmation.
