120H NET: LSTM AND DIFFERENT TREE FAMILIES

cd "C:\Users\Siddh\Desktop\ML\November\Week 3"
python Code/fit_net_magnitude_120h_alternative_models.py

Models: RandomForest,ExtraTrees,LSTM. Each standalone plus equal tree pair,
50/50RF+LSTM,50/50ExtraTrees+LSTM,and equal three-way. Components fitted once
per feature-pack/window; blends reuse predictions. L1/L2 heads each; headweights
0/.5/1,mean/median shrinkage .05/.1/.25/.5/.75/1.
RandomForest andExtraTrees32trees,depth6,leaf32,maxfeatures.75,bootstrap1024,
seed48,one estimator thread. RF uses best split;ExtraTrees randomized splits.
LSTM16hidden->Linear16to1,12fast epochs,AdamWlr.001decay.03,batch256,clip1.
Sequences36/24/12/0h on fulltimeline.2000estimator-row cap.

Same paired feature packs control_trend22 andendpoint_memory40; exact feature
ablation and formulas retained. Same4earlier3month windows15/12/9/6months,
120hmaturity,train-onlypreprocessing,dualMAE/meanRMSEguard,meanfallback and
separate diagnostic candidate. All later outcomes excluded from selection.
192innercomponent-head fits,up to48outer; treeabsolute-error fitting can be slow.
No runtime guarantee/resume. Additional transforms multiply fits.

Outputs Code/experiments/net_magnitude_120h_alternative_models_v1/<timestamp>/
and matching Report roots. Existing results unchanged. --check-inputs audits
without fitting; local canonical receipt default; --canonical-manifest override.

Status: syntax,synthetic LSTMforward shape/finite output,and treeparameter
construction checked; no model fitting,actual input audit or realdata inference.
Inherited feature/guard checks from hybrid runner. No baseline win established;
historical exposed adaptive evaluation,not independent confirmation.
