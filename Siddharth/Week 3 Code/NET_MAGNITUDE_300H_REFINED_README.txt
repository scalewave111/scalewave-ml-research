300H NET MAGNITUDE: FOCUSED MAE REFINEMENT

cd "C:\Users\Siddh\Desktop\ML\November\Week 3"
python Code/fit_net_magnitude_300h_refined.py

Input audit: add --check-inputs. Non-fitting checks:
python Code/check_net_magnitude_300h_refined.py

Focuses on leverage38,the pack used in both accepted learned periods of the
previous run. Retains HGB depth2/4leaves/60rounds/minleaf256 and GRU16 L1
12epochs/lr.001/decay.03 controls. Adds HGBdepth1/2leaves and GRU16decay.1.
Other settings,caps3000HGB/2048GRU,training preprocessing and90/60/30/0h
snapshots unchanged. No previously observed quarterly choice is forced.

Modelweights .05/.1/.25/.4/.5/.6/.75/.9/1;
anchor=(1-q)*trainingmedian+q*trainingmean,q0/.25/.5/.75/1.
Blends reuse predictions.64innerfits versus96previously;8outerpolicyfits.
Same four earlier3monthwindows offset12/9/6/3months;strict300h maturity.
Same MAEguard against BOTH mean/median,>=.1%pooledgain,3/4windowwins/ties,
no>2%windowloss;medianfallback. Candidate diagnostic separate.
RMSE,medianabsoluteerror,bias reported,not optimized. JointHGB/GRUquarterly
selector; not four independent model families. No guarantee of better MAE.

Previous selected445.150944 and diagnostic437.514488 predictions are hashed,
key/truth/baseline aligned comparison columns only; never selector inputs.
Existing run20261116T111254635059Z preserved. New results may be worse.
New outputroots Code/experiments/net_magnitude_300h_refined_v1/<timestamp>/
and Report/net_magnitude_300h_refined_v1/<timestamp>/.
Contracts/config/source/inputhashes,versions,features,population,validation,
fitboundaries,selection,savedmodels/replay,predictions,comparisons andmetrics.
Local net300h loader plus read-only Week2 backend required. --canonical-root,
--canonical-manifest,--week2-code,--threads supported. No resume.

STATUS: actual --check-inputs passed46488eligible rows/16873evaluationorigins,
leverage38/190includingsequencecolumns;causality and4validationhistories passed.
Non-fitting configdispatch/restoration,retainedcontrols,dualbaselineguard and
sequence timing checks passed. No model fitting/training/inference executed.
Historical exposed adaptive research,not independent confirmation.
