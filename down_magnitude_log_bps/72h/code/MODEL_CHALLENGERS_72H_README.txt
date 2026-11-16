72H EXTRA TREES + NEURAL NETWORK CHALLENGERS

Run both sequentially:
cd "C:\Users\Siddh\Desktop\ML\November\Week 2"
python .\Code\run_model_challengers_72h.py

Or separately:
python .\Code\fit_extra_trees_excursions_72h.py
python .\Code\fit_neural_excursions_72h.py

Audit only: add --check-inputs. Does not fit candidate models, but replays saved
reference models. CPU threads default2; Extra Trees uses one forest worker.
Combined output: Code/experiments/model_challengers_72h_v1/<timestamp>/
with extra_trees and neural subdirectories. Separate scripts default to
excursion_extra_trees_72h_v1 and excursion_neural_72h_v1 respectively.
No automatic resume. If stage2 fails, completed stage1 remains saved; use the
neural script alone rather than rerunning both. No runtime estimate promised.

COMMON COMPARISON
Independent UP/DOWN canonical72h excursions in logbps, allvalidzeros included.
Same18385evaluation origins and original48h gap; every inner/outer training
label must mature strictly before prediction/validation boundary (72h purge).
QuarterlyJan/Apr/Jul/Oct2024 expanding training; October model retainedJan2025.
Six candidate combinations: three settings x two feature packs.
Packs: previously selected features; those plus72/168/336/720h price blocks
(total/up/down RV,netreturn,range,distancefromhigh/low,maxabs5mreturn).
Existing7d features deduplicated. Past returns strictly before origin only.
No new recency-window search: all training histories are expanding.
Training-only imputation/scaling/missingflags. Same previous source/input
hashes, canonical labels and reference-model replay checks as72h refinement.

EXTRA TREES
ExtraTreesRegressor,64trees,absolute_error criterion (median leaf predictions,
averaged across trees),bootstrap4096samples per tree (capped at training count),
seed48. Settings(depth,minleaf,maxfeaturefraction): (4,64,1),(6,32,.75),(8,16,.75).
This is a randomized forest, not the earlier histogram gradient booster.

NEURAL NETWORK
PyTorch CPU MLP, ReLU hidden activations, linear output, MAE/L1Loss.
Settings(hidden widths,epochs,AdamWweightdecay): (32/16,40,.01),
(64/32,60,.01),(32,60,.1). AdamW learningrate.001,batch256,gradientnormclip1,
seed48,deterministic algorithms. No random validation split or early stopping.
Input standardized with training-only statistics then clipped+/-8; target
divided by max(trainingmedian,1), restored to logbps for outputs. Predictions
clipped nonnegative. Last-layer bias initialized to training median in scaled
units. Epoch counts are selected as part of configurations using older windows.
Weights saved as arrays; inference uses PyTorch float32 linear kernels, matching
training. Exact same-batch replay is checked at fit time. This replaces NumPy
matrix multiplication, whose different rounding caused a small replay mismatch.

SELECTION AND CONTROLS
Each candidate tested at weights0/.25/.5/.75/1 blended with expanding training
median. Earlier3quarter windows select minimumMAE among candidates beating
median by>=.1% pooled and winning>=2/3windows; otherwise median fallback.
Blends reuse predictions; maximum144innercandidatefits per family +8outer,
plus24inner/8outer boosting controlfits. Selectors do not consult outerlabels.
Compare previous=pricefollowup,previous_tuned=completed72h tuning,selected,
training_median andtraining_mean. Full period and pooledMAE/RMSE/IC saved.
Training/evaluation level diagnostics, trials, selections beforeouterfit,
savedmodelreplay,model/input/source/guidancehashes,CONTRACT,SUMMARY,COMPLETE.
Onlyselected candidates are fitted/scored on outerdata; unselected models have
validation scores only. No best-outermodel promotion is performed.

Defaults reference pipeline20261113T111815030467Z/price_followup/
20261113T113500606206Z and tuning20261113T120003142401Z.
Requires existingWeek1/helper/data paths,sklearn andPyTorch (installed2.5.1CPU).
Historical evaluation is exposed and reference feature lists used the same
validation windows before: adaptive exploratory research,not fresh confirmation.
Neither different architecture nor validationgate guarantees baseline improvement.

STATUS
Preparation includes synthetic small-model selection,strictpurge,outerisolation,
missingness,zero fallback,causalfeatures andserializedreplay checks; actualinput
andreference-model replay audits only. Fullrealdata training is user-run.
