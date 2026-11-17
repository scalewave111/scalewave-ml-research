48H DIRECTION FOCUSED PROBABILITY REFINEMENT

cd "C:\Users\Siddh\Desktop\ML\November\Week 3"
python Code/fit_direction_48h_calibrated_refined.py

Six paired tree specs: HGBnewstatecontrol/regularized,ExtraTreescorecontrol/
regularized,HGBcoreregularized,ExtraTreesnewstateregularized. Originalcontrol
settings retained;regularizedHGB60rounds/depth2/4leaves/leaf128/L2=20/rate.03;
regularizedExtra48trees/depth4/leaf64. Same directioninputs,target,48hmaturity,
2000cap andfour earlier3monthwindows15/12/9/6months. No magnitudeinputs.
Newstage focused after observed HGBnewstate/ExtraTreescore candidate scores;
this is exploratory,not a new untouchedtest.

Each fittedmodel tests raw,offset andregularizedPlatt probabilities.
Validationblockjcalibration uses ONLY preceding purgedOOFblocks withlabelend
strictlybeforejcut. Firstblockidentity. No in-sampleclassifierpredictions or
outeroutcomes used to fit calibration. Finalcalibrator uses allfour oldOOFblocks
withlabels maturebeforeouterissue;not a recentonlinecalibration.
Offset:half(logit(actualrate)-logit(meanpredictedrate)),capped+/-.5;
apply50/50raw/calibratedblend. Platt:LogisticRegressionC.01 on rawlogits,
then50/50raw/calibratedblend. Min200labels/bothclasses,elseidentity.
Weights.25/.5/.75/1 toward maturetrainingprevalence also chosen earlier.

Sameoriginalselectionguard:rank mean(relativepooledlogloss,relativeBrier),
.1%gainBOTH,3/4windowwins/ties,no>2%loss each;otherwiseprevalencefallback.
Diagnosticcandidate separate. Priorselected0.691272 andcandidate0.686161
are hash/alignmentchecked comparison controls only,never used to choose weights.
Do not removeguard solely because previous diagnostic outer score was better.

96innerclassifierfits/up to8outerclassifierfits plussmallcalibrationfits.
Samefastrowcap/2threads,no runtimeguarantee/resume. --check-inputs nofit audit.
New Code/experiments/direction_48h_calibrated_refined_v1/<timestamp>/
and matching Report roots;oldresults preserved. Saves validationtrials,
calibrationaudits/parameters,exactfeatures,models/replay,probabilities,predicted/
actualdirections,logloss/Brier/AUC/accuracy andcomparisons.

Status:syntax/import,nonfitting logit/offset/identityPlatt arithmetic,
probabilitybounds,calibrationparameterJSONreplay,minimumsupport andselection/
fallback checked. No classifierorPlatt fitting,actualinputaudit orrealdata
inference. Originalrunner actualinputs audited earlier;new fullfit/calibration
pipeline remains unexecuted. No improvedscore established.
