48-HOUR DIRECTION CLASSIFICATION: LOG LOSS AND BRIER

cd "C:\Users\Siddh\Desktop\ML\November\Week 3"
python Code/fit_direction_48h_classifiers.py

Direction task: canonical BTCUSDT horizon2880 endpoint direction>0 is UP1;
negative/zero are non-UP0. p_up forecasts P(UP); p_down=1-p_up includes zero.
This is classification,not magnitude regression. Canonical48h checksum receipts,
VALID labels and strict48h maturity;source-binding crossings excluded.
Originalmasked X_train/X_development projected directly,never magnitude16inputs.
No magnitude predictions or future excursion labels used as features.

Eight classifiers/objective variants:
logistic: L2 logistic,C.01,300iterations.
hgb: loglossboosting40rounds,depth3,7leaves,minleaf64,L2reg10,rate.05.
random_forest:32trees,Gini,depth5,minleaf32,maxfeatures.75,bootstrap1024.
extra_trees:32trees,loglosssplits,depth5,minleaf32,maxfeatures.75,bootstrap1024.
mlp_logloss/mlp_brier:MLP16ReLU,16epochs,BCE or squaredprobabilityloss.
gru_logloss/gru_brier:GRU8+linear,16epochs,BCE or squaredprobabilityloss.
NeuralAdamWlr.0005decay.1,batch256,gradclip1,seed48,zero finalweights,
trainingprevalence logitbias. GRU36/24/12/0h snapshots from fulltimeline.
Trees use native classifier objectives; all evaluated by BOTH logloss/Brier.
No classbalancing/oversampling;probabilities preserve observed classfrequency.

Three independent directionfeaturepacks:
direction_core5: supplied historical directioncore.
direction_original384: historicalv45 directioninventory only,not fitted weights.
direction_new_state99: core5+94new strictlypast direction-state features.
Seven1hsourcechannels:flowpressure,CVDpathefficiency,fundingchange,basischange,
basispersistence,OIposition,leverageunwind. Past24/48/120hmean,deviation,zscore,
exact-gridchanges,lag30min and3centeredproducts. Mean/std need80%expected
half-hourcoverage;missinggridrows staymissing. Percentiles/interactions are
statehypotheses,not assumed causal signedbuy/sell signals. No source semantics
reconstructed beyond preserved identifiers. Exactnewformulas saved.
Allfeaturequality/imputation/median-IQRscaling/clipping/missingflags trainingonly.
384inventory was historicallyselected;this is not fresh nestedfeaturediscovery.
feature_pack_comparison.csv reports earlier logloss/Brier per pack/model/issue.

QuarterlyJan/Apr/Jul/Oct2024 fits;October retained intoJan2025.
Four earlier3monthwindows15/12/9/6months;trainlabelend<fitcutoff;
validationlabelend<windowend and outerissue. All outerchoices frozen before scoring.
Rank mean(relativepooledlogloss,relativepooledBrier) vs trainingprevalence.
Require .1%gain BOTH,3/4windowwins/ties,no>2%windowloss each;else prevalence.
Probabilityweights.25/.5/.75/1 toward maturetrainingprevalence chosen earlier.
Primaryp_up is globalvalidationselectedpolicy;each classifiercolumn is its own
validationselectedpack/blend and may fallback;candidate diagnostic separate.
Fair.5 and historicalprevalence baselines reported. Accuracy/AUC secondary.
Calibration.csv fixedbins are diagnostics,never used for fitting.

Default2000estimatorrows,2threads.384innerfits with all8variants/3packs,
up to36outerfits;no runtimepromise/resume. To shorten:
python Code/fit_direction_48h_classifiers.py --models logistic hgb extra_trees mlp_logloss
--check-inputs performs nofit audit;--row-cap changes estimatorcap.
Models and files require same dependencies as previous runners.
Localcanonicalreceiptdefault;--canonical-manifest,--canonical-root,--data-dir,
--direction-template override paths.

Outputs Code/experiments/direction_48h_classifiers_v1/<timestamp>/
and Report/direction_48h_classifiers_v1/<timestamp>/.
predictions.csv contains decision_time,y_up,actual_direction,predicted_direction,
p_up,p_down,signed_return_log_bps,baselines,permodelpolicies,candidate.
Saves exactfeatures/formulas,modelhashes/replay,validationledgers,selections,
featurepackcomparisons,calibration and pooled/periodmetrics.
Existing48hmagnitude and historicaldirectionresults preserved.

STATUS: actual --check-inputs passed48000eligible/18385evaluationorigins;
packs5/384/99. Sourcechecksums,labels,originaldirectionfeatures and94newfeature
prefix/futureisolation passed. Synthetic logloss/Brier arithmetic,dualmetricguard,
neuralpriorinitialization passed. No classifierfitting orrealdata inference run
by assistant;end-to-endfitting/serialization remains unverified. Historical exposed
adaptiveevaluation,no independentconfirmation or guaranteedbaselinegain.
