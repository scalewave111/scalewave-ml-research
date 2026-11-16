120-HOUR SMALL MLP EXPERIMENT

Run from November/Week 2:
python .\Code\fit_mlp_compact_120h.py
Audit inputs without fitting:
python .\Code\fit_mlp_compact_120h.py --check-inputs
Synthetic tests:
python .\Code\check_mlp_compact_120h.py

Same120h canonical UP/DOWN excursion labels in logbps, including valid zeros.
Same47,808eligibleinputs/18,193evaluationorigins. Jan/Apr/Jul/Oct2024 fits;
October retained forJan2025. Strict120h label maturity at all train/validation
boundaries; inherited48h initial gap plus additional maturity exclusions.

FOUR PRESPECIFIED SMALL FEATURE GROUPS (four inputs each)
Volatility:120hvolatility,720hvolatility,120/240hvolratio,120hupvariance share.
Trend:120hnormalized momentum,168htrend efficiency,120/240hdrift change,720hnet.
Range:120hrange position,336hdrawdown/volatility,240hrange,1440hrange.
Mixed:720hvolatility,120hnormalized momentum,240hrange position,1440hupvariance.
These are tested separately; no automatic inclusion of old F4Z features.
This is fixed group comparison, not exhaustive selection of every1-4column subset.
All groups use existing causal LONG formulas; full dictionary saved per run.
Each input must have80percentcoverage,1000readings,variation in every innertrain.

MLP
Architecture1: fourinputs plus fourmissingflags ->8ReLU ->1Softplus;40epochs.
Architecture2: fourinputs plus fourmissingflags ->16ReLU ->8ReLU ->1Softplus;60epochs.
Train-only median imputation/standardization; standardizedinputs clipped[-8,8].
AdamW lr.001,weightdecay.1,batch256,gradientnormclip1,CPUdefault2threads.
Deterministic seeds48/148 are averaged. Final weights start atzero with output
initialized at the training normalized-target median. No early stopping.

RAW AND VOLATILITY-NORMALIZED OUTPUTS
Raw: train-median affine scale for numerical conditioning only.
Normalized: scale at each timestamp is120hvol_per_sqrt_hour*sqrt(120).
Scale uses only completed historical hours. TrainingP10 floor(minimum1logbp)
and trainingmedian replacement for missing scale are frozen for prediction.
Network output is converted back to logbps BEFORE absolute-error loss.
Thus both modes optimize original-unit MAE, not unweighted normalized MAE.
The scale is an additional causal dependency even for groups without120hvol.
No future realized volatility, target clipping or log-of-target transformation.

SELECTION
Per UP/DOWN/refit:4groups x2architectures x2modes=16candidates.
Three earlier quarterly validationwindows. Eachcandidate is also blended with
the expanding trainingmedian using modelweights.25/.5/.75/1.
Eligibility:>=.1percentpooledMAEimprovement versusmedian,>=2/3windowwins,
and no validationwindow morethan2percentworse thanmedian.
Choose lowest pooledMAE among eligible; otherwise median (zero modelweight).
The chosen model can still lose to the baseline on later data.
No candidate is selected using outer evaluation or old comparison predictions.
At most768innerseedfits and16finalseedfits. No reliable runtime estimate.
No automatic resume; rerun creates a new folder. Originals remain untouched.

COMPARISONS AND OUTPUT
Code/experiments/mlp_compact_120h_v1/<timestamp>/
metrics.csv/SUMMARY.txt compare mlp_selected,previous,long_reference,median,mean.
previous=pricefollowup20261113T120541578982Z (better historicalUP model).
long_reference=long_features20261113T123515172594Z (better historicalDOWN model).
Both reference CSVs have hash/one-to-one timestamp/issuance/truth checks;
old models are not replayed or retrained here.
Saves selectedfeature/config/mode/blend cards, allvalidationtrials,packadmission,
models with preprocessing/scalefloors/seedweights, savedinference replay,
fitboundary ledger, predictions, provenance/source snapshots,versions andCOMPLETE.
Predictions report all18193origins pertarget; metrics byquarter/Jan2025 andpooled.
The model backend uses the same float32 torch linear operations during replay.

LIMITATIONS
The historical evaluation is already exposed and this design responds to those
results. Earlier validation is reused adaptively; no fresh confirmation claim.
120h labels overlap, so rows are not independent examples. Improvements remain
unproven until user runs the experiment; even historical gains need fresh data.
Prepared using synthetic fitting/tests and real-input checks only; actual-market
training is user-run. No report or other-horizon experiment is started.
