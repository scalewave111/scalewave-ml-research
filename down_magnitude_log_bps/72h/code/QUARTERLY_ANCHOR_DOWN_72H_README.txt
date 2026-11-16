QUARTERLY-MEDIAN ANCHOR: CONTROLLED DOWN72h EXPERIMENT

Run from November/Week 2:
python .\Code\fit_quarterly_anchor_down_72h.py

Read-only real-input/stability audit: add --check-inputs
Synthetic tests: python .\Code\check_quarterly_anchor_down_72h.py

PRIMARY QUESTION
Can small adjustments around the ORIGINAL quarterly median beat that median?
The selected policy uses quarterly anchor ONLY. Earlier validation chooses
weight .10/.25/.50; weight0 (unchanged quarterly median) is fallback.
UP is untouched. No automatic promotion of a hindsight-best variant.

HISTORICAL RESIDUALS WITHOUT FUTURE LEAKAGE
For every historical training observation, calculate the median known at the
start of THAT observation's calendar quarter. Only labels with origin+72h
strictly before that quarter start can enter. Require48maturelabels; early rows
without a historical anchor are omitted from learned fits, not zero-filled.
Train on actual minus that historical anchor, not actual minus today's median.
The quarterly benchmark itself still uses ALL eligible mature labels.
October2024 model/anchor remains frozen throughJanuary2025, matching prior runs.
historical_anchor_ledger.csv records every anchor,count and latest label maturity.

MATCHED ANCHOR COMPARISON
Also fit a residual model against the causal180day rolling median. Its training
residual is actual minus the rolling median available at that historical origin.
The rolling helper uses strict72h maturity,min48,expanding fallback as documented
in GUARDED_DOWN_72H_README.txt. Both branches use IDENTICAL training rows where
both historical anchors are available, identical feature selection/preprocessing
and identical model settings. Only anchor/residual definition changes between
these matched branches. Save all6fixed anchor/weight variants for diagnosis.
Their outer scores do NOT choose the primary policy. Rolling variants are
comparison columns; selected never switches to rolling based on outer results.

FIXED MODEL AND FEATURES
Use fresh120feature universe; same training-only3block sign/coverage screen as
guarded runner:three trailing3month blocks,>=300finitevalues/80%coverage each,
coverage spread<=10points,same Spearman sign with abs>.01 in each block.
Rank by minimum absolute block correlation, then remove redundant inputs with
absolute training Spearman>=.90 and >=1000complete paired observations. Retain
up to6features. Both anchor branches obtain the same columns in each fit.
Training-only median imputation,missing indicators,empirical rank transform.
Fixed MAE histogram boosting:80iterations,depth2,max4leaves,minleaf256,rate.05,
L2=10,seed48,no early stopping. Residuals divided by positive training-target
median(floor1) during fitting and restored to original logbps for predictions.
No family,depth,history,deadzone or dynamic-distribution-gate search.

MANDATORY MISSING-INPUT FALLBACK
If ANY selected feature is missing/nonfinite, force adjustment weight0 at that
origin. If final fit has no eligible features, predict its anchor everywhere.
Otherwise prediction=max(0,anchor+weight*predicted_residual).
Missing flags still enter training preprocessing; test-time fallback is mandatory.

EARLIER VALIDATION
Four previous3month windows,strict72h maturity before each training/validation
boundary. >=.1% pooledMAE improvement,beats/ties median in>=3of4windows,and no
window>2% worse. Primary chooses among quarterly weights only; otherwise median.
Same quarterly expanding outer fits and same18385evaluation origins. No later
outcomes enter fitting,feature selection,preprocessing or weight selection.
Stable-feature filtering and redundancy removal do not prove future usefulness.

PAIRED CALENDAR-BLOCK BOOTSTRAP
After predictions are frozen, resample paired error differences:
abs(actual-quarterlymedian)-abs(actual-prediction). Positive means improvement.
5000draws by default; nonoverlapping7/14/28day calendar blocks, independently
within each model-issue period. Stratum weights stay equal to original row
fractions; resampled block sums/counts preserve the paired comparison. October's
stratum includesJanuary2025, matching its unchanged model. Intervals are2.5/97.5
percentiles. Saved for selected and all6diagnostic fixed variants.
Bootstrap is conditional on saved forecasts; no retraining or reselection.
It does NOT correct repeated adaptive searches or multiple comparisons. Block
lengths approximate dependence; longer blocks leave few blocks per quarter.
The fraction of bootstrap gains>0 is not a probability of true superiority.
If selected equals baseline exactly, its gain and interval are zero.

OUTPUT AND STATUS
Code/experiments/quarterly_anchor_down_72h_v1/<timestamp>/
predictions.csv,metrics.csv,bootstrap.csv/json,selections.json,fit_windows.csv,
historical_anchor_ledger.csv,feature_dictionary.csv,exclusions,source snapshots,
contract,versions,input/guidance hashes,per-cycle trials/audits/model/hash,
SUMMARY.txt andCOMPLETE.json. All6saved variant predictions must replay.
Controls: previous guarded20261114T112121591551Z,transformed20261114T085824921737Z,
neural20261113T123304519186Z,quarterly/rollingmedian,andmean. Previous forecasts
are hash/key/truth/issuance aligned; their old models are not replayed here.
About32innerfits+8outerfits;weights reuse predictions. Default2threads,no resume.
Optional --bootstrap-draws changes repeats(minimum100); default5000 recommended.

Synthetic historical anchor/maturity/October freeze,redundancy,matched fitting,
missing fallback,serialization,selection isolation/zero fallback,paired bootstrap
and fullsave-scorepipeline passed. Real-input-only audit performed separately.
No real-data model fitting executed by assistant. This remains historically
exposed research; a baseline win is not guaranteed and needs later confirmation.

Actual-input audit:48000rows/18385origins passed. January's first inner fit has
no stable eligible features after the historical-anchor warmup, so January's
primary selection must use the median fallback. This is intentional conservative
handling of insufficient earlier history, not permission to relax the gate.
