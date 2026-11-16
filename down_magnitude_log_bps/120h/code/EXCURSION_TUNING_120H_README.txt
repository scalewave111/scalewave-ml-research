120-HOUR UP/DOWN MAE REFINEMENT

Run from November/Week 2:
python .\Code\tune_excursions_120h.py

Input and saved-model replay only (no training):
python .\Code\tune_excursions_120h.py --check-inputs
Synthetic checks:
python .\Code\check_excursion_tuning_120h.py

Reference: excursion_pipeline_120h_v1/20261113T114805583307Z/
price_followup/20261113T120541578982Z. Existing artifacts remain unchanged.
Both targets remain canonical UP/DOWN excursions in log bps, including valid
zeros. Same 18,193 eligible development origins; invalid labels stay excluded.
Jan/Apr/Jul/Oct expanding refits; October model also predicts January 2025.
Every inner and outer training label must end strictly before its fit boundary.
The original 48h gap remains; additional strict 120h purges are applied.

Search independently for each target/refit:
 - Current features; current plus eight new past-120h price measurements;
   current F4Z features plus all 59 causal price features.
 - Full mature history or the preceding 365 calendar days.
 - Absolute-error histogram boosting, learning rate .05, seed48,
   no random early stopping, training-only binning, native missing values.
 - Small: 120 iterations, depth2, 4leaves, minleaf128, L2=10.
   Medium: 160 iterations, depth3, 7leaves, minleaf96, L2=5.
   Flexible: 200 iterations, depth4, 15leaves, minleaf64, L2=10.
 - Pure candidate or 50/50 blend with current model.
New 120h price measurements use return endpoints strictly before the origin:
realized volatility, up/down semivolatility, net return, past range,
distance from high/low, maximum absolute five-minute return.

Three earlier quarterly validation windows select minimum pooled MAE.
Acceptance requires >=0.1% MAE reduction and >=2/3 window wins against BOTH
current model and training median, plus pooled MSE no worse than current.
Otherwise retain current model. Median is a benchmark, not an automatic
replacement. Earlier-selected reference features reuse validation history;
this is adaptive research, not a nested or independently confirmed experiment.

18 candidates x3 windows x8 target/refits =432 inner candidate fits,
plus24 inner controls,8 outer control checks and up to8 outer candidates.
Defaults to2threads. No automatic resume; reruns create separate directories.

Output: Code/experiments/excursion_tuning_120h_v1/<timestamp>/
metrics.csv compares previous/selected/mean/median overall and by period.
Also saves predictions, validation trials, selections before outer fits,
training-boundary ledgers, models/replay, input/source/guidance hashes,
source snapshots and COMPLETE.json after all checks succeed.
All predictions remain nonnegative log bps. No direction or signed-return model.

Validation gains do not guarantee lower historical evaluation errors.
Evaluation has already been exposed; original source-availability limits remain.
Preparation includes synthetic tests and real-input/saved-reference replay;
new real-market model training is user-run.
