48-HOUR NET MAGNITUDE FEATURE AND MODEL SEARCH

Canonical target: net_magnitude_log_bps = 10000 * abs(ln(P_end/P_start)).
Verified definition: October Week 3 canonical package team_tools/build_dataset.py,
signed = log_bps(endpoint); net_magnitude_log_bps = np.abs(signed).
This is a nonnegative endpoint magnitude. It is not UP minus DOWN excursion.

RUN IN POWERSHELL
cd "C:\Users\Siddh\Desktop\ML\November\Week 3"
python Code/fit_net_magnitude_48h.py

FOUR SEPARATE MODELS ACROSS ALL PERIODS (latest requested workflow)
python Code/fit_net_magnitude_48h_separate.py
Runs HGB, ExtraTrees, MLP and GRU sequentially as four independent experiments.
Each uses only its own family across all periods. Earlier validation chooses
its feature pack independently; there is no model-family mixing or median
fallback in the primary forecast. All four use the same target, time grid,
validation windows and estimator cap. Each has separate predictions, metrics,
models and reports. No training is launched by creating these scripts.
This latest mode has not been executed or tested.

Run one family independently:
python Code/fit_net_magnitude_48h.py --pure-model hgb
python Code/fit_net_magnitude_48h.py --pure-model extra
python Code/fit_net_magnitude_48h.py --pure-model mlp
python Code/fit_net_magnitude_48h.py --pure-model gru

Separate output directories for each family:
Code/experiments/net_magnitude_48h_<family>_only_v1/<timestamp>/
Report/net_magnitude_48h_<family>_only/<timestamp>/
Compare hgb/extra/mlp/gru rows respectively in their metrics.csv files.
The separately shown guarded_selected row is an audit comparison, not the
primary pure-model forecast. Validation-gate acceptance is not guaranteed.

Optional input audit (no fitting):
python Code/fit_net_magnitude_48h.py --check-inputs
Optional synthetic checks (does fit synthetic models):
python Code/fit_net_magnitude_48h.py --self-test
Trees only:
python Code/fit_net_magnitude_48h.py --models hgb extra

PURE HGB ACROSS ALL PERIODS (user clarification)
python Code/fit_net_magnitude_48h.py --hgb-only
Only HGB is fitted. For each quarterly issue, earlier validation selects its
feature pack. The primary hgb/selected forecast always comes from HGB, with no
GRU, MLP, ExtraTrees or median fallback. Mean/median remain comparison baselines.
guarded_selected records the gate-based policy separately; a pure HGB forecast
does not imply the acceptance gate passed. Up to60 inner and8 outer HGB fits.
Outputs: Code/experiments/net_magnitude_48h_hgb_only_v1/<timestamp>/ and
Report/net_magnitude_48h_hgb_only/<timestamp>/. This mode has not been executed.

Uses existing Week 2 code and Week 1/October sources in place; override
--week2-code and --canonical-root if paths change. Requires numpy, pandas,
scikit-learn, duckdb, joblib, threadpoolctl; GRU requires torch.

FLOW
1. Reuse checksum-verified canonical 48h labels and established feature loader.
   Retain VALID, nonnegative labels and source-binding transition exclusions.
   Read actual canonical net target; direction is used only for a label audit.
2. Reuse market/price features and create 48 new causal price features:
   4/24/48/168/336h volatility, signed/absolute moves, absolute movement sum,
   trend efficiency, upside variance share, jump concentration, RMS return,
   positive-return share, plus 4/24/48 versus168h volatility ratios.
   Every rolling window excludes return endpoints at/after decision time.
3. Compare five packs: market, prior price, new price, all, market+new.
   This is a feature-group search, not exhaustive individual-feature search.
   Per-fit quality filter: >=80% finite training coverage and variation.
   Missing flags; training median/IQR scaling and clipping to +/-8.
4. Compare HGB, ExtraTrees, MLP16 and GRU16 independently for the net target.
   GRU uses snapshots at12/8/4/0h, from the timeline before label filtering.
   All valid training labels are used, with no sign conditioning.
5. At Jan/Apr/Jul/Oct2024 issues, select using three earlier two-month windows
   starting9/6/3months before issue. Training label_end must be strictly earlier
   than each fitting cutoff. Inner validation label_end must precede outer issue.
   Rank pooled MAE. Accept a model only with >=0.1% median-baseline improvement,
   >=2/3 window wins/ties and no >2% per-window MAE loss; otherwise use median.
   Keep best diagnostic candidate separate. Report MAE/RMSE/bias against mean
   and median. Later evaluation outcomes never choose that issue's model.

Default budget: up to240 inner and8 outer fits (5packs x4models x3windows
x4issues). Estimator sample capped at6000 evenly spaced mature rows; quality
filter, preprocessing, target scaling and baselines use all mature rows.
Fixed seeds48; no resume or runtime estimate. GRU CPU/24epochs, MLP40epochs.

OUTPUTS
Code/experiments/net_magnitude_48h_v1/<UTC timestamp>/
  contract/source hashes, population audit, feature formulas, all validation
  trials, exact fitted features, saved models/replay receipts, predictions,
  metrics and COMPLETE.json only after successful completion.
Report/net_magnitude_48h/<same timestamp>/
  SUMMARY.txt, metrics.csv, feature_selection.json.

STATUS
Generated for user execution. Initial HGB/ExtraTrees/MLP synthetic checks passed
before user requested no execution. Initial input audit exposed a prefix-check
boundary error (dropping the origin index); corrected to retain origin while
closed-left windows exclude its value. Full-timeline GRU support was then added.
These latest changes, GRU and complete end-to-end execution have NOT been run.
No real-data model training or new model performance result is established.
Historical evaluation has already been exposed; no fresh-test/significance claim
or guaranteed baseline improvement. Upstream masked feature semantics inherited.
