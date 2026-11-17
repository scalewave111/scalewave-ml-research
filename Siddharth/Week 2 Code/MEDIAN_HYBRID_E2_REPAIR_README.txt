RUN FROM NOVEMBER/WEEK 2
python .\Code\repair_package_median_hybrid_e2_48h.py

Default --threads 2. Resume an interrupted run with:
python .\Code\repair_package_median_hybrid_e2_48h.py --resume "C:\full\path\to\run"
Use identical inputs, code, package versions and thread count.
Optional: --self-test (synthetic fits); --check-data-only (real-input loading
and saved baseline inference, no model fitting).

WHAT CHANGES
Frozen baseline: package_median_hybrid_signed_48h_v1/20261113T071027569020Z.
Keeps its exact seven base inputs, median regression, three absolute-error
forests, seeds, bootstrap caps and quarter-specific blend/settings.
Each outer issue's card is also used for that issue's earlier monthly OOF
and validation baseline fits. These inherited historical cards were already
selected on exposed history; this is not fresh nested validation of them.

Monthly out-of-fold hybrid predictions use expanding side-specific training,
at least1024 rows, and only labels that finished before the forecast month.
The correction target is actual ordinary-bps magnitude minus that OOF hybrid
forecast. Correction fitting uses only residuals whose48h outcomes matured.

Same E2 correction technique:235 tree inputs, including24 long-price measures
over24h/48h/7d/30d;15 constrained LightGBM L2 configurations; expanding history;
latest20percent early-stop slice with48h label purge;1000tree cap,50patience;
full mature-history refit at selected iteration count. Same normalization,
regularization, sampling and seed48 as E2. Final magnitude=max(0,base+correction).
This is mean-oriented error correction of a median base, NOT a pure median
forecast. UP/DOWN remain endpoint-conditional magnitudes, not path excursions.

SELECTION AND CONTROLS
Per side, select correction by pooled magnitude MSE across three earlier
two-month windows starting9/6/3months before issuance. No outer-outcome tuning.
corrected: best earlier-validation correction, full weight, like E2.
selected: use that correction only when earlier MSE beats unchanged hybrid;
otherwise retain unchanged hybrid exactly. This fallback cannot guarantee
better later results. No E3 history search or E4 joint/shrinkage policy.
Fixed v45-style direction probabilities combine the two magnitudes.
Compare unchanged hybrid,corrected,selected,E2,Ridge,zero on same18481origins.

OUTPUT
Code/experiments/package_median_hybrid_e2_repair_48h_v1/<timestamp>/
SUMMARY.txt, predictions.csv, cycle_signed_metrics.csv,
cycle_magnitude_metrics.csv, cycle_2_3_4_mean_metrics.csv,
prediction_vs_actual_with_direction.csv, tail_error_contributions.csv,
validation_metrics.csv, OOF_AUDIT.json, SELECTIONS_AND_FIT_AUDIT.json,
CONTRACT.json, replay-checked inference bundles, hashed cache and COMPLETE.json.
Cycle1-4,January2025,pooled,and unweighted cycle2-4 means reported.
Tail threshold is training-only P90 of absolute endpoint returns per issuance.
COMPLETE only written after outputs and saved-bundle replay pass.

RUNTIME
Monthly base training is much heavier than Ridge E2: three128-tree
absolute-error forests per side/card/month, plus validation base models.
Identical cards/months share a checksum-verified cache. Completed monthly
models and correction models survive resume; partially written files without
matching checksums are recomputed.15x3x2x4=360 inner correction fits before
cache sharing plus8 selected outer fits; each includes early stopping/refit.
Do not assume a short runtime. No guaranteed improvement on exposed history.
