300-HOUR UP/DOWN EXCURSION WORKFLOW

From November/Week 2:
python .\Code\run_excursions_300h.py

Input audit only (no training):
python .\Code\run_excursions_300h.py --check-inputs

Uses canonical horizon_minutes=18000 BTCUSDT up_magnitude_log_bps and
down_magnitude_log_bps: the positive upward/downward extremes of the future
one-minute reference-price path including origin through 300 hours later.
Both targets include all valid rows and zeros, with no endpoint-direction filter.
All metrics and predictions use log bps.

Stage 1 repeats the 4h search using 300h labels and strict 300h maturity purges:
522 original masked F4Z candidates, earlier quarterly validation, independent
UP/DOWN selection, Ridge and MAE histogram boosting, median fallback.
Original 29615 initial / 18481 development input keys and original 48h boundary gap
are retained. Jan/Apr/Jul/Oct 2024 expanding refits; October also predicts Jan2025.
The inherited 48h gap alone is insufficient for 300h targets: each fit excludes
additional origins until origin + 300h is strictly before issuance. Thus 29615
is the initial input inventory, not the January fit's eligible training count.
See EXCURSION_FEATURE_SEARCH_300H_README.txt for full search settings.

Stage 2 repeats the 4h price-feature follow-up using the NEW completed 300h run.
It adds 51 strictly past-price inputs over 1/4/12/24/48/168h and volatility ratios.
Past feature lookbacks remain unchanged; only the forecast horizon becomes 300h.
MAE boosting tests block additions, all-price combinations and group removals.
Earlier validation must improve on original and median by >=0.1% and win at
least two of three windows, otherwise original predictions are retained.
The first-stage reference may select a tree, Ridge or median independently.
Original predictions and saved models are replay-checked.

Outputs: Code/experiments/excursion_pipeline_300h_v1/<timestamp>/
  search/<timestamp>/ and price_followup/<timestamp>/
Each stage saves predictions.csv, metrics, selections, model files, source/input
hashes, CONTRACT.json and COMPLETE.json. Only COMPLETE.json marks stage success.
The follow-up includes original and selected predictions plus mean/median controls.
No automatic resume. To run a follow-up after an interrupted second stage:
python .\Code\improve_excursion_features_300h.py --reference-run "PATH_TO_COMPLETED_SEARCH_RUN"

Dependencies and Week 1 canonical-return helper are inherited from the 4h workflow.
Historical evaluation has already been exposed; improvements are not guaranteed.
Original feature availability/provisional source limitations remain.
No real-data training was executed while preparing these runners.

300H DATA AUDIT (2026-11-13)
1008 development labels are unusable: 504 INVALID_PRICE_PATH and 504 OUTSIDE_RELEASE.
The eligible population is 47088 total rows and 17473 development origins.
January has 29111 mature training rows after the strict 300h purge.
Both stages use the same eligible keys. Stage 1 saves excluded_labels.csv.
No invalid label is filled with zero; valid zero excursions remain included.
Actual-input audit and synthetic search/price-feature checks passed. No real-data
training was executed. Full follow-up reference replay awaits a completed first stage.
