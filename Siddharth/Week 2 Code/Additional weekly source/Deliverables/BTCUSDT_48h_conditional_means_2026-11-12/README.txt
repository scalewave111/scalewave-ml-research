BTCUSDT 48-hour conditional mean endpoint-return handoff

1. Files and coverage
conditional_mean_predictions.csv has exactly the five requested columns.
UTC half-hour grid: 2024-01-01T00:00:00Z through 2025-01-22T23:30:00Z, inclusive.
Grid rows: 18624. Rows with both predictions: 18481.
Available predictions: 2024-01-01T00:00:00Z through 2025-01-22T00:00:00Z.
Rows with missing predictions: 143. See missing_predictions.csv.
Absent source timestamps are retained as blank predictions, not zero. No inference
or interpolation was performed for gaps. Source eligibility reasons for internal
gaps have not been independently reconstructed. Rows after the source's final
timestamp are beyond available coverage. Both side predictions are supplied at
every available source timestamp; future realized direction is never used to
choose, filter, or replace either side during export.

2. Outcome and units
For forecast origin t, r = 10000 * (P(t+48h)/P(t) - 1), ordinary basis points.
UP means r > 0. DOWN means r < 0. Flat outcomes are excluded from side training;
the source evaluation runners reject flats in their development population.
mean_return_if_up_bps estimates E[r | information at t, r > 0].
mean_return_if_down_bps estimates E[r | information at t, r < 0] and is NEGATIVE.
The source models train on absolute ordinary-bps endpoint returns separately by
realized historical side. Targets were converted observation by observation from
stored signed log bps with 10000*expm1(log_bps/10000) BEFORE fitting.
The DOWN source is positive magnitude; export negates it. UP values are copied
unchanged. Predictions are not log bps, medians, maximum excursions, or error
metrics. The squared-error objective targets conditional averages; calibration
and predictive accuracy are not guaranteed by the objective alone.
100 bps equals one percentage point. No baseline averages substitute for models.

3. Reference price
The original package CONTRACT.json identifies P as the 'historical canonical
f066 reference price'. The exact exchange, construction and price type (close,
mid, mark, last or aggregation) are NOT established by the inspected contract.
Do not assume an executable trade price. The prepared-data contract records a
prior audit matching 48-hour log targets to sums of 576 canonical five-minute
log returns; this export does not rerun that audit or reconstruct raw prices.
Source configuration includes the target contract and canonical-return helper.
To resolve the remaining definition, request the canonical f066 label-builder
script/configuration and reference-price specification from its data owner,
including price timestamps, exchange/aggregation and missing-price treatment.

4. Model policy and refits
UP: saved squared-error Random Forest, 128 trees, max depth 3, min leaf128,
bootstrap4096, seed48. Its 4/5/8 feature pack was selected separately each fold
using preceding validation MSE. Uses UP_rf_mean_bps, not the selected MLP/RF column.
DOWN: saved Ridge(alpha1) plus squared-error RF with the same RF capacity and
eight features. RF blend weight selected by row-weighted MSE across earlier
validation cutoffs 9/6/3 months before issuance. Weight zero means fitted Ridge
alone; it is still a feature-based learned model, not a constant side average.
2024-01-01T00:00:00Z: UP RF four features; DOWN RF weight 0.25
2024-04-01T00:00:00Z: UP RF four features; DOWN RF weight 0.25
2024-07-01T00:00:00Z: UP RF four features; DOWN RF weight 0.0
2024-10-01T00:00:00Z: UP RF eight features; DOWN RF weight 0.0
These fits use expanding historical data with label_end STRICTLY before issue.
Training-row counts, latest training outcome times, artifact IDs and SHA256
identities are in training_refit_ledger.csv. An origin's outcome must have fully
matured before it can enter a later fit. Model weights/preprocessing are fixed
within each quarterly evaluation block. October's fit serves the final block
through the available January 2025 endpoint; there was no January 2025 refit.
The UP and DOWN policies use historical actual sides only to form training sets.
At inference, each model predicts every origin using its input features.

5. Freeze and forecast availability
This is a fixed, hash-identified export of retrospectively generated walk-forward
predictions from completed November 2026 runs. It is NOT evidence that predictions
were generated live in 2024/2025. The UP-RF/DOWN-hybrid family pairing was proposed
after inspecting historical development scores; it is not a prospectively frozen,
untouched-holdout policy. This selection exposure must follow downstream reports.
timestamp_utc is the forecast origin; issue is the simulated fit/refit cutoff.
No actual historical publication/arrival timestamps are recorded for these
predictions. At-origin availability is assumed by the research setup, not proven
by a live availability audit. The original package retains PROVISIONAL source
timing limitations and unresolved funding-boundary construction. Derived price
features use canonical five-minute return endpoints strictly before origin;
that does not independently certify all other feature availability.

6. Export and configuration scripts
export_conditional_mean_handoff.py reproduces this export using Python's standard
library and the existing workspace source CSVs/artifacts. It performs no training.
Example: python export_conditional_mean_handoff.py --workspace "PATH_TO_WEEK_2" --output-dir "NEW_EXPORT_FOLDER"
Use a new output path; existing exports are never overwritten. The script is
pinned to the source prediction hashes and run IDs in export_configuration.json.
training_scripts contains the exact existing training-code snapshots; source
configuration contains contracts, completion and replay-audit records and UP
feature-pack selections. Training scripts still need the original Week 1 data,
helper modules and installed dependencies; the package is not a standalone model
retraining environment and does not include weights or raw data.

7. Verification
Source completion records, contracts, model hashes and saved replay audits were
checked. Export was reopened and all values compared to source values, allowing
only the declared DOWN sign change. Schema, unique grid, horizons, issuance and
training-label cutoffs were checked. Missing rows are explicitly listed.
CHECKSUMS.json records each packaged file's SHA256. No new performance, calibration
or fresh-data claim is made by this export.
