# BTC 48h team playground

## Two objectives. One shared test bed.

1. **Magnitude:** improve absolute endpoint move forecasts beyond frozen MP0004. Primary score: MAE.
2. **Direction:** improve UP/DOWN probabilities beyond frozen P3908. Primary score: log loss; Brier must also improve.

| Track | Original no-feature baseline | Frozen model to beat |
|---|---:|---:|
| Magnitude MAE (log bps) | 187.831072 | **182.036977** |
| Direction log loss | 0.692022854 | **0.684958481** |
| Direction Brier | 0.249437861 | **0.245924833** |

These scores belong to the included development rows. Compare predictions on those exact rows; do not compare a new sample against an old number. `benchmarks/scores.json` has full precision.

## Start in five minutes

Extract the ZIP completely. Open a terminal in the extracted folder. Python 3.12 was used to validate this delivery.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python tools/playground.py verify
.venv\Scripts\python tools/playground.py replay --output submissions/member_01/baseline_replay.csv
.venv\Scripts\python tools/playground.py score --predictions submissions/member_01/baseline_replay.csv --metadata submissions/member_01/replay_metadata.json --output submissions/member_01/replay_check
```

The replay should reproduce both frozen models, not beat them. Do not edit original files to force a pass. View `research/OVERVIEW.html` for the process and findings. The exact fitted coefficients/preprocessing are readable JSON, with no pickle required.

## The playground is already defined

- BTCUSDT, decision every 30 minutes, future horizon 48 hours.
- Training: 29,615 rows, 2022-04-22 00:30 through 2023-12-29 23:30 UTC.
- Boundary purge: 96 rows. No purged observations are supplied as training rows.
- Development: 18,481 rows, 2024-01-01 through 2025-01-22 UTC.
- All 522 original F4Z fields supplied, with invalid readings already masked to NaN. Corrected full-training admission identifies 384 fields with >=1,000 finite readings and >=2 distinct values. Recheck admission inside each inner fold; do not use final admission to infer historical fold availability.
- No changing Y, horizon, cadence or evaluation rows. Missing X must not remove scored rows. Missing Y is never imputed.
- Models can differ. Select hyperparameters, preprocessing and features using earlier training folds only. No random time-series splits. `training_folds()` supplies three purged expanding folds.
- Initial budget: **20 fits per teammate**, including CV fits and failed fits; 80 team-wide. Register any extension before running it. Do not reset the count with a new run name.
- Both tasks remain the only objectives. No extremes, room-only scoring, trading optimization, new targets or model search launched by this package.

## Four teammates

| Folder | Suggested first focus | Question |
|---|---|---|
| member_01 | Magnitude, additive regularization | Can broader values add beyond the liquidation pair? |
| member_02 | Magnitude, nonlinear effects | Can a bounded nonlinear model improve beyond additive effects? |
| member_03 | Direction, additive regularization | Can broader values improve probability quality beyond basis + variance? |
| member_04 | Direction, conditional effects | Can a bounded interaction model improve both log loss and Brier? |

These are suggested divisions, not extra objectives or compulsory algorithms. Everyone uses the same data and scorer. A member working on one task submits the frozen forecast for the other task. Never count that unchanged track as a new result.

## Every submission

1. Copy `submission_template.json` into your member folder and complete the hypothesis, input list, fit budget and selection plan **before fitting**.
2. Implement the procedure; `tools/challenger_template.py` is the entry-point template. Record each attempt in your `experiments.csv` (including failures). Selection based on old reported winners remains historical development selection.
3. Export **all** development predictions as CSV: `decision_time,magnitude_log_bps,p_up`, in canonical order. Dates must include UTC. Do not submit predicted class labels instead of probabilities.
4. Submit code, environment lockfile, model, transformations, final metadata and fit ledger. Provide one replay command. No label columns may enter X.
5. Run the common scorer into a new result folder. Review monthly results and paired 7/14-day block uncertainty, not only averages.

`magnitude_average_win` means lower MAE than both M0 and MP0004. `direction_average_win` means lower log loss **and Brier** than both M0 and P3908. These flags are scoreboard results, not automatic promotion. Intervals are nominal diagnostics and do not correct the entire historical/team search. Human review must check lineage, practical effect, concentration and code. Self-declared metadata is not proof of no leakage.

## Evidence status

This is an **open development playground**, not a hidden contest or untouched holdout. The labels are included for reproducibility and scoring; they must not be used for final model fitting. Past features, pairs and models were selected using this history. Repeatedly inspecting team scores adds selection exposure. A credible survivor must be frozen and tested on genuinely fresh data later. No claim of validated alpha follows from winning this scoreboard.

## Contents

- `data/`: prepared train/development X and Y, split ledger, field dictionary and corrected admission.
- `benchmarks/`: exact two frozen model cards, M0 constants, saved development predictions and replay scores.
- `tools/`: common loader, folds, frozen model replay, scorer and challenger template.
- `research/`: team findings, top-20 tables, HTML overview, original top.md/top1.md and selected original contracts/verification.
- `reference/`: source package entry notes and feature/formula documentation.
- `submissions/member_01..04/`: separate work areas and metadata/ledger templates.
- `MANIFEST.json`: hashes of immutable delivered files. Submission folders are excluded so members can work.

This ZIP is self-contained for the frozen sampled experiment. The 15.6 GB X source ZIP and 3.0 GB canonical Y source ZIP are **not** duplicated. Their local source paths/recorded identities are in `PROVENANCE.json`. Raw exchange reconstruction, new features/horizons and original-price backtests need those separate sources. Prepared exports retain historical PROVISIONAL/source-timing limits; unresolved funding-boundary construction is not repaired here.

Expected signed return and a coherent return distribution are not current outputs. Magnitude is a conditional median, not an expected signed return. Do not multiply it by `(2*p_up-1)` and label the result expected return.
