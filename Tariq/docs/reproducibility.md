# Check results and replay models

**You do not need to train a model to check this handoff.** Run commands from the repository root in PowerShell. Existing research scripts may train or write outputs; use the specific commands below.

| Operation | What it checks | Data needed | Training? |
| --- | --- | --- | --- |
| Package checks | Files, hashes, links, notebook outputs and common credential patterns | Repository only | No |
| Timing saved-score verification | Receipt checksums, scores, inner choices, recorded purges/counts and actual frozen source hashes | Repository only | No |
| Magnitude inference replay | Reproduce saved predictions using retained fitted models and original prepared inputs | Four approved X/y files | No |
| Synthetic tests | Code behavior on small generated test values | Repository plus dependencies | Small synthetic fits only |
| Retraining | Fit a model again on real research data | Approved sources and a separately reviewed experiment | Not run for this handoff |

## 1. Repository-only checks

These commands use the standard library and do not rewrite historical receipts:

```powershell
Set-Location C:ScaleWavescalewave-btc-research
python -B toolsalidate_handoff.py
python -B toolserify_timing_receipts.py
```

The timing wrapper runs the original verifier's checks after suppressing its two file-write statements in memory. It also hashes the actual frozen source files and pilot range-reader code. The original verifier remains byte-for-byte unchanged. Its raw-data and inference limits still apply.

## 2. Magnitude inference

The four current forest bundles, their preprocessing cards, predictions and v2 outer pair cards are packaged at their expected relative paths. Older v4 inner pair cards support the selection evidence. The models are period-specific development models; this package is not a deployed live forecasting service.

Create a local environment if you do not already have the tested dependencies:

```powershell
python -m venv .venv-magnitude
..venv-magnitudeScriptspython -m pip install -r environmentsmagnitude
equirements.txt
```

Use an approved folder containing X_train.parquet, y_train.parquet, X_development.parquet and y_development.parquet. On the original workspace:

```powershell
..venv-magnitudeScriptspython -B tools
eplay_magnitude.py --data-root 'C:ScaleWaveTeam 48_HTEAM_48H_PLAYGROUND_V1TEAM_48H_PLAYGROUND_V1data'
```

On a colleague's machine, substitute the approved input folder. The wrapper verifies all four data hashes, checks retained model/source hashes, and redirects only the original replay's four prepared-input reads to that folder. It runs the unchanged replay.py. It neither copies the datasets nor trains models. Only load the retained model binaries from the approved, hash-verified handoff.

Expected: PASS, 18,481 rows, MAE 180.79851809782613, Spearman IC 0.07852651265983954, new_fits 0. A mismatch is a failed replay; do not adjust expected scores to force it to pass.

The original replay command can also run if the four inputs are explicitly restored under playground/data. That directory is ignored by Git. The external-folder wrapper avoids an unnecessary local dataset copy.

## 3. Synthetic timing checks

```powershell
python -m venv .venv-timing
..venv-timingScriptspython -m pip install -r environments	iming
equirements.txt
..venv-timingScriptspython -B research	ime_to_high	est_staged_timing.py
..venv-timingScriptspython -B research	ime_to_high	est_drive_rate_guard.py
..venv-timingScriptspython -B research	ime_to_high	est_source_metadata_compat.py
```

These tests use synthetic/mock inputs. They do not read company archives or contact Drive. They check chronological selection, interrupted-run recovery, missing-value fallback and I/O compatibility. They do not establish predictive performance on real data.

## 4. PDF and earlier notebooks

The one-page summary PDF is already included. Regeneration uses PyMuPDF and writes reports/research_summary.pdf:

```powershell
python -B toolsuild_summary_pdf.py
```

Regeneration changes the package manifest's PDF hash; record and review any resulting changes. The earlier notebook builder's relative root was checked without running the builder. Do not execute it merely to read the notebook: it overwrites its generated notebook. The earlier synthetic test writes fixtures/results under research/feature_exploration/Reports; those outputs are ignored by Git.

## What was verified during packaging

See [packaging verification](../results/packaging_verification.json). Magnitude inference passed using the approved local inputs. Timing saved-score checks passed, including 38,600 cheap scores and 50 selected folds. The three named synthetic timing tests passed. No real-data training or remote archive download ran.

Full timing inference remains unavailable locally: the saved receipts are not a serialized collection of fitted models. Colleagues need source access and a separate approved execution plan for any new data-level replay. GitHub access alone is insufficient.
