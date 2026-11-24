# Check and replay magnitude research

Run these commands from this magnitude_48h folder in PowerShell. Do not run historical training scripts just to inspect results.

Check retained source hashes and current file manifest (standard-library Python):

```powershell
python -B tools/validate_handoff.py
```

For model replay, use the tested dependencies in environments/magnitude/requirements.txt. Reuse a suitable environment, or create one:

```powershell
python -m venv .venv-magnitude
& '.\.venv-magnitude\Scripts\python.exe' -m pip install --no-cache-dir -r environments/magnitude/requirements.txt
& '.\.venv-magnitude\Scripts\python.exe' -B tools/replay_magnitude.py --data-root 'APPROVED_PREPARED_DATA_FOLDER'
```

Replace APPROVED_PREPARED_DATA_FOLDER with the authorised location of the four X/y Parquet files. The wrapper verifies input/model/source hashes and runs the retained inference replay without training or copying data.

Expected historical replay: PASS, 18,481 rows, MAE 180.79851809782613, Spearman IC 0.07852651265983954, new_fits 0. The current reorganization checks file integrity; it does not rerun this data-level replay.

A magnitude-only summary PDF can be generated from reports/research_summary.md with python -B tools/build_summary_pdf.py in an environment containing PyMuPDF. Regeneration changes its file hash; update the manifest after review.
