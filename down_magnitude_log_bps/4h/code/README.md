# down_magnitude_log_bps - 4h code

Original horizon scripts and their local Python dependencies are included. Shared scripts may train both UP and DOWN; this folder does not change their target behavior. Files from other horizons are included only when imported or referenced by the horizon scripts.

Primary horizon files:
- `build_excursion_4h_report.py`
- `discover_excursion_features_4h.py`
- `improve_excursion_features_4h.py`
- `rank_excursion_features_4h.py`
- `render_excursion_4h_report.py`

These are copies of existing research code, not a newly executed pipeline. Existing data paths, prior-run references and report-builder layouts must be mapped before execution from this new folder structure. Raw datasets, saved models and full predictions remain in the original workspace/shared storage. No training or inference was performed during reorganization.
