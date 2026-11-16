# down_magnitude_log_bps - 16h code

Original horizon scripts and their local Python dependencies are included. Shared scripts may train both UP and DOWN; this folder does not change their target behavior. Files from other horizons are included only when imported or referenced by the horizon scripts.

Primary horizon files:
- `build_excursion_report_16h.py`
- `check_excursion_followup_16h_synthetic.py`
- `check_excursion_search_16h_synthetic.py`
- `check_excursion_tuning_16h.py`
- `discover_excursion_features_16h.py`
- `improve_excursion_features_16h.py`
- `render_excursion_report_16h.py`
- `run_excursions_16h.py`
- `tune_excursions_16h.py`

These are copies of existing research code, not a newly executed pipeline. Existing data paths, prior-run references and report-builder layouts must be mapped before execution from this new folder structure. Raw datasets, saved models and full predictions remain in the original workspace/shared storage. No training or inference was performed during reorganization.
