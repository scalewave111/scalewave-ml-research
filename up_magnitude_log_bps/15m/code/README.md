# up_magnitude_log_bps - 15m code

Original horizon scripts and their local Python dependencies are included. Shared scripts may train both UP and DOWN; this folder does not change their target behavior. Files from other horizons are included only when imported or referenced by the horizon scripts.

Primary horizon files:
- `build_excursion_15m_final_report.py`
- `build_excursion_15m_report.py`
- `check_excursion_followup_15m_synthetic.py`
- `check_excursion_search_15m_synthetic.py`
- `check_excursion_tuning_15m_synthetic.py`
- `discover_excursion_features_15m.py`
- `improve_excursion_features_15m.py`
- `render_excursion_15m_final_report.py`
- `render_excursion_15m_report.py`
- `run_excursions_15m.py`
- `tune_excursions_15m.py`

These are copies of existing research code, not a newly executed pipeline. Existing data paths, prior-run references and report-builder layouts must be mapped before execution from this new folder structure. Raw datasets, saved models and full predictions remain in the original workspace/shared storage. No training or inference was performed during reorganization.
