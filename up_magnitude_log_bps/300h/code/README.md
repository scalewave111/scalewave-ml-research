# up_magnitude_log_bps - 300h code

Original horizon scripts and their local Python dependencies are included. Shared scripts may train both UP and DOWN; this folder does not change their target behavior. Files from other horizons are included only when imported or referenced by the horizon scripts.

Primary horizon files:
- `build_report_300h.py`
- `check_compact_features_300h.py`
- `check_excursion_followup_300h_synthetic.py`
- `check_excursion_search_300h_synthetic.py`
- `check_long_features_300h.py`
- `check_mae_both_300h.py`
- `check_mae_up_300h.py`
- `check_magnitude_models_300h.py`
- `compact_search_300h.py`
- `discover_excursion_features_300h.py`
- `fit_mae_both_300h.py`
- `fit_mae_up_300h.py`
- `fit_magnitude_models_300h.py`
- `focused_market_trees_300h.py`
- `improve_excursion_features_300h.py`
- `long_features_300h.py`
- `mae_both_300h.py`
- `mae_up_300h.py`
- `magnitude_inputs_300h.py`
- `magnitude_models_300h.py`
- `market_state_features_300h.py`
- `neural_market_300h.py`
- `refine_long_features_300h.py`
- `render_report_300h.py`
- `run_excursions_300h.py`
- `search_compact_features_300h.py`

These are copies of existing research code, not a newly executed pipeline. Existing data paths, prior-run references and report-builder layouts must be mapped before execution from this new folder structure. Raw datasets, saved models and full predictions remain in the original workspace/shared storage. No training or inference was performed during reorganization.
