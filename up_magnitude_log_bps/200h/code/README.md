# up_magnitude_log_bps - 200h code

Original horizon scripts and their local Python dependencies are included. Shared scripts may train both UP and DOWN; this folder does not change their target behavior. Files from other horizons are included only when imported or referenced by the horizon scripts.

Primary horizon files:
- `build_report_200h.py`
- `check_dual_objective_200h.py`
- `check_excursion_followup_200h_synthetic.py`
- `check_excursion_search_200h_synthetic.py`
- `check_long_feature_diagnostics_200h.py`
- `check_long_features_200h.py`
- `check_magnitude_models_200h.py`
- `check_transformed_dual_200h.py`
- `diagnose_long_features_200h.py`
- `discover_excursion_features_200h.py`
- `discover_long_features_200h.py`
- `dual_objective_200h.py`
- `fit_dual_objective_200h.py`
- `fit_magnitude_models_200h.py`
- `fit_transformed_dual_200h.py`
- `focused_market_trees_200h.py`
- `improve_excursion_features_200h.py`
- `long_horizon_features_200h.py`
- `magnitude_inputs_200h.py`
- `magnitude_models_200h.py`
- `market_state_features_200h.py`
- `neural_market_200h.py`
- `render_report_200h.py`
- `run_excursions_200h.py`
- `transformed_dual_200h.py`

These are copies of existing research code, not a newly executed pipeline. Existing data paths, prior-run references and report-builder layouts must be mapped before execution from this new folder structure. Raw datasets, saved models and full predictions remain in the original workspace/shared storage. No training or inference was performed during reorganization.
