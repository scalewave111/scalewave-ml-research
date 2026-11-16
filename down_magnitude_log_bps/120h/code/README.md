# down_magnitude_log_bps - 120h code

Original horizon scripts and their local Python dependencies are included. Shared scripts may train both UP and DOWN; this folder does not change their target behavior. Files from other horizons are included only when imported or referenced by the horizon scripts.

Primary horizon files:
- `build_report_120h.py`
- `check_dual_objective_120h.py`
- `check_excursion_followup_120h_synthetic.py`
- `check_excursion_search_120h_synthetic.py`
- `check_excursion_tuning_120h.py`
- `check_long_features_120h.py`
- `check_magnitude_models_120h.py`
- `check_mlp_compact_120h.py`
- `check_transformed_dual_120h.py`
- `discover_excursion_features_120h.py`
- `discover_long_features_120h.py`
- `dual_objective_120h.py`
- `fit_dual_objective_120h.py`
- `fit_magnitude_models_120h.py`
- `fit_mlp_compact_120h.py`
- `fit_transformed_dual_120h.py`
- `focused_market_trees_120h.py`
- `improve_excursion_features_120h.py`
- `magnitude_models_120h.py`
- `market_state_features_120h.py`
- `mlp_search_120h.py`
- `neural_market_120h.py`
- `render_report_120h.py`
- `run_excursions_120h.py`
- `transformed_dual_120h.py`
- `tune_excursions_120h.py`

These are copies of existing research code, not a newly executed pipeline. Existing data paths, prior-run references and report-builder layouts must be mapped before execution from this new folder structure. Raw datasets, saved models and full predictions remain in the original workspace/shared storage. No training or inference was performed during reorganization.
