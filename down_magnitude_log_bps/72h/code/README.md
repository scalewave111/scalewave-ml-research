# down_magnitude_log_bps - 72h code

Original horizon scripts and their local Python dependencies are included. Shared scripts may train both UP and DOWN; this folder does not change their target behavior. Files from other horizons are included only when imported or referenced by the horizon scripts.

Primary horizon files:
- `build_report_72h.py`
- `check_compact_trees_72h.py`
- `check_excursion_diagnostics_72h.py`
- `check_excursion_followup_72h_synthetic.py`
- `check_excursion_search_72h_synthetic.py`
- `check_excursion_tuning_72h.py`
- `check_extra_trees_excursions_72h.py`
- `check_focused_hgb_both_72h.py`
- `check_focused_market_down_72h.py`
- `check_gru_market_up_72h.py`
- `check_guarded_down_72h.py`
- `check_long_features_72h.py`
- `check_market_state_down_72h.py`
- `check_market_state_fast_72h.py`
- `check_neural_excursions_72h.py`
- `check_neural_market_down_72h.py`
- `check_quarterly_anchor_down_72h.py`
- `check_rolling_medians_72h.py`
- `check_transformed_models_72h.py`
- `compact_trees_72h.py`
- `compare_rolling_medians_72h.py`
- `diagnose_excursions_72h.py`
- `diagnose_fresh_features_72h.py`
- `discover_excursion_features_72h.py`
- `discover_long_features_72h.py`
- `export_report_72h_pdf.py`
- `fit_compact_trees_72h.py`
- `fit_extra_trees_excursions_72h.py`
- `fit_focused_hgb_both_72h.py`
- `fit_focused_market_down_72h.py`
- `fit_gru_market_up_72h.py`
- `fit_guarded_down_72h.py`
- `fit_market_state_down_72h.py`
- `fit_neural_excursions_72h.py`
- `fit_neural_market_down_72h.py`
- `fit_quarterly_anchor_down_72h.py`
- `fit_transformed_models_72h.py`
- `focused_market_trees_72h.py`
- `fresh_features_72h.py`
- `gru_market_up_72h.py`
- `guarded_down_72h.py`
- `improve_best_models_72h.py`
- `improve_excursion_features_72h.py`
- `market_state_features_72h.py`
- `market_state_trees_72h.py`
- `neural_market_down_72h.py`
- `quarterly_anchor_down_72h.py`
- `run_excursions_72h.py`
- `run_model_challengers_72h.py`
- `transformed_models_72h.py`
- `tune_excursions_72h.py`

These are copies of existing research code, not a newly executed pipeline. Existing data paths, prior-run references and report-builder layouts must be mapped before execution from this new folder structure. Raw datasets, saved models and full predictions remain in the original workspace/shared storage. No training or inference was performed during reorganization.
