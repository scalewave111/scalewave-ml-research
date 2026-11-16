# down_magnitude_log_bps - 48h code

Original horizon scripts and their local Python dependencies are included. Shared scripts may train both UP and DOWN; this folder does not change their target behavior. Files from other horizons are included only when imported or referenced by the horizon scripts.

Primary horizon files:
- `build_excursion_48h_report.py`
- `discover_mean_magnitude_features_48h.py`
- `fit_excursions_48h.py`
- `fit_hist_excursions_48h.py`
- `fit_package_median_hybrid_signed_48h.py`
- `fit_report_hybrid_means_48h.py`
- `lightgbm_repair_experiments_48h.py`
- `render_excursion_48h_report.py`
- `tree_interaction_signed_search_48h.py`
- `tree_signed_return_search_48h.py`
- `tune_conditional_mean_magnitudes_48h.py`
- `tune_lightgbm_residual_48h.py`
- `tune_ridge_mean_magnitudes_48h.py`

These are copies of existing research code, not a newly executed pipeline. Existing data paths, prior-run references and report-builder layouts must be mapped before execution from this new folder structure. Raw datasets, saved models and full predictions remain in the original workspace/shared storage. No training or inference was performed during reorganization.
