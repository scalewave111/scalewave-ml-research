# Time-to-high research

We predict how many minutes it takes to reach the highest price within a future window. If the maximum occurs more than once, use the earliest occurrence; minute zero is included.

| Evidence | Open |
| --- | --- |
| Three baselines across ten horizons | [Baseline matrix](baseline_matrix.csv), [receipt](BASELINE_RESULTS.json) |
| Sampled feature coverage | [Audit](FEATURE_AUDIT_RESULTS.json), [verification](feature_audit_verification.json) |
| Original pilot | [Pilot matrix](pilot_matrix.csv), [verification](pilot_verification.json) |
| Completed staged screen | [Horizon matrix](Results/staged/horizon_matrix.csv), [selected folds](Results/staged/selected_policy_folds.csv), [verification](Results/staged/verification.json) |
| Individual cheap-screen rankings | [Ranking index](../../results/time_to_high/README.md) |
| Current next step | [Continuation guide](../../CONTINUE_RESEARCH.md) |

The staged method bins each of 386 features using training quantiles and training target medians. For each outer fold, the three best inner features are shortlisted. Six tree settings are tested: depths 2, 3, 5 and leaves 64, 256; criterion absolute_error and seed 48. Inner scores choose one feature/settings combination. Outer scores never choose the shortlist or settings.

This is not exhaustive tuning of every feature and is not a feature-combination model. A cheap shortlist can miss useful tree features.

Scripts, notebooks, configuration and receipts remain siblings because notebook builders depend on that layout. Do not click Run All simply to read a completed notebook: some cells authenticate to Drive, train models and write checkpoints. The original verify_staged_results.py also writes progress/verification files. Use the read-only packaging verifier in the reproducibility guide instead.

Runtime receipts include synthetic-only and cloud-pending statuses. research_progress.json also records user-reported cloud worker evidence. Preserve those differences. No complete local checkpoint for the paused exhaustive screen was retrieved during packaging.
