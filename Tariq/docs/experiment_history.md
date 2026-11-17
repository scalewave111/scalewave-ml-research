# Experiment history

The [experiment registry](../results/experiment_registry.csv) contains 82 entries. The original magnitude history has 56 entries, including early summary records and later detailed attempts. Registry entries are not a count of individual fits.

| Stage | Outcome | Evidence |
| --- | --- | --- |
| Frozen magnitude and refitted pair | Starting benchmark and small refit improvement | Magnitude milestone table and v2 results |
| Training windows and residual follow-up | Completed historical comparisons | v2/v3 receipts |
| Pair plus random forest | Improved benchmark | v4 results |
| Nonlinear/linear additions | Many failed to improve | v5/v6 and original history |
| RF tuning, bootstrap, leaves, seed average, context features | Several successive promotions | Benchmark registry and history |
| Later feature batches | Completed; current model retained | magnitude_active/OFFLINE_RESULTS.txt |
| Alternative models and shorter windows | Completed; no promotion | model_trial_run and window_trial_run |
| Timing baselines, audit and pilot | Completed within their stated scopes | Timing root receipts |
| Runtime engineering | Mixed synthetic and reported cloud evidence | Exact setup receipts and research_progress.json |
| Original exhaustive 15m | Partial / paused | User-reported flow completion; checkpoints remain external |
| Ten-horizon staged | Completed, mixed performance | Results/staged |
| Long-horizon alternatives | Planned, not run | CONTINUE_RESEARCH.md |

Read [magnitude results](../reports/magnitude_results.md) and [timing results](../reports/timing_results.md) for the numerical story. Historical receipts can refer to intentionally pruned models or predictions. Original source paths in those receipts identify the original run, not guaranteed current files. The [artifact manifest](../artifacts/manifest.json) records current files separately.
