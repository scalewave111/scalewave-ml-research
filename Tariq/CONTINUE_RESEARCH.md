# Continue the research

**Start with the longer-horizon time-to-high results. Do not restart completed screens.**

| Order | Next action | Why |
| --- | --- | --- |
| 1 | Read the staged policy results and separate individual-feature rankings | The chosen tree policy lost at longer horizons even when some cheap models looked better |
| 2 | Compare inner ranks, shortlisted features, tree settings and outer errors by fold | Find where potentially useful features were lost, without using outer results to rewrite past selections |
| 3 | Prepare a focused follow-up contract | Choose a small set of alternative models and explicit feature transformations, especially for 300h |
| 4 | Define chronology, baseline, population, fit budget and stopping rule before fitting | Keep comparisons fair and prevent repeated unrecorded searching |
| 5 | Run only after the next experiment is separately authorized | No follow-up training was part of this handoff |
| Later | Consider combinations, time_to_low_min and other targets | Each needs its own baseline, contract and evaluation |

## Work already completed

Do not rerun the ten-horizon baselines, sampled coverage audit, 315-fit pilot, or completed ten-horizon staged screen to recreate documentation. Use saved evidence. The old exhaustive 15m screen was paused after the flow family according to user logs; preserve its existing checkpoints and never call it complete.

The older research_progress.json points toward time-to-low and contains historical deadline/authorization fields. Those fields are preserved as history. This guide records the user's later priority: investigate longer horizons first.

## Rules for the next experiment

- Forecast every 30 minutes. Use the approved source months only: April 2022-June 2024; skip every July 2024 and later partition.
- Origins finish before 2024-07-01T00:00:00Z. Outcomes finish strictly before 2024-07-24T00:01:00Z.
- Use the five chronological outer periods and the preceding three-month inner validation, with strict label-end purges.
- Fit preprocessing on training data only. Compare identical evaluation rows; missing feature values must not silently remove rows.
- Treat currently identified candidates as development leads. Their outer results have already been inspected.
- Multiple specialist models are allowed. A downstream model must receive chronological out-of-sample upstream predictions during training and evaluation. Connected models have not been implemented.
- Keep compact scores and decisions for failed attempts. Do not accumulate losing binaries or duplicate archives.

See [timing results](reports/timing_results.md), [experiment history](docs/experiment_history.md), and [reproducibility](docs/reproducibility.md).
