# Timing results

**Small gains at shorter horizons; losses at longer horizons.** Lower MAE is better. Minutes are prediction error, not the feature lookback or the predicted time.

| Horizon | Best baseline MAE (min) | Selected policy MAE (min) | Improvement % |
| --- | --- | --- | --- |
| 15m | 4.887309 | 4.876254 | 0.2262 |
| 60m | 18.673993 | 18.599480 | 0.3990 |
| 240m | 72.948120 | 72.833273 | 0.1574 |
| 480m | 143.232936 | 143.221601 | 0.0079 |
| 960m | 285.145773 | 283.764495 | 0.4844 |
| 2880m | 846.782959 | 862.352839 | -1.8387 |
| 4320m | 1276.198214 | 1369.980314 | -7.3486 |
| 7200m | 2156.585095 | 2269.247511 | -5.2241 |
| 12000m | 3526.324387 | 3849.495566 | -9.1645 |
| 18000m | 5259.473988 | 5867.384327 | -11.5584 |

## What was completed

Three baselines across ten horizons; sampled F4Z coverage (522 registered, 386 usable candidates, 133 without usable sampled values, three constant); original 315-fit pilot; and the staged screen of all 386 features at each horizon.

The staged method uses 16 training-quantile bins with target medians, inner-only top-three shortlisting and six absolute-error tree settings. Five chronological outer folds each select one feature/settings pair. It is not exhaustive tree tuning and does not combine features.

## Individual cheap models tell a different story

| Horizon | Best individual feature | MAE minutes | Improvement % |
| --- | --- | --- | --- |
| 2880m | F4Z__flow__flow_activity_rz__1h__value | 840.7709478021978 | 0.7100 |
| 4320m | F4Z__flow__flow_activity_rz__5m__value | 1271.5990042217327 | 0.3604 |
| 7200m | F4Z__price__price_acceleration_percentile__5m__15m__value | 2148.8593289085547 | 0.3582 |
| 12000m | F4Z__funding__funding_change_percentile__1h__value | 3523.9444418647845 | 0.0675 |
| 18000m | F4Z__basis__basis_path_retention_percentile__5m__value | 5307.950787401574 | -0.9217 |

These are retrospective development candidates. The chosen tree policy lost at 48h-200h even though some cheap single-feature models beat baseline. At 300h no screened individual feature beat baseline. See [all rankings](../results/time_to_high/README.md).

## What verification means here

Saved receipt verification covered 38,600 cheap inner/outer fold scores, 50 selected outer folds, state checksums, frozen identities, selection arithmetic, reported purges and counts. It did not independently replay raw data or timing-model inference locally. Equal counts do not independently prove identical row identities.

The original exhaustive 15m experiment remains partial/paused, with flow completion reported by the user. The staged completion does not turn that different experiment into a completed one. Runtime setup receipts retain their original pending/synthetic scope.

Next: investigate missed longer-horizon gains and propose focused alternatives/transforms, especially for 300h. No follow-up training has started. Gains do not establish significance, profitability or global optimality.
