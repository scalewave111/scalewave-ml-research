# Magnitude results

**The current model reduces average error by 1.238459 log bps (0.6803%) from the original frozen model.** This is a modest improvement on development data.

| Milestone | MAE (log bps) | Spearman IC |
| --- | --- | --- |
| frozen_MP0004 | 182.03697721076213 | 0.03025442548026061 |
| walkforward_refitted_pair | 181.97660210071422 | 0.027241073957620092 |
| selected_pair_RF_hybrid | 181.5238678328609 | 0.07737672515285152 |
| rf_depth_leaf_tuning | 181.13846791233013 | 0.06814772435805158 |
| tuned_pdf_flow | 181.06650820557817 | 0.0700806284070547 |
| tuned_flow_plus_venue_fragmentation_1h_temporal_concentration_1h | 181.0610241130761 | 0.07195947735301861 |
| rf_bootstrap_size | 180.89120445756495 | 0.07347771657743142 |
| rf_bootstrap_16384_extension | 180.87738581427703 | 0.07583203436248488 |
| rf_leaf_after_bootstrap | 180.86314075140652 | 0.07623424191635585 |
| rf_three_seed_average | 180.82053308956776 | 0.07690384610046537 |
| ensemble_context_flow_flow_activity_percentile__15m | 180.81520672728772 | 0.07795173134982929 |
| ensemble_context_flow_flow_activity_percentile__5m | 180.79851809782613 | 0.07852651265983954 |

All milestone scores refer to the saved 18,481-row development benchmark. The current model blends a refitted liquidation pair with three equally weighted random forests (128 trees per seed, seeds 48/17/91). Seven raw features feed the forest, with training-only preprocessing and missing-value indicators. Earlier inner validation chooses the blend weight per period.

## What did not improve

Feature additions, linear additions, alternate blending, recency weighting, deeper trees and boosting attempts often failed to improve the incumbent at that time. Do not compare every rejected attempt only with the original frozen score: the incumbent changed during the search.

Later alternative-model scores:

| Candidate | MAE |
| --- | --- |
| incumbent | 180.79851809782613 |
| hgb:standalone | 190.930859320845 |
| hgb:blend | 182.43871249699856 |
| nn:standalone | 182.87995361193103 |
| nn:blend | 181.98189981448135 |

Later fixed-window RF scores:

| Training window | MAE |
| --- | --- |
| 6 months | 184.40489754309007 |
| 12 months | 182.79163056619444 |
| 18 months | 182.43698737466912 |

None of those later alternatives beat 180.79851809782613. Their saved receipts survive; losing binaries are not included. See the [full registry](../results/experiment_registry.csv), [history](../docs/experiment_history.md), and [current configuration](../research/magnitude_48h/playground/submissions/magnitude_current/run_001/config.json).

Magnitude inference can be replayed with retained model files and approved prepared inputs. It does not need retraining. Repeated development selection means this is not untouched-test evidence, statistical significance, profitability or global optimality.
