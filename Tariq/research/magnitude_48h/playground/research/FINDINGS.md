# Findings to build on

| Stage | Magnitude | Direction |
|---|---|---|
| M0 | Median forecast161.461839; MAE187.831072 | P(UP)0.5057909843; LL0.692022854 |
| Singles |256 fits;32 improved. Cross-asset liquidation activity15m:MAE182.284669 |256 fits;217 improved. Variance percentile4h:LL0.685546 |
| Addition |190 top20 pairs;108 beat parent. Activity15m+5m:182.036977 |23436 pairs;11941 beat parent. Basis persistence1h+variance4h:0.684958481 |
| Interactions |Room-specific candidates; none established a global upgrade |Best tested flow/variance product0.684953, but Brier worse than winning global additive pair; no full advancement |
| Trees |Compact tree184.0463; broad tuned tree188.6110; full256forest194.9815 |20feature tree0.707316, worse than M0 |

Magnitude recurring concepts: liquidation activity and fast/slow lift. Three standalone fields (two concepts) passed development checks, but best average single/pair did not. Saturation and narrow forecasts require level-versus-information controls.

Direction recurring concepts: variance, basis and flow. The frozen additive pair is the two-metric simple comparator; it is not the absolute lowest log-loss number. Probability improvements are modest and no candidate is confirmed.

Selected-room magnitude scores use different rows; never compare them directly with global182.04. Historical report appendices contain other targets for context only. This challenge has exactly two objectives.

All findings use exposed development history. Magnitude predicts a median absolute move; direction predicts P(UP). Expected signed return is not supplied. Source reports preserve successes, failed complexity upgrades and limitations. See OVERVIEW.html for the full walkthrough and top20 tables.
