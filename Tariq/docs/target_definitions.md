# Targets and units

**A feature lookback describes the past. A forecast horizon describes the future. The prediction is the model's answer; the error measures how far that answer is from reality.**

Example: a feature named 5m may describe the previous five minutes. It can be used to predict time-to-high over a future 48-hour horizon. A prediction of 900 minutes and an observed value of 840 minutes give an absolute error of 60 minutes, or 2.0833% of that horizon.

| Target | Meaning | Status |
| --- | --- | --- |
| Absolute endpoint magnitude | 10,000 × abs(log(P at horizon / starting P)) | Completed 48h magnitude work; stored column magnitude_log_bps |
| time_to_high_min | Minutes to the earliest maximum over the inclusive future horizon, including minute zero | Current timing research |
| time_to_low_min | Minutes to the earliest minimum over that same interval | Next target after focused high-timing follow-up |
| direction | Direction of the endpoint change | Planned in timing program; earlier frozen direction context is separate |
| net_magnitude_log_bps | Endpoint-magnitude target in the wider target list | Planned; confirm canonical signed/absolute convention before implementation |
| up_magnitude_log_bps | Upward excursion from the starting price to the future maximum | Planned |
| down_magnitude_log_bps | Downward excursion from the starting price to the future minimum | Planned; confirm canonical sign convention |
| extreme_order | Which future extreme occurs first | Planned; confirm tie/encoding convention |
| signed_return_log_bps | 10,000 × log(endpoint P / starting P) | Signed endpoint return |
| signed_return_simple_bps | 10,000 × (endpoint P / starting P - 1) | Same outcome as signed log return, in different units |

Excursions start from the origin price. They are different from the absolute endpoint target. The 120h candidate whose name contains 5m__15m is one exact supplied feature name; it is not two separate experiments.

Timing horizons: 15m, 1h, 4h, 8h, 16h, 48h, 72h, 120h, 200h and 300h. Primary error: MAE in minutes. Secondary error: 100 × MAE / horizon minutes. Improvement percentage is a different number: 100 × (baseline MAE - candidate MAE) / baseline MAE.

Future target conventions must be checked against the canonical source definitions before implementation. Do not infer missing encodings from similar names. See the [target/horizon matrix](../results/target_horizon_matrix.csv).
