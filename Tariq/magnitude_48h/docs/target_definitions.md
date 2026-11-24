# Magnitude target and units

The target is 10,000 * abs(log(endpoint price / starting price)) over 48 hours, stored as magnitude_log_bps. It measures the endpoint change, not the maximum excursion within the window.

The main metric is MAE in log basis points; lower is better. A feature lookback describes past information, while the horizon describes the future outcome. Do not compare these numbers directly with the separate 15-minute integration's simple-bps forecasts.
