# 48-hour magnitude research

We predict the size of the endpoint price change after 48 hours, ignoring whether price rose or fell. This is not the maximum upward move within the period.

Read [results](../../reports/magnitude_results.md), [history](../../docs/experiment_history.md) and [replay instructions](../../docs/reproducibility.md).

The playground layout is preserved so paths continue to make sense. Current artifacts are in [magnitude_current](playground/submissions/magnitude_current/FINAL_HANDOFF.txt). Outer pair cards in magnitude_walkforward_v2 are needed for inference. Inner pair cards in magnitude_rf_hybrid_v4 support selection history. Original data is external.

Historical scripts can train models or change the incumbent when executed. They are supplied for inspection and future controlled reuse, not as a request to rerun completed work. The original playground README covers a broader team benchmark, including direction and old teammate instructions. Those instructions do not override the current repository guide.
