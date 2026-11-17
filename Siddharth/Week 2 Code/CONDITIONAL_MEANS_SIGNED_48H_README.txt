Purpose
-------
Train new conditional-mean models, then calculate a 48-hour signed-return forecast.
The previous MAE models are unchanged. This script performs NEW TRAINING when run.

Run from November/Week 2:
    python .\Code\fit_conditional_means_signed_48h.py

Models
------
UP: GRU with 48 half-hour snapshots, eight features and eight missing flags,
16 hidden units, positive output, and squared-error loss. Initialize at the UP
training mean. Select 0-24 epochs using earlier validation MSE (patience 5).
DOWN: Ridge alpha=1 plus squared-error Random Forest (128 trees, depth 3,
leaf minimum 128, bootstrap sample 4096, seed 48). Choose RF blend weight
0/.1/.25/.5 by row-weighted MSE over three earlier validation windows.
Both use the existing eight magnitude features; no feature search.

Four expanding-history fits: January, April, July and October 2024.
Training rows must have label_end strictly before the relevant fit cutoff.
UP training uses positive historical outcomes; DOWN uses negative outcomes.
Both fitted models predict EVERY eligible development timestamp.
Saved model replay is checked before each fold is accepted.

Target and formula
------------------
Convert stored signed log bps first: ordinary_bps = 10000*expm1(log_bps/10000).
The magnitude target is abs(ordinary_bps). Zero outcomes are excluded from both
side fits; zero development outcomes fail explicitly pending a flat policy.

The saved direction_v45_style_v1/20261112T045009318177Z probabilities are reused.
Direction is not retrained. Exact timestamp sets and outcome signs must agree.
Only float32-sized complement rounding is accepted; p_down is then 1-p_up.

signed_return_forecast_bps = p_up*up_mean_bps - p_down*down_mean_magnitude_bps
signed_return_forecast_pct = signed_return_forecast_bps/100

DOWN mean magnitude is positive. down_contribution_bps is NEGATIVE in this new
runner, so signed forecast equals the SUM of the two contributions.
A forecast_pct value of 1.5 means +1.5%, not a decimal fraction or log return.

Files
-----
Outputs go to a NEW timestamped folder under:
    Code/experiments/conditional_means_signed_48h_v1/

signed_return_predictions.csv: probabilities, both means, signed forecast,
                              ordinary percentages, actual outcomes, baselines.
signed_metrics.csv: RMSE, MAE, MSE, bias and rank IC, pooled and by period.
magnitude_metrics.csv: each side model versus its training-side average.
YYYY-MM_mean_models.joblib: fitted UP/DOWN models and saved preprocessing.
validation_scores.csv, FIT_AUDIT.json, CONTRACT.json, source_snapshot.py,
SUMMARY.txt and COMPLETE.json: selection, replay, settings and provenance.

Signed baselines: zero; expanding training-only average signed return; and
the SAME probabilities multiplied by training-only UP/DOWN mean magnitudes.
No model is automatically selected using development results.

Requirements and scope
----------------------
Uses Week 1 Code/experiments/eight_feature_exhaustive_v1/prepared.joblib and
the completed direction run above. Paths can be changed using --prepared,
--direction-run and --output-dir. Dependencies: numpy, pandas, scipy,
scikit-learn, torch, joblib and threadpoolctl.

The script is a historical training/evaluation runner, not a live feed service.
Its predict_means(bundle, x, keys) function supports label-free magnitude
inference from a saved bundle and correctly constructed timestamped feature
history. It does not obtain market data or direction probabilities for live use.

Squared-error training targets conditional means; it does not guarantee accurate
or calibrated means. These historical periods were already inspected. Scores
are exploratory; no uncertainty analysis, fresh confirmation or trading policy
is included. No source datasets or old results are overwritten.

Delivery status: source reviewed only; not executed or runtime-tested by assistant.
