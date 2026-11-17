FOCUSED UP/DOWN CONDITIONAL MEAN MAGNITUDES

Run from Week 2:
  python .\Code\tune_ridge_mean_magnitudes_48h.py

Requires the existing tune_conditional_mean_magnitudes_48h.py helper alongside
this runner, the prepared.joblib data, and the completed controlled-ablation run
20261112T082233470776Z. Paths can be overridden with --prepared, --reference-run,
and --output-dir. CPU threads default to 2 (--threads).

UP: fixed core four inputs (15m and 5m historical liquidation activity,
4h/24h volatility ratio, and 1h liquidation venue breadth).
DOWN: those same four inputs plus 1h basis level.
Full source column names are saved in CONTRACT.json.

Ridge alpha = training rows times one of [0.0001,0.001,0.01,0.1,1,10].
Forecast = w * Ridge magnitude + (1-w) * training-only side average.
w is one of [0,0.25,0.5,0.75,1]. Zero is explicitly a baseline policy.
Two policies reported: tuned_ridge (w=1) and selected_blend (all weights).
Selection minimizes row-weighted MSE across three earlier validation windows.
Exact ties prefer lower Ridge weight, then stronger regularization.
No feature-mask or family selection is performed in this experiment.

Four quarterly expanding fits start Jan/Apr/Jul/Oct 2024. For every fit,
only labels ending strictly before its cutoff are used. Inner cutoffs are
9/6/3 months before the outer cutoff. Validation origins cover the first
two months after each inner cutoff; labels must mature before the following
quarter and before outer fitting. All preprocessing uses only fit-training rows.
144 inner Ridge fits; up to 16 final Ridge fits. Blend scoring reuses predictions.

Target: abs(10000 * (price_after_48h / price_at_forecast - 1)).
The input log-bps labels are converted observation by observation using expm1.
Both sides output NONNEGATIVE ordinary-bps MAGNITUDES, including DOWN.
These are conditional mean estimates from squared-error training, not medians,
excursions, or signed-return forecasts. No direction probabilities are loaded.
Both side predictions are supplied at every reference forecast time, without
choosing the forecast side from the realized outcome. Outcome side is used only
to define historical training labels and evaluate conditional errors.

Prediction bundles contain preprocessing, Ridge model, blend weight, and the
training-side average. predict_bundle(bundle, features) accepts features only.
Do not combine bundles across cutoffs or supply times earlier than their issue.
No automatic future refit or live feature-availability guarantee is provided.

Outputs: Code/experiments/focused_ridge_mean_48h_v1/<UTC run timestamp>/
  SUMMARY.txt                 concise pooled comparison
  magnitude_metrics.csv       pooled and per-period side errors
  magnitude_predictions.csv   both side forecasts on every reference origin
  validation_metrics.csv      individual earlier-window scores and cutoff audit
  validation_candidates.csv   pooled scores for every alpha/weight pair
  SELECTIONS.json             earlier-validation choices saved before outer scoring
  FIT_AUDIT.json              training cutoff, artifact hashes and replay status
  tail_errors.csv             errors above/below training-side P90 magnitude
  REFERENCE_COLUMNS.json      maps comparison columns to previous configurations
  CONTRACT.json              units, settings, source/input hashes, package versions
  *.joblib                   saved per-side/per-cutoff prediction bundles
  COMPLETE.json              written only after computation and replay checks pass

Reference configurations are copied unchanged for comparison, never selected by
their outer scores. UP comparisons include core4 Ridge, core4-minus-breadth Ridge,
and the earlier validation-selected MLP. DOWN comparisons include core4+basis
Ridge and full8-minus-1h-liquidation Ridge.

This reuses the exposed 18,481-origin historical evaluation population ending
January 22, 2025. The feature sets were proposed after inspecting this period.
It is NOT an untouched test or proof of a reliable improvement. No verified
untouched later dataset has been supplied to this runner. Freeze settings before
evaluating such data when available; do not choose another winner on its outcomes.
RMSE is primary; also inspect bias, period consistency and tail error.
The latest saved quarterly model is not automatically a current/live model.
