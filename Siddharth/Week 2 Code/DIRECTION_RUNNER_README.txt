48-HOUR DIRECTION TRAINING RUNNER

From PowerShell:
cd "C:\Users\Siddh\Desktop\ML\November\Week 2"
python .\Code\fit_direction_v45_style.py

Dependencies: numpy, pandas, scipy, scikit-learn, torch, joblib,
threadpoolctl, and duckdb (or a working pandas Parquet engine).

This trains a NEW model using the v45 approach. It does not try to match
archived predictions and does not require the missing shared.py or v26 weights.
It uses the original Week 1 package X_train/y_train and development tables,
the same source package used by magnitude research. The supplied v45 pipeline
provides column names only; fitted preprocessing and weights are newly trained.

Default work: three chronological validation periods and one final training
period, each with one logistic fit, three initial neural fits (32 epochs), and
six fine-tuning fits (16 epochs). Total: 40 fits. All run on CPU with four threads.
Fine-tuning follows the supplied loss, two branches, three seeds and 75/25 blend.
The fixed 384-feature list was selected historically; these validation scores
do not measure a fresh nested feature-selection procedure.

Missing original implementation details are explicit choices in CONTRACT.json:
ReLU, initial LR .001, initial residual penalty coefficient 1, median imputation,
mean/std scaling, clipping at +/-8, blocked missing flags, standardized raw-product
interaction and sklearn logistic fitting. Historical scores are not guaranteed.
The development set was already studied. No model/branch is selected on it.

Outputs: Code/experiments/direction_v45_style_v1/<UTC timestamp>/
  CONTRACT.json                 Exact settings, inputs, hashes and versions
  metrics.csv                   Log loss, Brier, AUC, accuracy and probability means
  validation_predictions.csv    Purged validation probabilities and labels
  development_predictions.csv   Development p_up, p_down and comparisons
  fold_1/, fold_2/, fold_3/      Fitted validation models and training loss histories
  final/                        Fitted preprocessing, logistic model and neural weights
  COMPLETE.json                 Written only after successful completion

Lower log loss and Brier are better. p_down = 1-p_up; zero returns are assigned
to the non-UP class, matching the package's direction_up = (return > 0) target.
Final training always runs; it is not gated by the historical v45 acceptance rule.
Failed runs remain in their own timestamped folders and are not marked complete.

To predict another Parquet containing the same feature columns:
python .\Code\fit_direction_v45_style.py --model-run "FULL_PATH_TO_COMPLETED_RUN" --predict-input "FULL_PATH_TO_FEATURES.parquet"

Status at delivery: source inspected only. No Python execution, training,
prediction, syntax execution or real-data testing performed by the assistant.
