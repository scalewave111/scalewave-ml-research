# Project context

## Recovered stopping point — 2026-11-04

Source inspected: all six pages of `../Report/ScaleWave_Model_Results_Ranked.pdf`, dated 2026-11-03. This is a report-based recovery; underlying experiment outputs and code have not been independently revalidated in this session. No training was started.

Objective: build and compare models to beat the baseline for 48-hour absolute endpoint log-return magnitude. Main comparison: 18,481 evaluation rows, 2024-01-01 through 2025-01-22, ranked by test MAE in bps with Spearman IC also reported.

- Baseline RF + median regression remains best MAE: 181.526 bps, IC 0.07739. Saved reconstruction is 181.525993; original PDF baseline was 181.523868.
- No listed model achieved MAE below 180 bps.
- Residual correction achieved highest IC: 0.08465, with MAE 181.866.
- Latest experiment: original 5m and 15m liquidation historical-active-rate controls plus `F4Z__flow__flow_activity_rz__4h__value` and three missing-value flags. Eight RF/CatBoost/LightGBM configurations; 37 fits total. CatBoost was tested but not selected.
- Latest test standalone tree: MAE 184.198, IC 0.06204; hybrid: MAE 182.837, IC 0.05318. Hybrid validation MAE 177.466 versus baseline 178.630 did not translate into a test win.
- Latest experiment source paths named by report: `../Experiments/three_feature_tree_tuning_v1/{scores.csv,SELECTIONS.json,COMPLETE.json,flow_internal_importance.csv,AUDIT.json}` and `three_feature_tree_tuning.py`.
- Separate last-three-month comparison contains 3,937 rows and must not be mixed with the main ranking.
- The report does not specify a committed next experiment. Recover underlying outputs and applicable research methods before selecting or running further work.

## Workspace availability

On 2026-11-04, the instruction-referenced `PROJECT_CONTEXT.md`, `FILE_INDEX.csv`, `ml_framework/AGENTS.md`, `ml_framework/CONTEXT.md`, and `ml_framework/docs/INDEX.md` were absent from this Code directory. This context file was newly created to preserve the recovered stopping point. `refresh_file_index.py` was also absent from the directory listing, so the prescribed index refresh could not be run.

## LSTM comparison in progress — 2026-11-04

User authorized training LSTMs to beat the baseline using the same walk-forward strategy. `lstm_walk_forward.py` implements six fixed LSTM configurations, plus a validation-top-three ensemble and nonzero-LSTM blends with expanding median regression. The run uses the same four quarterly expanding origins, strict label-end purges, and 18,481 test rows as the report baseline. Training was launched on the actual dataset; completion and results must be checked in `experiments/lstm_walk_forward_v1/COMPLETE.json`, not inferred from this note. Log: `lstm_walk_forward_run.log`. Syntax and synthetic gap/causal-prefix checks passed. Checkpoints permit resumption without repeating completed fits. No test-driven retries are planned.

Correction to earlier workspace availability note: the parent research workspace is `C:/Users/Siddh/Desktop/ML`. Its `PROJECT_CONTEXT.md`, `FILE_INDEX.csv`, `AGENTS.md`, `refresh_file_index.py` and `ml_framework` documents were found and read on 2026-11-04. They were absent only from this Code subdirectory. The report's November experiment files, rather than older October summaries, define this comparison. New artifacts are saved inside Code to respect the current writable workspace.

## LSTM result — executed and audited, 2026-11-04

The requested LSTM comparison finished on real data: 24 inner tuning fits and 8 outer refits, all using the baseline's four quarterly expanding windows, strict 48-hour label-end purges, and the same 18,481 evaluation rows. Audit passed, including baseline reproduction, training-only preprocessing, causal sequences, and checkpoint replay.

- Baseline test MAE 181.525993 bps; Spearman IC 0.077390.
- Standalone LSTM test MAE 183.682723 bps; IC 0.088500.
- LSTM + median-regression hybrid test MAE 183.593437 bps; IC 0.079617.
- The LSTM did NOT beat the full-period baseline MAE. Validation gains did not carry over to test. No additional test-driven tuning was performed.
- Sources: `experiments/lstm_walk_forward_v1/scores.csv`, `COMPLETE.json`, `AUDIT.json`, `CONTRACT.json`, `SELECTIONS.json`, `MANIFEST.json`, `predictions.csv`, `RESULTS.txt`, and `UNCERTAINTY.json`.
- Results notebook: `BTC_48h_LSTM_WalkForward.ipynb`. Runner: `lstm_walk_forward.py`; replay with `python lstm_walk_forward.py audit`.
- Original training source is preserved in the run directory. After training, the replay check was corrected to compare float32 CSV values after conversion back to float32, requiring exact equality; float64 values use 1e-10 absolute tolerance. Maximum CSV decimal difference was 0.000014893 bps. Models, selections, and predictions were unchanged. COMPLETE.json records separate training and audit source hashes.
- Historical evaluation was previously exposed, not a fresh holdout. The local file index covers this Code directory; the broader parent inventory remains a separate snapshot.

## Feature-first LSTM experiment in progress — 2026-11-04

The user stopped the larger optimization sweep before it started training. `lstm_optimize.py` is a prepared, unexecuted draft, not a completed experiment.

The user then authorized a narrower roadmap: select contributing features first, freeze those choices, then test LSTM architectures and overfitting controls. The computation budget is 15 minutes. Runner: `lstm_feature_architecture.py`; output: `experiments/lstm_feature_architecture_v3/`; log: `lstm_feature_architecture_run.log`.

The new run uses the previous 17-input LSTM checkpoints to rank feature contributions by three 7-day-block permutations on the first two months of each prior quarter. It compares top 2/4/8 features and all 17 as a control. Architecture selection uses a separate final month, with labels purged at each boundary. All four windows finish feature selection before any architecture tuning begins. Six designs compare 1/2/3 LSTM layers, 4/8/16 hidden units, and an optional dense layer. AdamW, weight decay excluding biases, dropout, gradient clipping, warmup/cosine learning rate and early stopping are applied. Outer windows and evaluation rows remain unchanged. One seed is used to meet the time cap.

Syntax and forward/backward checks passed; actual feature-selection training started. Inspect `COMPLETE.json`/`AUDIT.json` or `STATUS.json` before assuming the run finished. No results from this run are yet recorded in this context entry.


## Two-input MLP experiments — executed and audited, 2026-11-04

User requested a small MLP using only the original 5m and 15m liquidation historical-active-rate features, preserving the walk-forward comparison. User explicitly removed the time constraint and requested IC as well as MAE. MAE remained the selection objective. Exactly two inputs were used, with training-only median imputation and scaling; no missingness flags or lagged/extra features. Three seeds were averaged.

- Baseline: test MAE 181.525993 bps; Spearman IC 0.077390.
- Standard MLP: test MAE 182.637298 bps; Spearman IC 0.050053.
- Initialized MLP: test MAE 182.416272 bps; Spearman IC 0.051098.
- Recency-weighted MLP: test MAE 183.707231 bps; Spearman IC 0.040363.

All three runs completed on real data and passed strict label-purge, feature-count, training-only preprocessing, baseline reconstruction, and checkpoint replay checks. The recency experiment changes training-example weights only, retaining every expanding-window row; dates are not network inputs. Model selection uses only each preceding purged validation quarter. Subsequent experiments and the evaluation period are exploratory, not an untouched holdout.

Sources: `experiments/mlp_two_liquidation_comparison/RESULTS.txt`, `scores.csv`, `window_scores.csv`, `selected_architectures.csv`; individual runs `experiments/mlp_two_liquidation_v1/`, `experiments/mlp_two_liquidation_warmstart_v2/`, `experiments/mlp_two_liquidation_recency_v3/`. Runners share those names without the version suffix. Results viewer: `BTC_48h_MLP_Two_Feature_Comparison.ipynb`. Each folder contains COMPLETE.json, AUDIT.json, contract, source snapshot, predictions and models.

Correction to earlier in-progress LSTM note: `experiments/lstm_feature_architecture_v3/COMPLETE.json` records a completed audited run, with MAE 184.177232 versus baseline 181.525993; it did not win. `diagnose_lstm_features.py` was generated, but a completed diagnostics_v1 output was not observed before the user switched to MLP work. The previously stopped `lstm_optimize.py` remains an unexecuted draft.

## MLP hyperparameter tuning in progress — 2026-11-04

User explicitly authorized hyperparameter tuning and continuation. Runner `mlp_hyperparameter_tuning.py`, output `experiments/mlp_hyperparameter_tuning_v4/`. Training started after syntax and all 24 configurations' two-input forward/backward checks passed. Exactly the original two liquidation inputs remain; outer quarterly walk-forward windows and evaluation rows are unchanged. Three preceding purged monthly inner folds screen 24 fixed configurations with seed 48. The top three are confirmed with seeds 149 and 250, then the selected setting is refitted with all three seeds. MAE selects settings; Spearman and Pearson IC are reported. Learning rate, decay, dropout, batch size, activation, small architecture, and initialization vary. Planned checkpoints: 372. No wall-clock limit. Check STATUS.json and COMPLETE.json before assuming completion; individual checkpoints support resumption. Historical evaluation has already been examined.


## MLP hyperparameter tuning result — 2026-11-04

Executed on real data and audited: 372 model checkpoints, 24 settings, three purged monthly inner folds per outer window, three seeds for shortlisted settings and outer refits. Exactly two original liquidation inputs; unchanged four outer windows and 18,481 test rows.

- Baseline MAE 181.525993 bps; Spearman IC 0.077390.
- Tuned MLP MAE 182.608652 bps; Spearman IC 0.037884.
- Beats baseline MAE: False.

Sources: `experiments/mlp_hyperparameter_tuning_v4/COMPLETE.json`, `AUDIT.json`, `RESULTS.txt`, `comparison.csv`, `selected_hyperparameters.csv`, and `window_scores.csv`; results viewer `BTC_48h_MLP_Hyperparameter_Tuning.ipynb`. All purges, preprocessing, frozen selection, baseline reconstruction, and checkpoint replay checks passed. Earlier runs remain unchanged. Previously examined historical evaluation remains exploratory, not an untouched holdout.

## Guarded MLP baseline correction in progress — 2026-11-04

User authorized the suggested correction experiment. Runner `mlp_baseline_correction.py`; output `experiments/mlp_baseline_correction_v5/`. Syntax, two-input signed correction, zero initialization, gradients, and the validation eligibility gate passed initial checks. Actual historical baseline refits have started. Correction targets use chronological held-out baseline predictions from January 2023 onward, with earlier data reserved for baseline warmup. Six small networks, three seeds, three monthly validation folds, and correction strengths including zero are predeclared. Nonzero corrections require pooled gain >=0.1 bps, improvements in >=2 months and >=2 seeds, and worst monthly degradation <=1 bps. Original two source features and outer walk-forward windows remain unchanged. Inspect COMPLETE.json/AUDIT.json before assuming completion. No time cap; checkpoints allow resumption.


## Guarded MLP baseline correction result — 2026-11-04

User authorized the proposed baseline-error correction using only the original two liquidation inputs. Executed on real data and audited. Baseline chronological OOF predictions start January 2023, using older data for baseline warmup; correction training never uses in-sample baseline residuals. The baseline itself retains all original expanding training data. Three earlier monthly folds and three seeds select among six tiny correction networks and strengths including zero. Nonzero selection requires pooled gain >=0.1 bps, gains in >=2 months and >=2 seeds, and worst monthly degradation <=1 bps. Outer windows and 18,481 evaluation rows are unchanged.

- Baseline MAE 181.525993 bps; Spearman IC 0.077390.
- Corrected MAE 184.111883 bps; Spearman IC 0.013586.
- Beats baseline MAE: False; active correction windows: 2/4.

Sources: `experiments/mlp_baseline_correction_v5/COMPLETE.json`, `AUDIT.json`, `RESULTS.txt`, `comparison.csv`, `selected_corrections.csv`, `BASELINE_SELECTIONS.json`, `baseline_oof_predictions.csv`, and `window_scores.csv`. Runner: `mlp_baseline_correction.py`. Results viewer: `BTC_48h_MLP_Baseline_Correction.ipynb`. Baseline OOF replay, original baseline reconstruction, residual provenance, label purges, preprocessing, eligibility selection and correction replay checks passed. Prior results and raw data remain unchanged. Historical evaluation remains exploratory, not untouched holdout evidence.

## Baseline plus one feature in progress — 2026-11-04

User authorized the proposed controlled feature-addition experiment, relaxing the earlier two-feature-only restriction for this run. Runner `baseline_single_feature.py`; output `experiments/baseline_single_feature_v6/`. Both original baseline components retain their model parameters and preprocessing recipe; each challenger adds exactly one source feature to the original liquidation pair. Six candidates: 4h/24h/7d realized volatility, past 48h absolute move, 4h/24h volatility ratio, and funding persistence. Same quarterly outer windows, expanding training, purges and blend-weight rule. Primary strategy selects on validation with >=0.1 bps gain and wins in >=2 months, otherwise baseline; all six fixed-feature strategies are separately preregistered. Planned 96 fits. Syntax, synthetic causal-feature/blend/selection checks passed; actual validation fits have started. Check COMPLETE.json/AUDIT.json before assuming completion. No test-driven retries or time cap; historical evaluation remains exposed.


## Baseline plus one feature result — 2026-11-04

User authorized controlled feature addition to beat the baseline, relaxing the earlier two-input-only restriction for this experiment. Exactly one source feature was added to the original pair at a time: 4h/24h/7d realized volatility, past 48h absolute move, 4h/24h volatility ratio, or funding persistence. BOTH original baseline components were extended while retaining model parameters, preprocessing, blend-weight grid, training rows, and outer walk-forward splits. Primary selection used prior-quarter MAE and monthly consistency, including baseline fallback. All six fixed-feature strategies were preregistered and scored separately.

- Original baseline: MAE 181.525993 bps; Spearman IC 0.077390.
- Validation-selected feature strategy: MAE 182.035736 bps; Spearman IC 0.060956.
- + 4h realized volatility: MAE 181.957126 bps; Spearman IC 0.088050.
- + 24h realized volatility: MAE 183.030460 bps; Spearman IC 0.048860.
- + 7d realized volatility: MAE 181.462941 bps; Spearman IC 0.081046.
- + Past 48h absolute move: MAE 183.152795 bps; Spearman IC -0.008639.
- + 4h/24h volatility ratio: MAE 181.248134 bps; Spearman IC 0.106801.
- + Funding persistence: MAE 182.028282 bps; Spearman IC 0.044452.

Primary strategy beats baseline: False. Fixed-feature point-estimate wins: ['rv7d', 'vol_ratio']. These are historical exploratory results, not untouched holdout confirmation.

Executed on real data and audited: 96 fits; same 18,481 evaluation rows; label purges, training-only preprocessing, unchanged parameters, exact feature counts, baseline reconstruction, validation selection and saved-model replay checks passed. Canonical return source reproduced original labels; causal prefix, future-perturbation and manual past-window checks passed.

Sources: `experiments/baseline_single_feature_v6/COMPLETE.json`, `AUDIT.json`, `FEATURE_AUDIT.json`, `RESULTS.txt`, `comparison.csv`, `selected_features.csv`, `all_validation_trials.csv`, `uncertainty.csv`, and `window_scores.csv`. Runner: `baseline_single_feature.py`; results viewer: `BTC_48h_Baseline_One_Feature.ipynb`. Prior results and raw data remain unchanged.


## Volatility-ratio recipe frozen; fresh confirmation unavailable — 2026-11-04

User authorized freezing the fixed volatility-ratio candidate and comparing it with the original baseline on untouched data. The user confirmed that no newer dataset is available, then asked what could be done without additional data. The existing feature/evaluation origins end 2025-01-22 00:00 UTC, and their latest target end is 2025-01-24 00:00 UTC. Current canonical returns only cover that old evaluation. No fresh confirmation was run and no models were trained.

The exact existing baseline and fixed-volatility-ratio procedures, including the original predefined blend rule, were frozen. Archived October 2024 model copies and hashes are preserved as reference checkpoints. Historical month/window, exclude-one-window and 7/14/28-day paired-block-bootstrap diagnostics were computed from saved predictions; they do not constitute untouched evidence.

Historical MAE improvement persists in 3/4 original windows and 10/13 months. Improvement survives every individual window exclusion: True.

Sources: `experiments/vol_ratio_frozen_review_v1/FROZEN_PROTOCOL.json`, `COMPLETE.json`, `RESULTS.txt`, `window_checks.csv`, `monthly_checks.csv`, `leave_one_window_out.csv`, and `block_bootstrap_checks.csv`. Script: `freeze_and_review_vol_ratio.py`. Future confirmation requires newer matched liquidation features and price/return data; do not relabel old data as untouched.


## Volatility-ratio train/test metrics — 2026-11-04

User requested training and test MAE/IC for the frozen three-feature candidate. Calculated in-sample predictions using the saved four outer models and their original validation-selected blend weights; no retraining or model changes.

- training_in_sample: MAE 184.664338 bps; Spearman IC 0.228505.
- validation: MAE 178.444962 bps; Spearman IC 0.044075.
- test: MAE 181.248134 bps; Spearman IC 0.106801.

Pooled training has 144140 prediction rows over 42575 unique origins; dates repeat across expanding training windows. Sources: `experiments/vol_ratio_train_test_review_v1/scores.csv`, `training_window_scores.csv`, `AUDIT.json`, and `RESULTS.txt`. Runner: `report_vol_ratio_train_test.py`. Test remains the already examined historical walk-forward comparison.

## Volatility-ratio diagnostics and calibration — 2026-11-04

User authorized prediction-group checks, extreme-move diagnostics, and calibration with the same three features and walk-forward splits. Executed and audited on the original 18,481 saved real-data predictions; no base models retrained. Prior-quarter validation cutoffs and calibration parameters are strictly label-purged. Primary fixed half-offset calibration and two secondary fixed full-offset/scale-only procedures were declared before computation; no test-based choice among calibration methods.

Uncalibrated candidate MAE/IC: 181.248134 / 0.106801. Half-offset: 181.538343 / 0.069840; full-offset: 182.895209 / 0.050384; scale-only: 183.575083 / 0.038657. Baseline: 181.525993 / 0.077390. Keep the uncalibrated candidate. Within-window IC is unchanged by these monotone transformations; pooled IC changes because each window has different calibration parameters.

High prediction groups have greater mean actual movement than low groups in all four windows; complete low/medium/high ordering holds in three. Pooled extreme-event frequencies barely separate, although high exceeds low within each window. Outcome-selected extremes remain heavily underestimated, which alone does not diagnose miscalibration of a median/MAE forecast. Weekly block-bootstrap MAE-difference interval versus baseline is [-0.554731, +0.002476] bps, descriptive/unadjusted; prior near-zero upper endpoints vary with resampling seed. No untouched confirmation.

Sources: `experiments/vol_ratio_calibration_diagnostics_v1/INTERPRETATION.txt`, `RESULTS.txt`, `scores.csv`, `window_scores.csv`, `prediction_groups.csv`, `group_window_scores.csv`, `extreme_scores.csv`, `extreme_window_scores.csv`, `uncertainty.csv`, `CALIBRATION_PARAMETERS.json`, `CONTRACT.json`, `MANIFEST.json`, `AUDIT.json`, `COMPLETE.json`. Runner: `vol_ratio_calibration_diagnostics.py`; generated results viewer: `BTC_48h_Vol_Ratio_Calibration.ipynb`. A future single-feature absolute-volatility addition alongside the ratio is a proposed hypothesis only, not executed or established effective. Existing parent framework documents remain outside the writable root; this local entry links the detailed run evidence.

## Volatility ratio plus absolute level in progress — 2026-11-04

User authorized the proposed additional-feature test. One fixed challenger adds `PRICE__realized_vol_24h` to the liquidation pair and `PRICE__vol_ratio_4h_24h`. Both original median-regression and RF components retain their parameters, preprocessing, expanding rows and four quarterly walk-forward splits. Prior-quarter purged validation chooses the original blend grid; no calibration or new feature search. Planned 16 model fits. Runner: `vol_ratio_level_experiment.py`; outputs: `experiments/vol_ratio_plus_level_v1/`. Inspect COMPLETE.json/AUDIT.json before assuming completion. Training has started on the cached, previously causally audited real feature matrix. Historical evaluation remains exposed.


## Volatility ratio plus absolute level result — 2026-11-04

Four-feature experiment completed and audited on the original historical evaluation.
Features: original 5m and 15m liquidation historical-active-rate inputs, 4h/24h volatility ratio, and 24h realized volatility.
Models: original median regression and depth-3 random forest blend; same parameters, expanding training and prior-quarter blend selection. No calibration.

baseline: test MAE 181.525993 bps; Spearman IC 0.077390; Pearson IC 0.043871.
vol_ratio: test MAE 181.248134 bps; Spearman IC 0.106801; Pearson IC 0.067386.
ratio_plus_level: test MAE 182.645216 bps; Spearman IC 0.067057; Pearson IC 0.028540.
Four-feature MAE minus baseline: +1.119223 bps; wins in 1/4 windows.
Four-feature MAE minus vol_ratio: +1.397082 bps; wins in 1/4 windows.

This is an exploratory comparison on already-examined history. Weekly block-bootstrap intervals are descriptive and unadjusted for prior searches. Training scores pool overlapping expanding windows. No fresh confirmation or deployment.

Sources: `experiments/vol_ratio_plus_level_v1/SUMMARY.txt`, `RESULTS.txt`, `scores.csv`, `window_scores.csv`, `extreme_scores.csv`, `uncertainty.csv`, `AUDIT.json`, `COMPLETE.json`, `CONTRACT.json`, `MANIFEST.json`, `SELECTIONS.json`, and saved model checkpoints. Runner: `vol_ratio_level_experiment.py`. Viewer: `BTC_48h_Vol_Ratio_Plus_Level.ipynb`.

Decision: retain the uncalibrated three-feature candidate; the four-feature challenger worsened both historical test MAE and IC. Extreme-move MAE (>500 bps) increased from 508.356770 to 517.522966 bps. Independent `FEATURE_RECHECK.json` reproduced both cached volatility columns exactly from canonical returns and passed causal-prefix invariance. No broader feature search or further model tuning was run.

## Neural snapshot versus sequence comparison — started 2026-11-04

User clarified that the objective is a neural-network replacement, stopped the proposed two-versus-three-input MLP run before training, then authorized a residual MLP versus small PatchTST-style comparison. Runner: `neural_snapshot_vs_sequence.py`; output: `experiments/neural_snapshot_vs_sequence_v1/`. Both use exactly the liquidation pair plus volatility ratio. Residual MLP: 8/4 hidden units with current-value linear skip. Custom patch transformer: 48 half-hour snapshots, patches of 8, shared channel encoder, one layer, dimension16, two heads, feedforward32, current-value skip. It is not a full reproduction of the published PatchTST model. Fixed architecture/settings, three seeds, AdamW/MAE, dropout/weight decay, and prior-quarter validation-only early stopping followed by expanding outer refits. Same four outer windows and 18,481 rows; 48 checkpoints planned. No GARCH or architecture search. Inspect COMPLETE.json and AUDIT.json before claiming execution finished. Historical test data remains previously examined.


## Neural snapshot versus sequence result — 2026-11-04

Executed and audited: residual MLP versus small custom PatchTST-style transformer, both with liquidation 5m/15m historical active rates and 4h/24h volatility ratio.
Same four purged expanding walk-forward windows, three seeds, validation-only early stopping; 48 checkpoints. MLP uses current values; transformer uses 48 half-hour observations ending at prediction origin.

baseline: test MAE 181.525993 bps; Spearman IC 0.077390; Pearson IC 0.043871.
vol_ratio: test MAE 181.248134 bps; Spearman IC 0.106801; Pearson IC 0.067386.
residual_mlp: test MAE 182.497144 bps; Spearman IC 0.051730; Pearson IC 0.013140.
patch_transformer: test MAE 182.967117 bps; Spearman IC 0.028433; Pearson IC -0.015172.
residual_mlp minus baseline MAE: +0.971151 bps; wins in 2/4 windows.
residual_mlp minus vol_ratio MAE: +1.249010 bps; wins in 2/4 windows.
patch_transformer minus baseline MAE: +1.441124 bps; wins in 1/4 windows.
patch_transformer minus vol_ratio MAE: +1.718983 bps; wins in 0/4 windows.

Lowest neural test MAE in this experiment: residual_mlp. Beats original baseline MAE: False. Beats three-feature regression/RF candidate MAE: False.
These are exploratory historical comparisons, not untouched confirmation. Bootstrap intervals are descriptive and unadjusted for prior searches. Training scores are in-sample and pool overlapping expanding windows. The custom patch transformer is not a full PatchTST reproduction; the comparison changes architecture and capacity as well as history access. No GARCH model was trained.

Sources: `experiments/neural_snapshot_vs_sequence_v1/SUMMARY.txt`, `RESULTS.txt`, `scores.csv`, `window_scores.csv`, `seed_window_scores.csv`, `extreme_scores.csv`, `uncertainty.csv`, `SELECTIONS.json`, `CONTRACT.json`, `MANIFEST.json`, `PREFLIGHT.json`, `AUDIT.json`, `COMPLETE.json`, and saved `.pt` checkpoints. Runner: `neural_snapshot_vs_sequence.py`. Viewer: `BTC_48h_Neural_Snapshot_vs_Sequence.ipynb`.

Interpretation and decision: neither neural procedure beat either reference overall; retain the three-feature regression/RF candidate. In the first window all seeds selected epoch zero, so both procedures predicted the outer training median. Saved first-window IC zero is the project's constant-prediction convention (correlation is mathematically undefined), not learned ranking. MLP wins MAE in the final two windows; transformer wins against baseline only in the final window and never against the three-feature reference. Better selection-used validation scores did not transfer to test. See `experiments/neural_snapshot_vs_sequence_v1/INTERPRETATION.txt` for architecture limits, epoch-zero interpretation and extreme-move errors. No follow-up neural tuning was executed.

## Linear-initialized three-feature MLP — started 2026-11-04

User authorized the proposed fitted-linear initialization experiment. Runner: `linear_initialized_mlp.py`; output: `experiments/linear_initialized_mlp_v1/`. Training-only median regression initializes a frozen three-input skip; an 8/4 SiLU MLP adds a correction with zero-initialized output head. Nonnegative clipping replaces softplus to reproduce the linear forecast exactly at epoch zero. Exactly the liquidation pair plus volatility ratio; same preprocessing, seeds, purged quarterly windows, AdamW/MAE/dropout/weight decay and validation-only early stopping. Compare directly with the identical fitted linear model alone and both saved baselines. Fixed architecture and settings; 24 neural checkpoints plus five unique-cutoff linear fits planned, no test-driven search. Exact initialization and frozen-skip gradient checks passed. Check COMPLETE.json/AUDIT.json before assuming the real-data run finished. Historical evaluation is exposed.


## Linear-initialized three-feature MLP result — 2026-11-04

Executed and audited: training-only fitted linear skip plus zero-initialized 8/4 residual MLP.
Three inputs: liquidation 5m/15m historical active rates and 4h/24h volatility ratio. Same purged quarterly expanding walk-forward; three seeds. Frozen linear branch; nonlinear correction selected by earlier validation MAE.

baseline: test MAE 181.525993 bps; Spearman IC 0.077390; Pearson IC 0.043871.
vol_ratio: test MAE 181.248134 bps; Spearman IC 0.106801; Pearson IC 0.067386.
linear_only: test MAE 181.743611 bps; Spearman IC 0.061333; Pearson IC 0.065761.
initialized_mlp: test MAE 181.730526 bps; Spearman IC 0.061629; Pearson IC 0.066184.
Initialized MLP minus baseline: +0.204533 bps; wins in 2/4 windows.
Initialized MLP minus vol_ratio: +0.482392 bps; wins in 2/4 windows.
Initialized MLP minus linear_only: -0.013085 bps; wins in 3/4 windows.

Beats original baseline MAE: False. Beats three-feature regression/RF MAE: False.
No fresh holdout; historical comparisons are exploratory. Bootstrap intervals are descriptive and unadjusted for prior searches. Training scores pool overlapping expanding windows. The exact own-linear comparison tests added nonlinear correction; comparison with earlier softplus MLP also changes the output link and freezes the skip.

Sources: `experiments/linear_initialized_mlp_v1/SUMMARY.txt`, `RESULTS.txt`, `scores.csv`, `window_scores.csv`, `seed_window_scores.csv`, `extreme_scores.csv`, `uncertainty.csv`, `SELECTIONS.json`, `CONTRACT.json`, `MANIFEST.json`, `AUDIT.json`, `COMPLETE.json` and saved models. Runner: `linear_initialized_mlp.py`; viewer: `BTC_48h_Linear_Initialized_MLP.ipynb`.

Decision: retain three-feature regression/RF as best measured overall. Initialized MLP is the lowest neural MAE recorded here, but still fails to beat either reference. Its own-linear improvement is only 0.013085 bps, with descriptive weekly bootstrap difference interval [-0.085331, +0.062552], spanning zero. First-window constant prediction issue is resolved; no useful nonlinear advantage established. See `experiments/linear_initialized_mlp_v1/INTERPRETATION.txt`. No further tuning or deployment performed.

## Feature discovery for neural replacement — started 2026-11-04

User explicitly redirected work toward new contributing features, confirmed the final model must be the linear-initialized residual MLP, and authorized continuation. Runner: `mlp_feature_discovery.py`; output: `experiments/mlp_feature_discovery_v1/`. Screen the 520 original masked fields outside the two core liquidation inputs; retain volatility ratio as third core feature. Five training-quantile-bin residual corrections provide a cheap screening proxy, with per-fold >=80% train/validation coverage, >=10 unique values, and correlation redundancy checks. Residuals use saved inner three-feature MLP training predictions; screening validation is selection-used. Select at most one feature per window using >=.1 bps prior-quarter gain and wins in >=2 months; never test scores.

Then add that feature to the same 8/4 linear-initialized MLP, three seeds and unchanged training procedure. Actual MLP validation must separately pass the same gain/month consistency gate for the primary strategy to use the addition; otherwise preserve the three-feature model. Report the unguarded challenger separately. Up to24 neural checkpoints and8 linear fits; explicitly authorized research extends old package initial-fit budgets. No new dataset or fresh holdout. Inspect SCREEN_COMPLETE.json for completed screening and COMPLETE.json/AUDIT.json for finished MLP tests; do not infer completion from this note.


## MLP feature discovery result — 2026-11-04

Executed and audited: feature discovery followed by fixed MLP confirmation.
Screened 520 non-core original fields using earlier data only; selected at most one added feature per window. Existing three core features and 8/4 residual MLP settings retained.
Screening proxy uses regularized binned training residuals; actual MLP must separately pass prior-quarter MAE and monthly consistency gate. No test-driven feature selection.

baseline: test MAE 181.525993 bps; Spearman IC 0.077390; Pearson IC 0.043871.
vol_ratio: test MAE 181.248134 bps; Spearman IC 0.106801; Pearson IC 0.067386.
mlp_3features: test MAE 181.730526 bps; Spearman IC 0.061629; Pearson IC 0.066184.
challenger_mlp: test MAE 185.906856 bps; Spearman IC -0.032948; Pearson IC -0.067798.
selected_mlp: test MAE 185.906856 bps; Spearman IC -0.032948; Pearson IC -0.067798.

Per-window feature contribution:
                    issue                                                                          feature  screen_validation_gain  actual_mlp_validation_gain  validation_accepted  challenger_test_gain_vs_three_feature_mlp  primary_test_gain_vs_three_feature_mlp
2024-01-01 00:00:00+00:00                             F4Z__basis__basis_persistence_percentile__15m__value                1.942782                    4.897573                 True                                  -4.650250                               -4.650250
2024-04-01 00:00:00+00:00                             F4Z__basis__basis_persistence_percentile__15m__value                0.815882                    0.227096                 True                                  -0.152433                               -0.152433
2024-07-01 00:00:00+00:00                          F4Z__funding__funding_persistence_percentile__1h__value                1.793073                    4.561454                 True                                 -13.341481                              -13.341481
2024-10-01 00:00:00+00:00 F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__1h__value                1.331615                    2.590465                 True                                   0.417498                                0.417498

Primary uses the added feature in 4/4 windows. Beats original baseline MAE: False. Beats three-feature MLP: False. Beats three-feature regression/RF: False.
Broad multiple-feature search on already-examined history; no untouched confirmation. Selected features vary by window; pooled score is a selection strategy, not a single fixed feature set. Proxy gains alone do not establish neural benefit. Results do not establish that untested or differently combined features cannot help.

Sources: `experiments/mlp_feature_discovery_v1/SUMMARY.txt`, `RESULTS.txt`, `feature_contributions.csv`, `all_feature_screen.csv`, `shortlist.csv`, `selected_feature_definitions.csv`, `scores.csv`, `window_scores.csv`, `uncertainty.csv`, `SCREEN_COMPLETE.json`, `SELECTIONS.json`, `CONTRACT.json`, `MANIFEST.json`, `AUDIT.json`, `COMPLETE.json`. Runner: `mlp_feature_discovery.py`; viewer: `BTC_48h_MLP_Feature_Discovery.ipynb`.

Decision: reject this adaptive feature-addition strategy; validation gains failed to transfer overall. Basis persistence worsened test MAE in both assigned windows; funding persistence worsened its window by13.341481 bps. Only 1h liquidation historical active rate helped its assigned final window: MLP MAE167.146119 ->166.728621 (0.417498bps gain), Spearman0.059328 ->0.069589. This feature was NOT tested as a fixed addition across all four windows; it remains a one-window candidate, not a proven winner. Retain original three-feature MLP and best regression/RF reference. Detailed caveats: `experiments/mlp_feature_discovery_v1/INTERPRETATION.txt`. No further model fits or deployment performed.

## Fixed 1h liquidation fourth-feature MLP — started 2026-11-04

User authorized testing the 1h liquidation historical active rate as a fixed fourth input across all four windows. Runner: `mlp_fixed_liquidation_1h.py`; output: `experiments/mlp_fixed_liquidation_1h_v1/`. Same 8/4 linear-initialized MLP, three seeds, preprocessing, optimizer and validation-only early stopping. No feature selection, inclusion gate or fallback. Reuse final-window six neural checkpoints and July/October linear fits only after source-hash, feature-order, training-row and preprocessing checks. Planned new work:18 neural fits and3 unique linear fits. Report separate first-three-window scores to distinguish the already successful reused window. This is a post-selection historical retest, not independent confirmation. Inspect COMPLETE.json/AUDIT.json before assuming completion.


## Fixed 1h liquidation fourth-feature MLP result — 2026-11-04

Fixed fourth-feature MLP executed and audited. Added 1h liquidation historical active rate to the original 5m/15m rates and 4h/24h volatility ratio in every window.
Same linear-initialized 8/4 residual MLP and three seeds. 18 new neural fits plus six verified reused checkpoints; three new linear fits plus two reused. No feature gate or fallback.

baseline: test MAE 181.525993 bps; Spearman IC 0.077390; Pearson IC 0.043871.
vol_ratio: test MAE 181.248134 bps; Spearman IC 0.106801; Pearson IC 0.067386.
mlp_3features: test MAE 181.730526 bps; Spearman IC 0.061629; Pearson IC 0.066184.
linear_4features: test MAE 181.708124 bps; Spearman IC 0.061141; Pearson IC 0.064173.
fixed_four_feature_mlp: test MAE 181.615255 bps; Spearman IC 0.053954; Pearson IC 0.047770.
Four-feature MLP minus baseline MAE: +0.089262 bps; wins in 2/4 windows.
Four-feature MLP minus vol_ratio MAE: +0.367121 bps; wins in 2/4 windows.
Four-feature MLP minus mlp_3features MAE: -0.115271 bps; wins in 2/4 windows.
Four-feature MLP minus linear_4features MAE: -0.092869 bps; wins in 1/4 windows.

First three windows, excluding reused final window:
                 model        mae  rank_ic  pearson_ic  rows
              baseline 187.299604 0.115551    0.072577 13056
             vol_ratio 187.105257 0.128996    0.082927 13056
         mlp_3features 187.790606 0.065108    0.082575 13056
      linear_4features 187.795349 0.060078    0.069706 13056
fixed_four_feature_mlp 187.800916 0.059613    0.069396 13056

Feature chosen after observing its final-window success. This is a post-selection historical retest, not independent confirmation. Reused final-window results are not new evidence. Bootstrap intervals are descriptive and unadjusted for prior searches. Training scores pool repeated timestamps across expanding windows.

Sources: `experiments/mlp_fixed_liquidation_1h_v1/SUMMARY.txt`, `RESULTS.txt`, `scores.csv`, `window_scores.csv`, `first_three_windows_scores.csv`, `seed_window_scores.csv`, `extreme_scores.csv`, `uncertainty.csv`, `SELECTIONS.json`, `CONTRACT.json`, `MANIFEST.json`, `AUDIT.json`, `COMPLETE.json`. Runner: `mlp_fixed_liquidation_1h.py`; viewer: `BTC_48h_MLP_Fixed_Liquidation_1h.ipynb`.

Interpretation: lowest neural MAE so far is181.615255, but still0.089262bps worse than original baseline and IC worsens versus three-feature MLP. Excluding reused final window, four-feature MAE187.800916 versus three-feature187.790606, a0.010310bps deterioration; no aggregate benefit independent of the motivating window established. Preserve as a small/uncertain neural MAE improvement, not a validated replacement. Detailed caveats: `experiments/mlp_fixed_liquidation_1h_v1/INTERPRETATION.txt`. No additional tuning or deployment.

## Fixed 1h liquidation addition to best regression/RF — started 2026-11-04

User requested returning to the best RF-based result and using its conservative modeling procedure. Test the fixed1h liquidation historical active rate alongside the original5m/15m rates and4h/24h volatility ratio in BOTH median-regression and RF components. Retain exact original parameters, preprocessing with missingness flags, prior-quarter blend grid, strict label purges and expanding walk-forward rows. No hyperparameter search or calibration. Runner: `rf_fixed_liquidation_1h.py`; outputs: `experiments/rf_fixed_liquidation_1h_v1/`. Ten unique fits support16 window artifacts by reusing identical models at shared cutoffs. Syntax passed; inspect COMPLETE.json/AUDIT.json before claiming real-data completion. Feature and test history already examined, so this is exploratory rather than untouched confirmation.


## Fixed 1h liquidation regression/RF result — 2026-11-04

Fixed 1h liquidation feature experiment completed and audited: 10 unique models support 16 window artifacts.
Features: original 5m and 15m liquidation historical-active-rate inputs, 4h/24h volatility ratio, and 1h liquidation historical active rate.
Models: original median regression and depth-3 random forest blend; same parameters, expanding training and prior-quarter blend selection. No calibration.

baseline: test MAE 181.525993 bps; Spearman IC 0.077390; Pearson IC 0.043871.
vol_ratio: test MAE 181.248134 bps; Spearman IC 0.106801; Pearson IC 0.067386.
rf_plus_liq1h: test MAE 181.970900 bps; Spearman IC 0.044611; Pearson IC 0.016689.
Four-feature MAE minus baseline: +0.444907 bps; wins in 3/4 windows.
Four-feature MAE minus vol_ratio: +0.722766 bps; wins in 2/4 windows.

This is an exploratory comparison on already-examined history. Weekly block-bootstrap intervals are descriptive and unadjusted for prior searches. Training scores pool overlapping expanding windows. No fresh confirmation or deployment.

Sources: `experiments/rf_fixed_liquidation_1h_v1/SUMMARY.txt`, `RESULTS.txt`, `scores.csv`, `window_scores.csv`, `extreme_scores.csv`, `uncertainty.csv`, `AUDIT.json`, `COMPLETE.json`, `CONTRACT.json`, `MANIFEST.json`, `SELECTIONS.json`, and saved model checkpoints. Runner: `rf_fixed_liquidation_1h.py`. Viewer: `BTC_48h_RF_Fixed_Liquidation_1h.ipynb`.

Decision: retain the three-feature regression/RF candidate. Adding1h liquidation worsened test MAE by0.722766 bps and Spearman IC from0.106801 to0.044611, despite improved validation. April-window degradation dominates aggregate loss. Extreme-move subset improves but does not change full-population objective. Detailed findings: `experiments/rf_fixed_liquidation_1h_v1/INTERPRETATION.txt`. Original conservative modeling recipe passed all audits; no further tuning or deployment.

## RF tuning for fixed four-feature set — started 2026-11-04

User explicitly requested RF hyperparameter tuning suited to the four features, with beating the original baseline and best model as the objective. Runner: `rf_four_feature_tuning.py`; output: `experiments/rf_four_feature_tuning_v1/`. Fixed5m/15m/1h liquidation historical active rates plus volatility ratio. Eight declared RF configurations vary depth2/3/4/5, leaf64/128/256, feature sampling1/.75, bootstrap sample2048/4096 and trees128/256. Six RF blend weights0/.1/.25/.5/.75/1 include standalone RF. Median regression, preprocessing and seed48 remain fixed.

For each outer window, three preceding purged validation quarters select settings. Control uses original RF parameters/original blend grid on those same three folds. Tuned choices require>=.1bps pooled validation gain and wins in>=2/3 quarters; otherwise control. This changes selection history relative to the old one-quarter rule, so report nested original-parameter control separately from historical untuned four-feature and best three-feature references. Identical-cutoff fits cached; up to57 unique artifacts with at most47 new fits expected. No test-driven search expansion. Check COMPLETE.json/AUDIT.json for completion. Historical feature/test exposure remains a limitation.


## RF tuning for fixed four-feature set result — 2026-11-04

Four-feature RF tuning executed and audited. Same fixed liquidation5m/15m/1h historical active rates and4h/24h volatility ratio.
Eight RF configurations and six blend weights, selected using three earlier purged quarters per outer window; pooled MAE plus2/3-fold consistency gate. Median regression and preprocessing unchanged. No test-driven retries.

baseline: test MAE 181.525993 bps; Spearman IC 0.077390; Pearson IC 0.043871.
vol_ratio: test MAE 181.248134 bps; Spearman IC 0.106801; Pearson IC 0.067386.
untuned_4features: test MAE 181.970900 bps; Spearman IC 0.044611; Pearson IC 0.016689.
nested_original_control: test MAE 181.669739 bps; Spearman IC 0.074806; Pearson IC 0.038191.
tuned_blend: test MAE 182.252039 bps; Spearman IC 0.053990; Pearson IC -0.002216.
selected_rf_alone: test MAE 186.085689 bps; Spearman IC 0.022480; Pearson IC -0.023066.
Tuned blend minus baseline MAE: +0.726047 bps; wins in 1/4 windows.
Tuned blend minus vol_ratio MAE: +1.003906 bps; wins in 0/4 windows.
Tuned blend minus untuned_4features MAE: +0.281140 bps; wins in 2/4 windows.
Tuned blend minus nested_original_control MAE: +0.582300 bps; wins in 1/4 windows.

Selected settings:
                    issue            config  rf_weight  validation_mae  fold_wins  n_estimators  max_depth  min_samples_leaf  max_features  max_samples      criterion  random_state
2024-01-01 00:00:00+00:00          original       0.25      141.069065          0           128          3               128          1.00         4096 absolute_error            48
2024-04-01 00:00:00+00:00          original       0.25      159.739507          0           128          3               128          1.00         4096 absolute_error            48
2024-07-01 00:00:00+00:00 deeper_small_leaf       0.50      172.015203          2           128          5                64          1.00         4096 absolute_error            48
2024-10-01 00:00:00+00:00  deeper_subsample       0.25      187.119461          2           128          5               128          0.75         4096 absolute_error            48

Beats original baseline MAE: False. Beats best three-feature regression/RF MAE: False.
Nested original-parameter control isolates RF/blend tuning from changing the validation history. Nested validation repeats historical rows across outer windows and is selection-used, so do not compare its pooled score with old one-quarter validation scores. Entire test history is exposed; this is not independent confirmation. Bootstrap intervals are descriptive and unadjusted for earlier searches.

Sources: `experiments/rf_four_feature_tuning_v1/SUMMARY.txt`, `RESULTS.txt`, `scores.csv`, `window_scores.csv`, `selected_hyperparameters.csv`, `all_validation_trials.csv`, `uncertainty.csv`, `SELECTIONS.json`, `FIT_LEDGER.json`, `CONTRACT.json`, `MANIFEST.json`, `AUDIT.json`, `COMPLETE.json`. Runner: `rf_four_feature_tuning.py`; viewer: `BTC_48h_RF_Four_Feature_Tuning.ipynb`.

Decision: tuning failed both targets; retain best three-feature regression/RF. Actual tuned settings worsen MAE0.582300bps versus the same-three-fold original-parameter control, mainly because July's depth5/leaf64 model fails to transfer validation gains. Jan/Apr fall back to original RF with25% blend; Oct chooses depth5/leaf128/features.75 with25% blend.57 unique artifacts audited,47 new fits and10 reused. The standalone RF metric is the RF component selected for the blend, not separately optimized standalone selection. See `experiments/rf_four_feature_tuning_v1/INTERPRETATION.txt`. No additional search or deployment.

## Short daily PDF report delivered — 2026-11-04

Revision requested: expanded report generated at `output/pdf/ScaleWave_Full_Model_Ranking_2026-11-04.pdf` (five pages), with 31 distinct completed full-period comparisons ranked by test MAE, Spearman IC, full feature identifiers, measured feature effects, work performed and conclusions. Source mapping is in the adjacent CSV; score hashes and passed text/visual checks are in the adjacent `_manifest.json`. Generator: `build_ranked_daily_report.py`; five inspected previews: `tmp/pdfs/ranked_daily/`. Repeated baseline aliases and validation-only tuning trials are excluded from ranking. Previous short report preserved. No training performed for this revision.

User requested a short beginner-friendly PDF containing today's work, measured results and conclusions only, excluding assumptions and next steps. Created `output/pdf/ScaleWave_Daily_Research_Report_2026-11-04.pdf` (one page). Figures read directly from completed runs' scores.csv; source hashes recorded in the adjacent `_manifest.json`. Text values checked and rendered page visually reviewed with no clipping or overlap. Generator: `build_daily_research_report.py`; visual preview: `tmp/pdfs/daily_report_page1.png`. Report distinguishes small four-feature MLP MAE improvement from its lower IC and continued failure to beat baseline. No new training was performed for reporting.

## Two-page feature report revision - 2026-11-04
User requested a shorter beginner report with work performed, feature additions that reduced error, full feature names and conclusions. Created output/pdf/ScaleWave_Feature_Findings_Summary_2026-11-04.pdf (two pages). Both rendered pages visually checked; text checks passed. Includes RF volatility ratio, RF seven-day volatility and four-input MLP, with baseline reference. MLP improvement is distinguished from beating baseline. Generator: build_feature_summary_report.py. Source hashes in adjacent manifest. Earlier reports preserved; no training performed.


## Simplified model table report - 2026-11-04
User found previous reports confusing. Created output/pdf/ScaleWave_Simple_Model_Results_2026-11-04.pdf: two pages with ten main model comparisons ranked by MAE, algorithm names, plain-language features, MAE/IC, work performed and achieved results. Main comparisons only, not exhaustive variants. Both rendered pages inspected; text values verified from completed scores. Generator: build_simple_models_report.py. No training performed.


User wording edit, 2026-11-04: removed the two requested statements about neural networks not beating baseline and the 1-hour feature not improving RF from the simple model PDF. Numerical comparison table retained. Regenerated two pages and visually checked.


User wording edit, 2026-11-04: removed the final MLP conclusion paragraph and the repeated-history/new-data paragraph from ScaleWave_Simple_Model_Results_2026-11-04.pdf as requested. Regenerated two-page PDF; changed page visually checked. Scores and ranking remain unchanged.

