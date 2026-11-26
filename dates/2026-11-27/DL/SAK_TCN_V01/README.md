# SAK TCN V01 — direction replacement experiment

Built for the 27 November 2026 DL folder. This is research code, not a fitted model or a reported alpha result. H4 fitting requires the local audited SAK source package and inputs. GitHub CI uses synthetic data only.

The first question is whether a small causal sequence model improves SAK direction forecasts **without trading gates**. The magnitude model, label, original training indices and forecast calendar stay fixed for this comparison. This is a TCN trained on SAK inputs, not a conversion or distillation of CatBoost weights.

## Locked comparison

| Item | Contract |
|---|---|
| Asset / target | BTC; 15-minute delayed-entry signed return, 10,000 × (P(t+16min)/P(t+1min) − 1) |
| Baseline | Audited SAK_DIR_V01 predictions; frozen SAK_MAG_V01 magnitude for both arms |
| Feature set | Original 146 prepared direction fields, in exact saved schema order |
| Missing data | Preserve NaN; training-only median/IQR scaling; zero placeholder plus validity mask; no forward filling |
| Windows | 61 samples at 1-minute spacing and 61 at 4-minute spacing: 60 and 240 minutes of history |
| Model | Shared 292→8 projection; two causal residual branches, kernel 3, dilations 1/2/4/8, dropout 0.1; linear current-snapshot skip |
| Size | 5,854 trainable parameters |
| Objective | Binary log loss for r > 0; no P&L optimization, no threshold tuning |
| Fit | Original SAK direction train rows (roughly every fifth minute), prior 12 months, quarterly issues 2023Q2–2024Q4 |
| Early stop | Prior 30-day chronological validation with mature-label purge; then refit from scratch on the original full training rows for the selected epoch count |
| Fixed recipe | AdamW, learning rate 0.0003, weight decay 0.01, gradient clip 1, batch 256, at most 20 epochs, patience 3 |
| Prediction | Every eligible original SAK minute; all windows end at decision time |
| Protection | Training runner rejects 2025 origins and issues; no H4a access or fitting |
| Accounting | Every selected minute is a separate entry; overlaps allowed, as in SAK; no K80 trade merging |

Source: SAK's [interfaces](https://github.com/scalewave111/scalewave-ml-research/blob/ca64313fbafe6b4bcc5917e3746b579974c7b19c/sakthi/SAK_GROSS_V01/INTERFACES.md), [feature specification, pipeline rules](https://github.com/scalewave111/scalewave-ml-research/blob/ca64313fbafe6b4bcc5917e3746b579974c7b19c/sakthi/SAK_GROSS_V01/FEATURES.md), and [26 November audit, checks and limitations](https://github.com/scalewave111/scalewave-ml-research/blob/ca64313fbafe6b4bcc5917e3746b579974c7b19c/sakthi/SAK_GROSS_V01/audits/2026-11-26/FINDINGS.md). The TCN sizes, windows and recipe above are predeclared experiment choices, not demonstrated optima.

## Rules, one at a time

Let s = (2 × p_up − 1) × frozen_SAK_magnitude_bps. Model strength means |s| in this implementation.

| Order | Command rule | Exact decision |
|---|---|---|
| 0 | ungated | All forecasts with p_up ≠ 0.5; long above 0.5, short below. Magnitude does not select trades. |
| 1 | top1 | |s| ≥ the prior OOF 99th percentile of |s|; side = sign(s). No ±10 filter. |
| 2 | pm10 | s ≥ +10 bps or s ≤ −10 bps. No percentile filter. |
| 3 | both | Intersection of rules 1 and 2. |

The ±10 rule applies to the **predicted score**, never to realized returns. The earlier ±1-bp wording is superseded. Negative realized outcomes remain in every selected ledger.

Each model gets its own strength cutoff, fixed at each quarterly issue using only earlier out-of-fold scores whose targets have matured. The first OOF quarter is warmup for top1/both. The cutoff can select more or less than 1% in a new quarter; forcing exactly 1% using that quarter's distribution would use future information. Ties at the cutoff are retained.

This prior-OOF percentile policy is an additional experiment. The published SAK_GROSS_V01 interface uses the fixed ±10-bps rule by itself. Original SAK ±10 historical results must not be described as top1 results.

The training command automatically runs **only ungated** scoring. The other rules need separate commands, on the same frozen forecasts. No volatility, liquidation, regime, side, activity or further gate is applied.

## Local use

Use Python 3.12 in an isolated environment. Install a CUDA-compatible PyTorch build for the actual GPU, or use the CPU build below. Do not modify a shared research environment as part of setup.

From this folder:

~~~powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -r requirements-source.txt
.\.venv\Scripts\python -m pytest -q
~~~

Prepare a separate derived cache from the audited package. The adapter verifies the published hashes of the executable source, contracts, model files and reference predictions before loading it. It invokes the existing ResearchData feature constructor, checks exact reference target/calendar parity, creates a complete minute grid, and hashes the resulting cache. Source inputs are opened for reading; Python bytecode writing is disabled during import.

~~~powershell
.\.venv\Scripts\python -m sak_tcn.train prepare --source "C:\Users\Ajay\Desktop\APEX V1\dates\2026-11-26\SAK_GROSS_V01_AUDIT_20261126\source\SAK_GROSS_V01_COMPLETE" --inputs "C:\Users\Ajay\Desktop\APEX V1\dates\2026-11-26\SAK_GROSS_V01_AUDIT_20261126\independent_inputs" --output ".\cache_sak_h4"
~~~

Those paths come from the committed audit provenance; existence still needs to be checked on the PC. The original source modules/weights and large dataset are not in this code package. A source mismatch fails closed. Do not remove integrity checks to accommodate an unverified alternative package.

First run a one-quarter fit/replay smoke test; this is also the first planned fold, not a hyperparameter search:

~~~powershell
.\.venv\Scripts\python -m sak_tcn.train train --cache ".\cache_sak_h4" --output ".\runs_smoke" --folds 2023Q2 --kind tcn --device auto
~~~

Then run the fixed full comparison and controls. Each output must be a new directory:

~~~powershell
.\.venv\Scripts\python -m sak_tcn.train train --cache ".\cache_sak_h4" --output ".\runs_tcn_seed20261127" --kind tcn --device auto
.\.venv\Scripts\python -m sak_tcn.train train --cache ".\cache_sak_h4" --output ".\runs_snapshot_seed20261127" --kind snapshot --device auto
.\.venv\Scripts\python -m sak_tcn.train train --cache ".\cache_sak_h4" --output ".\runs_summary_seed20261127" --kind summary --device auto
~~~

The snapshot control is logistic regression on the same current values/masks. The summary control additionally receives mean, standard deviation and valid coverage over the exact same historical windows; it discards historical order. All controls use the same fit/early-stop procedure. They are neural-toolkit implementations of linear controls, not exact replicas of any earlier ridge experiment.

Read evaluation_ungated/comparison.csv and monthly.csv first. Only after reviewing those results, score the individual rules:

~~~powershell
.\.venv\Scripts\python -m sak_tcn.train score --run ".\runs_tcn_seed20261127" --rule top1
.\.venv\Scripts\python -m sak_tcn.train score --run ".\runs_tcn_seed20261127" --rule pm10
.\.venv\Scripts\python -m sak_tcn.train score --run ".\runs_tcn_seed20261127" --rule both
~~~

A single-quarter smoke run has no previous OOF quarter and cannot produce a top1 evaluation. Reuse the same commands on controls when comparing policies. Do not choose a winning cutoff after seeing these four tables.

Optional MLflow: install MLflow separately in this environment and pass an explicit --tracking-uri to train. Runs go to a new SAK_TCN_DIR_V01 experiment. No current SAK run ID is assumed and no existing experiment is changed. A live MLflow connection was not available during implementation.

## Outputs and decision criteria

Each fold writes weights.pt, scaler.npz, fit.json and predictions.parquet. Saved inference is replayed and compared with the fresh fit before scoring. Run metadata records cache/source identities, versions and the selected epoch. An incomplete job remains marked TRAINING_INCOMPLETE; there is no resume-by-overwrite.

Each policy writes:

- comparison.csv: SAK versus challenger on identical forecast origins; log loss, Brier, direction accuracy, probability and signed-score rank IC, coverage, gross bps/entry and cost sensitivities.
- quarterly.csv: entries and economic results for every requested quarter, retaining quarters with zero entries.
- monthly.csv: entries, long/short counts, gross and scenario net bps, including zero-entry months.
- trades.parquet: exact decisions, outcomes and t+1/t+16 entry/exit times.
- score_diagnostics.csv: all-forecast confidence, magnitude and score quantiles, plus ±10 hit counts, including rejected forecasts.
- paired_uncertainty.json: descriptive 14-day block-bootstrap uncertainty for the difference in daily summed entry-notional P&L, with an assumed 4-bp round-trip cost.

Costs of 0/2/4/8 bps are scenarios, not measurements. Entry-notional P&L is not account return: overlapping entries require separate capital/exposure accounting. The bootstrap includes calendar zero days but does not correct for past strategy search. Historical quarters are already research-exposed.

Ranked next steps:

1. Confirm local source parity and a successful one-quarter fit/replay. Investigate any failed timing or schema check.
2. Run the fixed seven-quarter ungated TCN and both controls. A claim of sequence value needs improvement over SAK and the same-history summary control, not merely over a weak constant comparator.
3. Review the registered rule ladder without changing model weights or targets. A gated improvement cannot be relabelled an ungated win.
4. If promising, repeat the same recipe with seeds 20261128 and 20261129; log all fits, including failures. Review paired economic deltas, worst-quarter/month behavior, concentration and loss of coverage alongside forecast quality.
5. Treat a candidate as provisional until incremental value survives measured execution costs and a prospectively frozen period. No automatic promotion or live trading is implemented.

Reject an alpha claim based only on a single seed, exposed-history maximum, a better IC with worse economics, or score contraction that eliminates activity. If uncertainty includes economically material losses, report inconclusive. A numerical deployment hurdle requires a measured cost/exposure model and is not invented here.

## Research rationale and remaining evidence

Causal convolutions are a plausible way to learn order-sensitive event responses. The baseline already contains engineered temporal features, so that benefit is a hypothesis. The model has a linear skip so it can retain a simpler current-state mapping; pooled-history controls test whether ordering earns its complexity.

| Family | Potential incremental information | Main failure mode | Priority |
|---|---|---|---|
| Small TCN | Local path order and cross-feature interactions across one- and four-hour histories | Relearning existing summaries; noisy overlapping labels; sensitivity to probability scale | Implemented first direction-only experiment |
| Compact N-HiTS | Multi-resolution structure and joint horizons | Pooling erases bursts; long-horizon evidence may not transfer to 15-minute returns | Conditional second family |
| N-BEATSx | Exogenous-driver interactions and residual basis structure | Capacity/basis mismatch to noisy signed returns; covariate availability mistakes | Deferred |

The [TCN paper, Bai et al., 2018, §3](https://vladlen.info/papers/TCN.pdf) motivates the causal/dilated architecture; it does not demonstrate BTC trading alpha. [N-HiTS, 2022 preprint / AAAI 2023, methodology and experiments](https://arxiv.org/abs/2201.12886) and [N-BEATSx, 2021 preprint / IJF 2023, model and electricity experiments](https://arxiv.org/abs/2104.05522) support credible alternatives on other forecasting tasks. [PatchTST, ICLR 2023, §3–4](https://arxiv.org/abs/2211.14730) is a further alternative; its channel-independent form does not by itself prove cross-feature interaction gains.

No claim is made that the local AlphaSense reports or Knowledge Bank guide were reread during this build. The local runner failed before file access. Their full sections, underlying citations, publication dates and data periods still need direct verification. The public SAK source/audit is the implementation contract; private H4a ledgers, recent terminal contents and live MLflow were inaccessible. H4a inactivity needs a separate saved-forecast coverage/score decomposition, not threshold tuning on the second block.
