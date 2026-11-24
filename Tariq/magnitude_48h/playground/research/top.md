> **Later audit correction:** The256-feature forest used the original admission list. A newer BTC audit corrected false technical exclusions and distinguished warm-up/inactive states from missing feeds. The384-feature results belong to1h, not48h. Read [context reconciliation](REPEATABLE%20PLAN/48H_ML/CONTEXT_RECONCILIATION_20261031.md) before treating old exclusions as final.

# TOP — Day-one 48h research: findings, process and next decisions

**31 October 2026 · BTC · magnitude, direction and endpoint extremes**

Today we moved from ranked measurements to fitted forecasts, controlled additions, explicit interactions, market-room conditioning and a first trading-rule simulation. We found recurring candidates, some passing their declared development evidence rules, and useful negative results. This is meaningful research progress. No selected candidate has yet been confirmed on verified untouched data or established as tradable alpha.

This report separates **best observed score**, **passing a development rule**, **unseen confirmation**, and **economic capture**. We should neither dismiss the work because it is unconfirmed nor promote it beyond its evidence. The complete requested research plan and detailed result reports are preserved in appendices below.

## 1. What we built

A reproducible process: exact targets, chronological splits, training-only baselines, feature qualification, explicit missingness, matched comparisons, saved models/predictions, monthly diagnostics, dependence-aware uncertainty, verification receipts and versioned decisions. We can now ask whether X improves a forecast, rather than stop at an association or a plausible story.

At time t, `r = 10000 * ln(P[t+48h]/P[t])`.

| Task | Target | Model output | Primary metric |
|---|---|---|---|
| Magnitude | abs(r) | Conditional median absolute endpoint move | MAE |
| Direction | r>0 versus r<0 | Probability of UP | Log loss; Brier secondary |
| Extreme | abs(r)>500 | Probability of either endpoint tail, combined | Log loss; Brier secondary |

Magnitude is not realized volatility or maximum excursion. Endpoint extreme does not mean merely touching a barrier. Log bps differ from simple-return bps. A probability is not a trading position. MAE182 is average forecast error, not a constant forecast of182.

The canonical package additionally contains upward/downward excursion, time to high/low and extreme ordering. Those are separate outcomes; their full cross-horizon study remains proposed. Future Y must never become a same-origin predictor X. The older one-hour study is separate from today's48h work.

## 2. Exact historical setup and baselines

| Item | Specification |
|---|---|
| Asset / cadence | BTCUSDT /30-minute origins |
| Training origins | 2022-04-22 00:30 through 2023-12-29 23:30 UTC |
| Training rows |29,615|
| Boundary purge |96 origins; training label endpoint strictly before2024-01-01 UTC|
| Development origins |2024-01-01 00:00 through2025-01-22 00:00 UTC|
| Development rows |18,481|
| Original population including purge |48,192|
| Input admission |522 F4Z fields;256 qualified,266 skipped with reasons|
| Untouched holdout |None verified|

| No-feature forecast, fitted on training only | Development score |
|---|---|
| P(UP)=0.5057909843 | Log loss0.692023; Brier0.249438 |
| Median magnitude161.461839 log bps | MAE187.831072 |
| Mean magnitude approximately258.06 log bps | RMSE approximately245.28; separate mean/squared-error track |
| P(extreme) approximately0.1449940908 | Log loss0.449294; Brier0.138123 |

Training constants are frozen. Full-history15.33% extreme frequency is descriptive, not the honest training-only baseline. Overlapping48h labels mean rows are not independent evidence. Historical stages used7/14-day paired calendar-block analyses; individual family corrections do not correct the entire adaptive research history.

## 3. Step by step: KS audition → standalone → addition → interactions

### Step 1 — Define Y and the comparison

The initial48h extreme screen contained4,179 UP extremes,3,209 DOWN extremes and40,804 in-range rows. Both tails were combined against in-range. This is distinct from direction at zero or magnitude regression. A different target needs its own baseline, loss and interpretation.

### Step 2 — Screen with KS

Compare the empirical cumulative feature distribution in extreme rows with that in in-range rows. KS is their maximum absolute vertical separation.19.73 KS points means distance0.1973, not19.73% accuracy or extreme probability. The accompanying higher/lower association arrow is a separate descriptive orientation; KS itself is unsigned. It is not a long/short arrow.

The supplied ranking began with funding persistence percentile1h (19.73 points, lower associated with extremes), funding persistence4h (14.71), cross-asset historical liquidation active rate5m (13.65), cross-asset1h (13.22) and BTC activity1h (13.15). Funding range, basis and breadth also appeared. Related windows/transforms reveal recurring concepts, not independent discoveries.

KS admits a hypothesis to testing; it does not establish useful probabilities, causation or generalization. The original ranking is historical context, not a newly recomputed screen in this report. The earlier magnitude Spearman around+0.195 for cross-asset activity1h was another descriptive association, not a fitted forecast.

### Step 3 — Qualify the measurement

Check value availability at t; historical-only normalization; valid coverage; missingness, resets and source-boundary behavior; formula/units/sign; constants and saturation. Account for all522 fields rather than silently omit failures.256 passed the existing technical admission. Export timing and bounded normalization replays do not substitute for complete raw upstream lineage review.

A useful correction: cross-asset activity1h equals1.0 throughout magnitude development. Its better MAE is a different constant forecast, not discrimination between observations. High screen rank did not protect against saturation.

### Step 4 — Freeze preprocessing and missingness

Impute missing X using its training median. Add M=1 for missing,0 for present. Fit mean/std on training only. Preserve all declared development rows. Never backfill from the future. Missing context is separate from neutral context; unavailable Y is not imputed.

All-valid/both-valid diagnostics are secondary matched checks. An explicit missing-only model is a recommended control in the forward process; not every historical stage fitted it separately merely because missing flags were included.

### Step 5 — Establish M0

Fit the target-appropriate training constant, freeze it, predict all later rows and save its score/keys. A feature model must improve this probability or numerical forecast on the same rows. Majority-class accuracy is unsuitable as the main extreme-probability measure.

### Step 6 — Fit the standalone tournament

Extreme/direction: logistic regression on standardized value plus missing flag. Magnitude: unpenalized median regression minimizing absolute training error, with predeclared nonnegative prediction clipping. The computer learns coefficients, not a manually assigned KS weight.

Only X changes within each tournament. Save predictions, losses, gain versus M0, coefficients, coverage, months, uncertainty, constants and failures. The first funding-persistence extreme model failed to beat M0 despite its strong KS ranking. That was a useful negative result.

### Step 7 — Automate unique additive pairs

Fit `intercept + X1 + M1 + X2 + M2`, with both weights learned jointly. No product yet. Reuse the same train/development rows, preprocessing, missingness and within-stage solver settings. Enumerate each unordered pair once, excluding self-pairs.

- Extreme:71 standalone improvers →2,485 pairs.
- Magnitude: displayed top20 →190 pairs; not all32 improvers.
- Direction:217 improvers →23,436 pairs. Its displayed top20 contains190 already-fitted pairs.

These are different branch scopes. Do not mix their counts or pretend every stage had identical regularization.

### Step 8 — Ask whether addition actually added information

Compare pair versus M0, each parent, its fixed better parent and the best global simple model. A pair can beat M0 while adding almost nothing beyond one feature. Save absolute and incremental rankings separately.

Actual example: magnitude best single182.284669 MAE versus best pair182.036977. The pair wins on average but adds only0.247692 log bps (0.136%), with uncertainty unresolved. Conversely, some tiny parent-relative gains passed their development rule while scoring worse than the best global single.

### Step 9 — Sanity-check gains

Inspect both-valid rows, missingness patterns, source periods, correlated inputs, rank/identifiability, monthly contributions and removal of the strongest one/two months. Never replace the primary population with a favorable subset. Stress concentration needs scoped interpretation and replication; it is not automatically proof for or against a rare-event forecast.

### Step 10 — Test explicit interaction

Add X1×X2 while retaining main effects and the exact additive comparator. Fit product normalization only on training. Ask whether multiplication improves beyond addition, not merely beyond M0. A product is a fitted conditional relationship, not a causal mechanism.

Room interaction uses categorical indicators and X×room terms; room IDs are not continuous numbers. Missing/sparse rooms retain explicit fallback. Keep every pair/room/Z trial in the search ledger. User-requested continuations after failed gates are new exploratory branches, not retroactive passes.

### Step 11 — Compare compact trees and broader additive models

Use matched inputs and rows, fixed capacity, no quiet development tuning. Trees can discover conditional structure but do not prove specific causal interactions. A tree failure rejects that tested specification, not every nonlinear model. A model family is an optional separately budgeted question, not an obligatory rung.

### Step 12 — Verify, record, decide

Freeze contracts; save manifests, models, predictions, scoreboards, uncertainty, diagnostics and decisions. Replay numerical predictions/scores; retain solver certificates where applicable. Keep failures and reruns. Numerical correctness does not establish generalization or executable profit. Stop at the declared budget rather than search until something looks favorable.

## 4. Extreme probability: additive average improvements, branch parked

| Model | Log loss | Brier |
|---|---:|---:|
| M0 |0.449294381|0.138122795|
| Best single: funding range percentile1h |0.446904|0.137515|
| Funding range percentile1h + cross-asset activity5m |0.444267760|0.136807454|
| Pair plus product |0.444236899|0.136807391|
| Compact12 L2 logistic |**0.443678457**|**0.136616599**|
| Compact XGBoost |0.452299758|0.139167643|
| Broader31-concept ridge |0.444191928|0.136738232|

71/256 singles improved log loss; none passed the full rule.1,418/2,485 pairs improved the better parent; none passed the full rule. Best-pair gains survived both-valid checks but were month-concentrated. The product contributed almost nothing; the tree failed. Broader ridge chose C=0.001 using training-only purged folds but did not beat compact12.

Compact12 improved7/13 months and on all-valid rows. Yet uncertainty included zero and removing March/August largely removed the edge.84.4% of incremental log-loss gain beyond the pair came from the6.29% of rows with at least one missing input. This does not prove missingness alone caused it.

**Achievement:** better-average development probability forecasts and reproducible rejection of several complexity upgrades. **Decision:** parked; reliable prediction unestablished. Recorded extreme fits2,772, including2 audit refits.

## 5. Magnitude: the clearest recurring candidate family

32/256 singles improved M0. Three fields representing two concepts passed the frozen development rule.

| Candidate | MAE | Improvement vs M0 | Months better | Status |
|---|---:|---:|---:|---|
| Cross-asset activity15m |**182.284669**|2.953%|10/13|Best single average; uncertainty failed|
| BTC activity15m |182.486396|2.845%|10/13|Exploratory|
| BTC activity1h |**184.446013**|1.802%|10/13|Development rule passed|
| Liquidation fast/slow lift15m/1h RZ |**187.314266**|0.275%|12/13|Development rule passed|
| Same lift percentile |**187.332386**|0.265%|12/13|Same concept; development rule passed|

BTC activity1h forecasts vary only approximately177.14–183.07logbps; some benefit may be forecast-level adaptation. Cross-asset activity1h is completely saturated and predicts constant194.233; its good score is not time-varying discrimination. These limitations do not erase the legitimate passing development results.

All190 top20 pairs beat M0;108 beat the better parent. Best absolute pair: cross-asset activity15m+5m, **182.036977**, only0.247692 beyond its parent and still exploratory.

Three pairs passed their parent-relative rule: liquidation severity percentile15m + acceleration RZ5m/15m (MAE186.583333,gain0.069864); severity RZ15m + acceleration (186.610895,gain0.046992); fragmentation percentile15m + acceleration (186.595031,gain0.031916). All improved10/13 months. These are two concept combinations with tiny gains, worse absolute scores than the best single.

**Achievement:** recurring liquidation activity and fast/slow activity-change candidates for endpoint magnitude, including two concepts surviving declared development uncertainty checks. The lowest score and strongest evidence are different lists.

## 6. Room/context progression, versions1–5

| Version | Experiment | Result |
|---|---|---|
| V1 |Global winning pair + joint1d offsets|MAE183.870886 versus182.036977; worse|
| V2 |Frozen pair + training-median room residual offsets|183.334720; worse overall,4 rooms improved|
| V3 |Top20 linear and fixed trees with1d/4h|Linear182.362546; trees184.046292/183.999201/183.999201;4h unused|
| V4 |190 pairs with1d-dependent slopes and offset controls|Every full-population interaction worse than own global pair; local candidates only|
| V5 |Selected room/pair + every eligible incremental Z|254 Z,509 fits;54 interactions beat all controls;zero full credibility passes|

Room = PRICE×FLOW×LIQ,27 possible states. Training-only thresholds, causal availability, explicit missing room, sparse fallback and source-transition masks are retained. V4 evaluated14 supported rooms, not27 equally populated independent states.

### Specific room relationship worth preserving

**UP / BUY / SHORT-LIQ**, BTC historical liquidation activity5m + cross-asset activity5m (MP0055).3,858 training and2,684 development rows,166 development UTC dates.

| Same-room comparator | MAE |
|---|---:|
| Own global pair |200.990990|
| Room offset |198.446871|
| Room-dependent slopes |**192.259952**|

Room slopes improved12/13 months versus both controls. Nominal intervals favored them; adjusted intervals across5,320 room contrasts included zero. Full-population performance was worse. This is a scoped conditional hypothesis, not a global context-model success. Its192 score cannot be directly compared to global182 on different rows.

### Incremental Z in that room

All522 accounted for:254 eligible Z,2 base features,266 earlier exclusions. One matched base refit +254 additions +254 product models =509 fits. Products X1×Z and X2×Z retain main effects/missing flags. No combined25-feature model was fitted.

98 additions beat anchor;82 interactions beat anchor;90 beat their matched addition;54 beat anchor/refit/addition. None passed the full credibility rule.

| Z | Addition MAE | Interaction MAE | Months better vs anchor |
|---|---:|---:|---:|
| Flow activity RZ5m |191.961900|**190.903227**|7/13|
| Basis range RZ15m |192.089882|190.975037|9/13|
| Basis range percentile15m |192.0460|191.036915|9/13|

Leading interaction gain1.356725logbps (0.7057%); all2,684 Z values valid. Its nominal14-day interval includes zero. A simpler varying-Z addition, cross-asset activity15m, scores **190.882727**, slightly better. The lower190.006884 addition using saturated cross-asset activity1h is not evidence of varying Z information.

Research card `BTC48H_MAG_R1_FLOW5M_RESEARCH_V1` freezes exact coefficients/transforms/source bindings. Research registration does not equal stage13 alpha promotion. Magnitude fits throughV5:1,340; extremes+magnitude4,112 before the separate direction branch.

## 7. Direction: probability improvements, no confirmed upgrade

Linked Knowledge Bank branch:256 singles,217 log-loss improvers;23,436 pairs successful;11,941 beat their better parent. One pair passed its parent-relative rule but failed the global-best-single requirement. No pair earned full advancement.

| Model | Log loss | Brier |
|---|---:|---:|
| M0 |0.692023|0.249438|
| Variance percentile4h |**0.685546**|0.246215|
| Liquidation active severity RZ4h |0.685558|0.246222|
| Basis persistence percentile1h + variance percentile4h |**0.684958**|0.245925|
| Flow activity RZ4h + variance RZ4h, addition |0.685277|0.246083|
| Same pair + product P9050_I |**0.684953**|0.245946|
| Matched20-feature logistic |0.691796|0.249280|
| Fixed20-feature tree |0.707316|0.256886|

Best single improved9/13 months but no single passed its uncertainty rule. The later authorized continuation fitted10 concept-distinct products plus logistic/tree controls:12 new fits. Six products lowered log loss versus their parents; none passed advancement. The initial direction report's 'not run' statement is historical; the continuation above is later evidence.

Unsigned variance can condition UP probabilities without an intrinsic directional sign. Ranking first does not imply high variance is bullish. AUC is ranking, not directional accuracy. Model associations do not identify causal explanations such as rebounds.

## 8. First trading-rule replay

A development price-proxy experiment, not proven executable performance. Room UP/BUY/SHORT-LIQ; long only;5% entry equity allocation;48h hold; one-minute delayed cross-venue VWAP proxies. Assumed5bp fee+5bp spread/slippage per side, approximately20bp round trip. Cash-long proxy, no funding; not a perpetual simulation.

92 shared nonoverlap slots. Rejected opportunities remain reserved until the shadow exit; this isolates filtering but differs from unrestricted policy-specific re-entry.

| Rule | Trades | Net account return | Maximum drawdown | Average account exposure |
|---|---:|---:|---:|---:|
| Room long |92|+1.3303%|-2.0326%|2.3676%|
| Direction P(UP)>0.5 |85|+1.6796%|-1.9126%|2.1880%|
| Magnitude>300logbps |6|+1.1747%|-0.3781%|0.1571%|
| Both filters |6|+1.1747%|-0.3781%|0.1571%|

Direction skipped4 losers and3 winners; improvement uncertainty includes zero. All6 magnitude entries already passed direction. Two February trades produced more than the entire filtered profit; excluding them leaves approximately-0.0520% account return. Lower drawdown partly reflects much lower exposure. The magnitude filter discarded44 winners and42 losers.

The300bps cutoff,5% allocation and long interpretation were illustrative, not learned optimal decisions. P(UP)>50% is insufficient for positive expected profit when loss/gain sizes and costs differ. No threshold search occurred; extremes remained parked.

All trades, opportunities, minute equity and diagnostics are saved. Independent compounded trade returns reconcile terminal equity. VWAP is not an executable quote; venue costs, fills and capacity remain unverified. No untouched outcomes were opened.

## 9. Values, probabilities and distributions

The machine learns a mapping X→Y by minimizing the specified training loss. Logistic yields event probability; MAE regression yields a conditional median. Their outputs change by observation; scores summarize errors.

We have magnitude point forecasts, UP/DOWN probabilities and combined-tail probabilities. We do not yet have an established expected signed-return model or one coherent signed-return distribution. Quantiles/distribution modelling were discussed, not executed. Do not multiply separate direction and unconditional magnitude predictions and call the result expected return. A richer output does not create information; it needs a matching baseline and calibration/scoring tests.

## 10. Best practices and lessons to carry forward

The adopted investment framework is data feeding expected returns, risk and costs **in parallel**, then portfolio construction, implementation and attribution. Forecasts are inputs, not orders.

1. Preserve target/formula/units, matched rows and training-only baselines/transforms. Purge by the actual horizon, not a universal48h gap for future tasks.
2. Keep missingness, source boundaries, saturation and upstream normalization visible. Scaling cannot repair noncausal data.
3. Compare against M0, missingness-only information where applicable, the better parent and the strongest simple alternative. Separate absolute and incremental rankings.
4. Predeclare meaningful effect size as well as uncertainty. A tiny passing gain may not matter economically.
5. Maintain one search ledger across features, rooms, models, targets/horizons/assets and restarts. Stage-wise uncertainty does not erase earlier selection.
6. Reused development history stays development, even if repartitioned. Selection based on full-development rankings cannot be made historically training-only just by refitting coefficients.
7. Standalone wins are not a mathematical prerequisite for joint value. Nor is singles→pairs→trees an obligatory ladder. Additional families require a bounded question.
8. Stress concentration is an evidence limitation requiring replication, not automatic disproof. Do not loosen historical gates after seeing failure.
9. Distinguish replay correctness, independent-person review, statistical generalization and economic usefulness. Books and verification receipts do not certify alpha.
10. Do not infer participant motives or causality from products or room labels. Conditions describe measurements; they do not promise continuation.
11. A trading test requires explicit side, exposure, overlap, entry/exit, costs and execution assumptions. Lower exposure can explain lower drawdown.
12. Save complete results, not only winners. Print numerical summaries and exact feature mappings. Log milestones and at least every10 minutes during active work; this is not a background scheduler.

The mistake to avoid is letting 'what else can we try?' outrun 'what have we established?' The correction is not to discard this day. Preserve its legitimate candidates, failures, measurement corrections and reproducible process, then choose the next question deliberately.

## 11. Completed, open and next

**Completed:** baselines/admission; extreme audit and parking; magnitude singles/pairs/contextV1–V5; direction singles/pairs/bounded complexity; exact research candidate card; four-policy replay; step-by-step reports and verification receipts.

**Retain:** liquidation activity and fast/slow activity change for magnitude;4h volatility/activity for direction; one selected room-conditioned liquidation relationship; exploratory flow/basis increments. Retain negative trees and saturated-feature corrections alongside the winners.

**Open:** full upstream lineage; unexposed data availability; level-shift versus changing-value information; useful-effect thresholds; independent replication; venue-executable costs/fills; justified decision policy. No full distribution model, universal horizon map or validated alpha exists yet.

Freeze candidates and audit fresh-data availability without opening reserve outcomes. If the next choice is further discovery instead, name it as development with an explicit budget. The source plan's80 BTC cells and eventual240 asset-task cells remain proposals.825 inputs×80 cells×4 folds could mean264,000 fits before controls; no such expansion is launched here. Continuing research is authorized by deliberate questions, not by an old document's next-step sentence.

## 12. Detailed evidence and preserved appendices

The following material preserves the full requested research plan and detailed branch reports. Earlier stopping statements, fit totals and proposed next actions remain historical. The current synthesis above incorporates later direction/context/trading results. Original run artifacts remain numerical authority; this consolidation performs no new fits or fresh holdout evaluation.

Each appendix includes an original-source link. Links embedded within preserved source text retain their original context; use that source for navigation. The first plan predates later work: its3,218-fit total and proposed80-cell next step are historical, not the final current status.

### Full research-process and Y-by-horizon plan

[Original document](<C:/Users/REDACTED_USER/Desktop/Docs Knowledge Bank Intro/research/ML_RESEARCH_PROCESS_AND_Y_HORIZON_PLAN.md>) ? SHA-256 `5abef6d53cf0480445a57e92bcf1a56c28fa15c0c8c6720b5852cc96c75c845b`

<details>
<summary>Read complete preserved document</summary>

# ScaleWave ML: research history and repeatable Y-by-horizon test plan

Prepared 31 October 2026. This is a research handoff, not a claim of validated alpha or practitioner endorsement.

**Purpose:** preserve how the BTC 48-hour work was done, explain the actual supplied Y vector, and specify how to automate the question: **which available X measurements predict which future outcome, at which horizon, beyond an appropriate baseline?**

**Status:** the previous 48-hour extreme search is parked. A separate 48-hour magnitude branch has completed 256 standalone median regressions and 190 additive pairs; its results are included below. The broader multi-target, multi-horizon workflow is proposed and has not been run. Preparing this document performed no model fits, changed no source data, and did not build a new research system.

## 1. The two packages we use

Both packages are required:

- **Handoff 1 — source X and reference prices:** `C:\Users\REDACTED_USER\Desktop\MFT ML\Team Handoffs\Handoff 1_F4_f4Z.zip`
- **Handoff 2 — canonical Y, signed returns, loader and contracts:** `C:\Users\REDACTED_USER\Desktop\MFT ML\Team Handoffs\HANDOFF_2_CANONICAL_XY_V1_1.zip`

Handoff 2 does not replace Handoff 1. Its historical references call Handoff 1 `Forge_Complete_90pct_2022-04-22_to_2025-01-23.zip`; use the actual local filename explicitly when resolving the source archive. Do not assume a renamed archive matches on filename alone. Its manifest records an expected Handoff 1 SHA-256 of `4d70380d7fdd90cc04edae1dbeb4837e901828db6709b03758a81d8ab6a8f31a`; a full archive rehash was not performed for this document.

The supplied release is `CANONICAL_XY_V1_1`, with Y version `forge_f066_path_v2` and returns version `forge_f066_complete_path_returns_v1`. The package describes:

| Item | Supplied scope |
| --- | --- |
| Assets | BTCUSDT, ETHUSDT, SOLUSDT |
| Source days | 1,008 daily partitions, 2022-04-22 through 2025-01-23 |
| Decision timestamps | 2022-04-22 00:01 UTC through 2025-01-24 00:00 UTC |
| Native clock | One-minute bucket ends |
| X | 303 F4 values and 522 F4Z values; quality fields remain separate |
| Families | Price, flow, basis, funding, open interest, liquidation, volatility |
| Y table | 43,545,600 rows, including explicit invalid/unavailable outcomes |
| Return table | 43,545,600 rows; same complete-path validity as Y |
| Valid rows in each table | 43,314,805, according to the release manifest |
| Review status | Internal QA passed; independent-person review pending |

These counts describe the release, not independent statistical observations, and were not fully recounted for this document. The final midnight belongs to the last supplied partition. The package says 112 source days, 2025-01-24 through 2025-05-15, were withheld from features and prices. They are not supplied here, and withholding does not establish that nobody previously inspected them.

Use the packaged loader and existing tables. Do not rebuild Y merely to change features, targets, models or horizons. The package's historical tree assignment is background, not the experiment executed in this thread or an instruction to execute now.

## 2. What the Y vector actually means

The Y key is `(canonical_symbol, decision_time, horizon_minutes)`. Each row has three keys, **seven outcome fields**, and `label_status`.

Let `H` be the horizon in minutes. The complete path contains `H+1` prices, from minute 0 through minute H, including the origin. Let `P0`, `PH`, `Pmax` and `Pmin` denote its starting, ending, highest and lowest prices.

| Outcome | Definition | Predictive question |
| --- | --- | --- |
| `direction` | `sign(PH - P0)`, coded -1, 0, +1 | Does the endpoint finish down, exactly flat, or up? |
| `net_magnitude_log_bps` | `abs(10000 * ln(PH/P0))` | How large is the endpoint move, regardless of direction? |
| `up_magnitude_log_bps` | `10000 * ln(Pmax/P0)` | How far above the origin does the path reach? |
| `down_magnitude_log_bps` | `-10000 * ln(Pmin/P0)` | How far below the origin does the path reach? |
| `time_to_high_min` | Earliest minute attaining Pmax, between 0 and H | When does the overall path high occur? |
| `time_to_low_min` | Earliest minute attaining Pmin, between 0 and H | When does the overall path low occur? |
| `extreme_order` | +1 if high precedes low; -1 if low precedes high; 0 if same time | Which overall extreme occurs first? |

The separate return table contains:

```text
signed_return_log_bps    = 10000 * ln(PH/P0)
                        = direction * net_magnitude_log_bps
signed_return_simple_bps = 10000 * (PH/P0 - 1)
                        = 10000 * expm1(signed_return_log_bps / 10000)
simple_return_percent   = signed_return_simple_bps / 100
```

Important distinctions:

- Net magnitude is not realized volatility. A path can move substantially and finish unchanged.
- Downward excursion is measured from P0; it is not running-peak maximum drawdown.
- Extreme ordering is not first barrier-hit ordering. A threshold-hit target would need its own definition.
- Times to extremes represent timing, not instantaneous speed. Minute zero is valid. A flat path has both times zero and order zero.
- Ties use the earliest occurrence of the original exact decimal price, not a rounded float comparison.
- Signed log and simple returns are representations of the same endpoint outcome; do not count them as independent discoveries.
- Multiplying a separately predicted direction by an unconditional predicted magnitude generally does not give expected signed return.
- Y fields describe the future. None may be inserted into X for predicting another Y field at the same origin.

The ten supplied horizons are:

| Display | Minutes |
| --- | ---: |
| 15 minutes | 15 |
| 1 hour | 60 |
| 4 hours | 240 |
| 8 hours | 480 |
| 16 hours | 960 |
| 48 hours | 2,880 |
| 72 hours | 4,320 |
| 120 hours | 7,200 |
| 200 hours | 12,000 |
| 300 hours | 18,000 |

An X lookback and a Y horizon are independent choices: a trailing one-hour funding feature may predict a future 48-hour outcome. Do not make feature lookbacks equal to target horizons by assumption.

### Price authority, validity and source changes

The reference price is `f066__value`, cross-venue minute trade VWAP, stored as decimal128. It is not a candle close or a guaranteed executable price. Admission requires valid and structurally valid source data, finite positive prices, unique exact-minute keys, and known `available_after <= decision_time`.

All H+1 prices must pass. No interpolation, forward-fill, backfill or endpoint-only exception is permitted under this contract. Both signed-return fields inherit this full-path rule. `OUTSIDE_RELEASE` takes precedence when the endpoint exceeds supplied coverage; invalid rows keep their keys and null outcomes. Never convert unavailable Y into zero or impute it.

The guide records an invalid BTC/ETH price minute at 2024-03-05 19:59 UTC. It invalidates H+1 overlapping origins per horizon, not just one endpoint. SOL has no such minute according to the package. Missing-label selection therefore varies with H.

A source-binding transition occurs at 2024-10-29 00:01 UTC. The canonical loader excludes crossing paths by default. Keep that default for new experiments and report pre/post behavior. Reproducing old 18,481-row runs requires their exact saved keys and policies; do not silently change their sample to the current loader default or compare mismatched populations.

### What was checked for this document

Read both archives' entry documents and Handoff 2's schemas, guide, manifests and relevant builder/loader code. Checked 28 Handoff 2 members against their internal manifest hashes, including the BTC price file and ten Y plus ten return partitions.

At the already historical BTC origin **2023-01-15 00:00 UTC**, independently reconstructed all ten horizons directly from the decimal price paths. Direction, earliest-extreme times and ordering matched exactly. The largest scalar discrepancy was approximately **1.52e-12**, below the contract's 1e-8 absolute log-bps tolerance.

This was a bounded read-only measurement check, not a new outcome screen, full archive audit, complete formula revalidation or independent-person signoff. No ETH/SOL predictive performance was assessed.

## 3. What happened in the research thread

The actual modeled task was mainly **BTC, 48-hour endpoint extreme probability**:

```text
r = signed_return_log_bps
y_extreme = 1 if abs(r) > 500 else 0
```

This is a derived binary endpoint event, not one of the seven canonical Y columns, not a path-excursion threshold, and not exactly a +/-5% simple-return event. The threshold is 500 log bps, or +/-0.05 in log return.

The historical experiment used 30-minute origins, 29,615 training rows, 96 boundary-purged rows, and 18,481 later development rows. The cutoff was 2024-01-01 UTC. Training required `t + 48h < cutoff`; development began at the cutoff. The original training event-rate baseline was approximately 0.1449940908.

| Stage | What was done | What was learned |
| --- | --- | --- |
| Initial screens and baselines | Earlier feature screens and no-feature direction, magnitude and extreme baselines were preserved. Direction/magnitude work did not establish a full Y-vector model search. | The later history had already been exposed before the model tournaments. |
| First M1 | Training-median imputation plus missingness flag; one funding-persistence logistic model. An independent optimizer refit and unpenalized comparison followed. | Probability losses did not beat M0. AUC around 0.5503 and descriptive bins showed association, not a successful probability forecast. |
| Standalone tournament | All 522 F4Z fields considered; 256 qualified and fitted; 266 skipped with reasons. Shared standardized value plus missingness design and fixed L2 logistic settings. | 71 improved development log loss; none passed the frozen stability/uncertainty rule. |
| Addition | All 2,485 unique pairs of the 71 improvers; matched rows and fixed preprocessing. Compared with M0, each parent and the better parent. | 1,418 improved beyond the better single parent; none passed the full rule. |
| Best-pair sanity | Funding range percentile 1h plus cross-asset liquidation historical active rate 5m. Checked both-valid rows and month contribution. | Both-valid gain survived, but removing March reversed the full-row pair-versus-parent gain; March plus August also reversed the both-valid gain. |
| Interaction | Added one standardized product to the same pair, with training-only product scaling. | Tiny average improvement; both-valid Brier worsened and intervals spanned zero. Conditional small interaction tournament did not run. |
| Compact tree branch | Later, one fixed 12-concept XGBoost and a same-input 12-feature L2 logistic control were fitted. | The tree failed to beat M0, the pair or the matched logistic. This was a separate branch after the earlier locked-path stop. |
| Broader ridge branch | 71 improvers reduced to 31 family:sensor representatives. Six C values, four purged chronological training folds, then one final fit. | C=0.001 selected; broader model failed to beat the 12-feature model. Features and C both changed. |
| Final saved-model audit | No new fits. Audited the 12-feature model against M0 and the pair, missingness subsets, months and uncertainty. | Reliable predictive improvement was not established; current target/model search was parked. |

The first unscaled M1 and later standardized tournament were separate specifications. Fixed settings within a stage do not mean every stage used an identical model design.

### Preserved development scores

| Model | Log loss | Brier |
| --- | ---: | ---: |
| M0 | 0.449294381 | 0.138122795 |
| Best standalone: funding range percentile 1h | 0.446904, rounded | 0.137515, rounded |
| Best additive pair | 0.444267760 | 0.136807454 |
| Pair plus product | 0.444236899 | 0.136807391 |
| 12-feature L2 logistic | 0.443678457 | 0.136616599 |
| Fixed compact XGBoost | 0.452299758 | 0.139167643 |
| 31-feature ridge, C=0.001 | 0.444191928 | 0.136738232 |

The 31-feature set was not a strict superset of the twelve: it retained eleven and omitted the realized-variance representative. Its best pooled inner-validation log loss, 0.378703149, also failed to beat the corresponding fold-specific constant baseline, 0.363265232. Best C within a grid is not evidence that a model beats a baseline.

The final 12-feature audit found **84.4% of its total incremental log-loss gain over the pair on the 1,163 rows with at least one missing feature**, only 6.29% of development. Its 17,318 all-valid rows still improved, but by only 0.000098374 over the pair. That does not prove missingness alone caused the gain. The uncertainty and concentration conditions failed.

The extreme branch's recorded fit accounting totals **2,772 feature-model fits**: 1 initial M1 + 2 audit refits + 256 singles + 2,485 pairs + 1 interaction + 2 compact-model fits + 25 broader-ridge fits. No-feature baseline estimates and earlier screens are separate history; the fit count is not a complete count of every research decision or hypothesis examined. Final diagnostic audits added zero fits.

## 3A. Added development: 48-hour magnitude, then all top-20 pairs

Updated from the saved magnitude contracts, scoreboards and result documents following the next user-supplied thread segment. Eight selected contract/result/scoreboard artifacts were checked against their run manifests for this update. No new models were fitted here; the stored verification reports describe the earlier full source/model/LP-certificate checks.

This branch asks **how large the absolute endpoint move will be**, using `net_magnitude_log_bps = abs(signed_return_log_bps)`. It does not ask whether the endpoint exceeds 500 log bps, which direction it moves, or how large the intrahorizon excursion is. That is a legitimate change of scientific question, recorded as a separate task rather than a rescue of the failed extreme model.

### Magnitude contract and completed standalone run

- Same BTC 48h horizon, 30-minute origins, 29,615 training rows, 96 purged rows and 18,481 development rows.
- M0 forecasts the **training median 161.461839 log bps**. Its development **MAE is 187.831072 log bps**. Forecast level and forecast error are different quantities.
- Reuse the 256 technically qualified fields from the 522-field audit, not the 71 extreme-target improvers. Reuse was justified by the same input timestamps and target-valid population; it does not make that list universal for other horizons/assets.
- Fit **unpenalized conditional median regression** with intercept, standardized training-median-filled value and missingness flag. Minimize absolute training error using the fixed HiGHS linear-program solver; project forecasts to nonnegative values under a predeclared rule.
- All 256 fits completed, zero failures. **32** lower development MAE. **Three fields, representing two concepts**, pass the declared development rule: BTC liquidation historical active rate 1h, and the RZ/percentile versions of liquidation fast/slow lift 15m/1h.
- The rule requires lower MAE, a majority of months better, nonconstant predictions, and negative family-simultaneous upper uncertainty bounds at both 7-day and 14-day block lengths. It is a conditional development rule, not confirmation across all previous searches.

The best average standalone is cross-asset liquidation historical active rate 15m: **MAE 182.284669**, a **5.546402-log-bps / 2.953%** reduction from M0, with 10/13 months better. It fails the declared uncertainty rule. The strongest average score and the candidates passing that rule are different lists.

The cross-asset 1h active-rate feature equals **1.0 on every development row** and predicts a constant **194.233 log bps**. Its improved MAE does not demonstrate discrimination among future magnitudes. The BTC activity 1h development lead also has a narrow forecast range, approximately 177.14?183.07. Keep forecast-level adaptation separate from changing predictions usefully across observations; a merely nonzero prediction range is a weak discrimination check. A historical pooled correlation does not resolve this issue.

### Top 20 standalone magnitude features actually used for addition

The S-identifiers below are display aliases for these exact fields in this document, not replacement source IDs. Lower MAE is better. Improvement is relative to M0 MAE 187.831072. `Lead` means the recorded development rule passed; it does not mean validated. The other two standalone leads, fast/slow lift variants, rank below this top 20 and therefore did not enter the requested top-20 pair tournament.

| ID | Exact feature identifier | MAE | Improvement vs M0 | Months better | Recorded status |
| --- | --- | ---: | ---: | ---: | --- |
| S01 | `F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__15m__value` | 182.284669 | 2.953% | 10/13 | Exploratory |
| S02 | `F4Z__liquidation__liq_activity_severity_historical_active_rate__15m__value` | 182.486396 | 2.845% | 10/13 | Exploratory |
| S03 | `F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__1h__value` | 183.077047 | 2.531% | 10/13 | Constant forecast; exploratory |
| S04 | `F4Z__liquidation__liq_activity_severity_historical_active_rate__5m__value` | 184.120051 | 1.976% | 8/13 | Exploratory |
| S05 | `F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__5m__value` | 184.300412 | 1.880% | 10/13 | Exploratory |
| S06 | `F4Z__liquidation__liq_activity_severity_historical_active_rate__1h__value` | 184.446013 | 1.802% | 10/13 | Lead |
| S07 | `F4Z__liquidation__venue_fragmentation_percentile__15m__value` | 186.626948 | 0.641% | 11/13 | Exploratory |
| S08 | `F4Z__liquidation__liq_activity_severity_active_severity_percentile__15m__value` | 186.653197 | 0.627% | 11/13 | Exploratory |
| S09 | `F4Z__liquidation__liq_activity_severity_active_severity_rz__15m__value` | 186.657887 | 0.625% | 11/13 | Exploratory |
| S10 | `F4Z__liquidation__liq_acceleration_rz__5m__15m__value` | 186.672217 | 0.617% | 11/13 | Exploratory |
| S11 | `F4Z__liquidation__liq_side_pressure_percentile__15m__value` | 186.712380 | 0.596% | 11/13 | Exploratory |
| S12 | `F4Z__liquidation__liq_temporal_concentration_percentile__15m__value` | 186.713031 | 0.595% | 11/13 | Exploratory |
| S13 | `F4Z__flow__flow_confirmation_rz__15m__value` | 186.713561 | 0.595% | 11/13 | Exploratory |
| S14 | `F4Z__flow__flow_confirmation_percentile__15m__value` | 186.733434 | 0.584% | 11/13 | Exploratory |
| S15 | `F4Z__liquidation__liq_side_pressure_rz__15m__value` | 186.755769 | 0.572% | 10/13 | Exploratory |
| S16 | `F4Z__liquidation__liq_acceleration_percentile__5m__15m__value` | 186.761071 | 0.570% | 11/13 | Exploratory |
| S17 | `F4Z__liquidation__liq_peak_age_seconds_percentile__15m__value` | 186.762139 | 0.569% | 11/13 | Exploratory |
| S18 | `F4Z__liquidation__cross_asset_liq_activity_raw_active_severity_rz__15m__value` | 187.052579 | 0.414% | 12/13 | Exploratory |
| S19 | `F4Z__liquidation__cross_asset_liq_activity_raw_active_severity_percentile__15m__value` | 187.056637 | 0.412% | 12/13 | Exploratory |
| S20 | `F4Z__liquidation__cross_asset_liq_activity_raw_active_severity_rz__5m__value` | 187.133024 | 0.372% | 11/13 | Exploratory |

### Addition: scope, comparison and results

The user requested **all unordered pairs of the displayed top 20**, giving `20*19/2 = 190` fits. This superseded the earlier suggestion to combine only the two passing concepts. It was **not** all pairs of the 32 improvers, which would have been 496. Preserve that scope change in the research ledger.

Each pair used intercept + Z1 + M1 + Z2 + M2, the same unpenalized median/absolute-loss fit, fixed nonnegative projection, rows and preprocessing. No products or tuning. Four single-thread workers changed runtime only. The better parent was fixed by its saved full-development standalone MAE; it was not reselected month by month. Both that choice and the top-20 list use exposed development outcomes.

All **190** fits completed with zero failures; all beat M0, **108** beat their better parent and **82** did not. **Three** passed the pair-family development rule. The saved separate verifier reported all-model replay and LP primal/dual certificate checks, with zero new fits.

| Comparison | MAE, log bps | What it establishes |
| --- | ---: | --- |
| M0 | 187.831072 | No-feature median benchmark |
| Global best standalone, S01 | 182.284669 | Best observed standalone average, uncertainty rule failed |
| Best absolute pair, MP0004: S01 + S05 | 182.036977 | 0.247692 better than parent/global best single; 0.136% incremental reduction; 8/13 months better; uncertainty rule failed |
| Largest incremental gain, MP0071: S05 + S06 | 183.411173 | 0.889239 better than its parent; 0.482% reduction; 11/13 months better; still worse than S01 and failed the two-block uncertainty rule |

Three passing pairs have parent-relative reductions of only **0.031916?0.069864 log bps** and all have worse absolute MAE than S01. They reduce to two concept combinations: active liquidation severity plus acceleration, and venue fragmentation plus acceleration. Passing a statistical development rule is not evidence that the improvement is practically useful.

The best absolute pair joins two windows of the **same** concept. It is not two independent discoveries. Removing its two most favorable contribution months reverses its incremental gain; this is a diagnostic, not permission to remove those months from official scores.

### Top 20 pairs by absolute MAE

Aliases refer to the exact standalone fields above. All reductions below are **comparator MAE minus pair MAE**, so positive means improvement and negative means deterioration. This is the opposite sign to a stored `delta = pair - comparator` column. Both rankings are shown because lowest absolute error and largest parent-relative gain answer different questions.

| Absolute rank | Pair | Inputs | Pair MAE | Reduction vs M0 | Reduction vs better parent | Reduction vs global best single | Months better vs parent | Development rule passed? |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | MP0004 | S01 + S05 | 182.036977 | +5.794094 | +0.247692 | +0.247692 | 8/13 | No |
| 2 | MP0020 | S02 + S03 | 182.246302 | +5.584770 | +0.240094 | +0.038368 | 9/13 | No |
| 3 | MP0018 | S01 + S19 | 182.260195 | +5.570876 | +0.024474 | +0.024474 | 9/13 | No |
| 4 | MP0008 | S01 + S09 | 182.270745 | +5.560327 | +0.013925 | +0.013925 | 7/13 | No |
| 5 | MP0017 | S01 + S18 | 182.273869 | +5.557203 | +0.010800 | +0.010800 | 9/13 | No |
| 6 | MP0001 | S01 + S02 | 182.284486 | +5.546585 | +0.000183 | +0.000183 | 7/13 | No |
| 7 | MP0014 | S01 + S15 | 182.290576 | +5.540496 | -0.005907 | -0.005907 | 8/13 | No |
| 8 | MP0006 | S01 + S07 | 182.299512 | +5.531559 | -0.014843 | -0.014843 | 7/13 | No |
| 9 | MP0007 | S01 + S08 | 182.304335 | +5.526736 | -0.019666 | -0.019666 | 7/13 | No |
| 10 | MP0019 | S01 + S20 | 182.324087 | +5.506985 | -0.039418 | -0.039418 | 6/13 | No |
| 11 | MP0009 | S01 + S10 | 182.344299 | +5.486773 | -0.059629 | -0.059629 | 7/13 | No |
| 12 | MP0013 | S01 + S14 | 182.352414 | +5.478658 | -0.067745 | -0.067745 | 6/13 | No |
| 13 | MP0012 | S01 + S13 | 182.357506 | +5.473566 | -0.072836 | -0.072836 | 6/13 | No |
| 14 | MP0011 | S01 + S12 | 182.363022 | +5.468050 | -0.078352 | -0.078352 | 6/13 | No |
| 15 | MP0010 | S01 + S11 | 182.367386 | +5.463686 | -0.082717 | -0.082717 | 6/13 | No |
| 16 | MP0016 | S01 + S17 | 182.372725 | +5.458346 | -0.088056 | -0.088056 | 3/13 | No |
| 17 | MP0015 | S01 + S16 | 182.383015 | +5.448057 | -0.098346 | -0.098346 | 4/13 | No |
| 18 | MP0005 | S01 + S06 | 182.421855 | +5.409216 | -0.137186 | -0.137186 | 6/13 | No |
| 19 | MP0036 | S02 + S19 | 182.436887 | +5.394184 | +0.049509 | -0.152218 | 9/13 | No |
| 20 | MP0035 | S02 + S18 | 182.446305 | +5.384766 | +0.040091 | -0.161636 | 9/13 | No |

### Top 20 pairs by incremental gain over the better parent

The best parent is fixed within each pair. A positive parent-relative reduction can coexist with negative improvement over the global best single. Do not promote a pair merely because it improved a weak starting point.

| Incremental rank | Pair | Inputs | Pair MAE | Reduction vs better parent | Parent-relative reduction % | Reduction vs global best single | Months better vs parent | Development rule passed? |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | MP0071 | S05 + S06 | 183.411173 | +0.889239 | 0.482% | -1.126503 | 11/13 | No |
| 2 | MP0056 | S04 + S06 | 183.510386 | +0.609664 | 0.331% | -1.225717 | 9/13 | No |
| 3 | MP0038 | S03 + S04 | 182.741888 | +0.335159 | 0.183% | -0.457218 | 5/13 | No |
| 4 | MP0086 | S06 + S07 | 184.130582 | +0.315431 | 0.171% | -1.845913 | 11/13 | No |
| 5 | MP0088 | S06 + S09 | 184.131853 | +0.314160 | 0.170% | -1.847184 | 10/13 | No |
| 6 | MP0087 | S06 + S08 | 184.147765 | +0.298248 | 0.162% | -1.863096 | 10/13 | No |
| 7 | MP0098 | S06 + S19 | 184.148176 | +0.297838 | 0.161% | -1.863506 | 11/13 | No |
| 8 | MP0097 | S06 + S18 | 184.150073 | +0.295940 | 0.160% | -1.865404 | 10/13 | No |
| 9 | MP0099 | S06 + S20 | 184.167901 | +0.278113 | 0.151% | -1.883231 | 11/13 | No |
| 10 | MP0187 | S17 + S20 | 186.508585 | +0.253554 | 0.136% | -4.223915 | 11/13 | No |
| 11 | MP0184 | S16 + S20 | 186.512780 | +0.248291 | 0.133% | -4.228111 | 11/13 | No |
| 12 | MP0004 | S01 + S05 | 182.036977 | +0.247692 | 0.136% | +0.247692 | 8/13 | No |
| 13 | MP0039 | S03 + S05 | 182.831968 | +0.245079 | 0.134% | -0.547299 | 6/13 | No |
| 14 | MP0175 | S14 + S20 | 186.489500 | +0.243934 | 0.131% | -4.204831 | 11/13 | No |
| 15 | MP0020 | S02 + S03 | 182.246302 | +0.240094 | 0.132% | +0.038368 | 9/13 | No |
| 16 | MP0091 | S06 + S12 | 184.219175 | +0.226838 | 0.123% | -1.934506 | 9/13 | No |
| 17 | MP0093 | S06 + S14 | 184.230859 | +0.215154 | 0.117% | -1.946190 | 10/13 | No |
| 18 | MP0145 | S10 + S20 | 186.461290 | +0.210927 | 0.113% | -4.176620 | 11/13 | No |
| 19 | MP0094 | S06 + S15 | 184.239263 | +0.206750 | 0.112% | -1.954594 | 8/13 | No |
| 20 | MP0180 | S15 + S20 | 186.549859 | +0.205910 | 0.110% | -4.265189 | 11/13 | No |

### What this adds to the research process

**Finding:** the supplied X contains some observed association with 48h endpoint magnitude under median regression. These results differ from the extreme-probability results because the questions, estimands and losses differ. They do not establish that liquidation features predict direction, path excursions, every horizon, or net trading returns.

**Current decision:** keep the best standalone as an exploratory comparison benchmark, retain the three passing standalone fields as conditional development candidates, and record that addition has not established a reliable upgrade over the strongest single. Do not automatically open an interaction/tree search. A tiny parent-relative gain is not an economically meaningful improvement just because an interval excludes zero.

**Updated accounting:** the closed extreme branch remains 2,772 fits; magnitude has 256 standalone + 190 pairs = **446 fits**. Combined recorded feature-model fits are **3,218**, comprising 3,216 primary fits and two earlier audit refits. This update itself adds zero fits. Do not erase the old branch totals when extending the research history.

Evidence: magnitude `02_STANDALONE_RESULT.md`, `03_ADDITIVE_PAIRS_RESULT.md`; standalone run `standalone256_20261031T095717_608161Z`; pair run `top20_20261031T100905_374149Z`. Full scoreboards retain exact identifiers, all failures/skips and unrounded scores. The top-20 tables here are reporting extracts, not another editable source inventory.

## 4. What carries forward, and what must change

Keep exact source/target definitions, matched comparator rows, training-only transforms, chronological purging, simple baselines, incremental comparisons, all failed attempts, saved predictions, replay checks and the ten-minute active-work log.

Change the selection design for the new question:

1. **Do not seed every new task with the 71 BTC 48h extreme improvers.** They were selected for one target on exposed outcomes. That would bias a direction, timing or different-horizon study toward the old question.
2. **Do not call the old history untouched.** Model fitting, feature screening, representative selection, interpreting plots and choosing the next experiment all count as exposure.
3. **Do not treat one stage's bootstrap bands as correction for the complete adaptive search.** Features, horizons, targets, assets and variants all enlarge the search.
4. **Do not require a standalone win as a mathematical prerequisite for joint value.** The standalone map answers a restricted question. Complementary or interaction-only information can be missed.
5. **Do not force the ladder singles → pairs → interactions → trees.** Use the simplest model suited to the estimand. An additional model family is a separately budgeted comparison.
6. **Do not automatically reject a rare-event forecast for concentrating in stress months.** Report dependence on those episodes and seek replication. The old gates remain the historical decisions; new gates must be justified prospectively rather than loosened after a failure.
7. **Distinguish numerical correctness from generalization.** Hashes and replay do not certify exchange latency, economic usefulness or independent review.

The frozen per-stage contracts were useful. They were not preregistration of the entire research sequence before historical outcomes were seen. Tree and ridge follow-ups remain additional branches, even if individually bounded.

## 5. The new automated question and scope

**Question:** for a declared asset, outcome and horizon, does a measurement available at time t reduce later predictive loss relative to the target-appropriate no-feature baseline, and does its value add information beyond its missingness pattern?

The desired output is a **feature/concept × outcome × horizon map**, not one universal winner.

Proposed staged scope:

- Phase A: BTC, all ten packaged horizons, seven canonical outcomes plus signed log return: **80 task cells**.
- Phase B: repeat the frozen procedure separately for ETH and SOL: **240 asset-task cells in total**, not 240 independent markets.
- Signed simple returns are a reporting conversion, not another discovery axis. Derived range and endpoint-threshold events are excluded from the first matrix. The historical 48h/500-log-bps event remains an archived benchmark.
- Proposed origin cadence: the existing 30-minute UTC grid, phase zero, for continuity and bounded compute. This undersamples possible 15-minute entries but does not alter canonical label paths. A cadence change is a separate version, not an automatic search.
- Initial eligible universe: the registered 303 F4 and 522 F4Z values, tracked as separate layer/transform variants. Eligibility is rechecked on each training fold; the old 256/71 lists are not universal eligibility lists.
- Group results by parent sensor/concept and retain exact lookback/transform identities. Many related windows are not separate discoveries. A shared concept name does not imply independence.

This is a plan for automation, not permission to launch all combinations now. Before execution, compile and save the exact task manifest and compute budget. With 825 features, 80 BTC cells and four assessment folds, a simple one-feature screen alone can reach **264,000 fits**, before missingness controls or any tuning; across three assets it can reach **792,000**. Eligibility will reduce counts. Do not hide that expansion behind a small loop or multiply it again by default model grids.

## 6. Match the model and baseline to each outcome

Proposed first-pass families are deliberately simple. These are new task specifications; they do not retroactively change earlier MAE or binary results.

| Task | Forecast object | Constant baseline | First-pass candidate | Primary loss |
| --- | --- | --- | --- | --- |
| Direction | Probabilities for down/flat/up | Training class proportions with fixed smoothing | Regularized multinomial logistic | Multiclass log loss |
| Extreme order | Probabilities for low-first/same-time/high-first | Training class proportions with fixed smoothing | Regularized multinomial logistic | Multiclass log loss |
| Signed log return | Conditional mean signed return | Training mean; also report zero-return baseline | Ridge linear regression | MSE |
| Net/up/down magnitude, separately | Conditional median magnitude | Training median of the respective field | Median regression with predetermined nonnegative prediction clipping | MAE |
| Time to high/low, separately | Conditional mean fraction of horizon | Training mean of `time/H` | Ridge linear regression with predetermined [0,1] clipping | MSE on `time/H` |

For magnitude, retain median/MAE continuity with the completed branch and report error in log bps. For signed return, report RMSE in log bps alongside MSE. For timing, also convert predictions back to minutes; time zero is retained. Brier is secondary for classification, using a declared multiclass convention. A conditional-mean magnitude/MSE track would be a separate prospective estimand with a training-mean baseline; do not switch between mean/MSE and median/MAE according to which result looks better.

Freeze one regularization specification per task family for the initial mapping pass; the magnitude continuation explicitly uses the existing unpenalized median fit unless a separately named comparison changes it. Record solver, objective scaling, treatment of the intercept, seed, maximum iterations, tolerances and clipping. Any later tuning uses purged inner folds only. Do not borrow the old winning C=0.001 as a universal setting.

For categorical outcomes, keep the declared three-class vocabulary, including exact flat/same-time. Specify smoothing and handling of a class absent from training before running; unsupported sparse-class tasks must be flagged, never silently converted to binary or given infinite losses without explanation. Do not assume the BTC 48h absence of exact flats holds at other horizons or assets.

The historical endpoint-extreme probability task is not `extreme_order`. Their names must remain distinct in code and reports.

## 7. One repeatable experiment procedure

### A clean way to explore different questions

Exploration is allowed; the control is that **each new question has its own declared outcome, baseline, loss, population and budget**. Failure of an extreme-probability model does not imply failure of a magnitude model. Success on magnitude does not imply success on direction. Do not retroactively describe either as a prediction of every aspect of Y.

Before each task, write one line: `asset / target and estimand / horizon / baseline / primary loss / candidate universe / selection procedure / maximum fits / stop rule`. Keep the extreme branch parked while recording magnitude, direction and path-outcome branches separately. A change requested after inspecting results is a new exploratory decision, not evidence the entire search was fixed in advance.

For every task or addition, distinguish four questions:

1. **Does it beat the no-feature baseline?** Compare matched losses, not just correlation, a coefficient sign or rank separation.
2. **Does it use changing feature values?** Check saturation, constant/narrow forecasts and missingness controls. A better constant can reflect a level shift. Any adaptive constant benchmark must be specified and fitted only from labels already matured at its update time; never fit the development median as a deployable comparator.
3. **Does added complexity beat both its own parent and the strongest simpler comparator?** Report absolute and incremental rankings separately. A family-adjusted parent-relative result does not automatically validate the global-comparator result.
4. **Is the gain reliable and worth retaining?** Assess its size, support, dependence, concentration, selection history and eventual unseen replication. Define a practical-effect criterion before the next run rather than discovering one after a tiny result passes.

Always print a concise numerical results summary when a run finishes: attempts/successes/failures, baseline, best absolute score, incremental gain, uncertainty status, constant/missingness flags and a clear decision. When the user requests top 20, show all 20 with an explicit ranking criterion, exact field mapping, units and sign convention; a file link alone is not the requested results summary.

The full cross-target map remains a discovery study with one shared search ledger. Do not stop searching only when a favorable cell appears, omit negative cells, reset multiplicity at each branch, or assume a new horizon is an untouched test. The package's Y definitions remain fixed throughout.

### A. Freeze the batch before reading new outcome summaries

Write one experiment contract with source identities, target formulas, assets, horizons, cadence, date bounds, exposure status, candidate universe, status allowlist, feature qualification, fold definitions, baseline, model family, missingness handling, score convention, comparison rule and maximum fits. Include failures and reruns in the budget.

Resolve the exact chronological fold dates from source availability before outcome-driven selection. Existing Q1–Q4 2023 windows can support a replay study, but previously inspected history is still development. Do not present a newly partitioned version of it as independent confirmation.

### B. Load exact aligned views

Use Handoff 2's `Dataset` with Handoff 1 supplied by its actual path. Join X to Y by exact asset/time keys; include horizon in Y keys. Apply the declared cadence to origins only. Keep one-minute price paths intact.

Use `label_status == VALID` and the declared binding policy to establish a target/horizon population. Save every excluded key with its reason. Do not exclude a 15-minute outcome because the 300-hour outcome is unavailable. Do not let an unused feature determine the row population.

Within each asset/target/horizon/fold, M0 and all compared models use identical eligible rows. Across horizons, populations can differ because of label maturation and missing paths. Report these differences. If a common-support comparison is needed, declare the intersection before scoring rather than choosing the most favorable sample later.

### C. Protect chronology at every level

For a training origin t, require its label to be available by the model's fit time and its endpoint to be **strictly earlier than the next validation block's first origin**. Purge by H: 15 minutes for a 15-minute label, 300 hours for a 300-hour label. Never reuse a fixed 48-hour gap across the entire matrix.

Use expanding or rolling chronological assessment folds as frozen in the contract. At each boundary, fit preprocessing, feature selection, concept representatives, hyperparameters and any states using only that fold's earlier training information. If choosing hyperparameters or a compact feature set, perform that choice inside further purged inner folds. Freeze the selected pipeline before predicting the outer block.

A feature selected using the old full-development rankings cannot become an honest historical training-only selection merely because its coefficients are refitted on earlier rows. New retrospective folds help diagnose the procedure; they do not erase the researcher's prior exposure.

### D. Qualify X without looking for favorable outcomes

Check exact formula, units, lookback, orientation, known availability, validity, source status, normalization anchors, training coverage and variation. F4Z is not automatically superior to F4. Keep `PROVISIONAL` distinct from approved; use an explicit status allowlist and normalization minimum, with 1,000 anchors as the existing documented convention unless a new contract changes it.

Reject critical timing failures. Mark unresolved lineage explicitly. Coverage thresholds use training data and are fixed in advance. Record constants, all-missing columns, insufficient history and convergence failures as outcomes, not disappearing rows.

For each continuous input: fit median imputation and standardization on that fold's training rows; preserve a missingness flag. No full-history scaling, backward filling, historical state cuts or full-history percentile fitting. Standardization does not repair a noncausal source feature.

### E. Compare value and availability information

For each admitted standalone feature, compare on matched rows:

```text
M0: target-appropriate constant forecast
M_missing: missingness indicator only
M_value: standardized value + missingness indicator
```

Use the same task-family settings and chronology. Compare `M_value - M0` and `M_value - M_missing`. When the missingness flag is constant, the control can reduce to M0; record reuse rather than count a nonexistent fit. These are predictive ablations, not causal attribution.

Save full-row predictions and losses first. Both-valid/all-valid subsets, missingness patterns and source periods are secondary diagnostics; they never replace the declared population. A one-feature win establishes only simple standalone information under this model specification. Failure does not prove no joint or nonlinear information exists.

### F. Summarize without picking the prettiest cell

Use paired loss differences: `loss(candidate) - loss(comparator)`, negative meaning improvement. Report the magnitude, uncertainty, row count, coverage, time blocks and all failures. Relative skill `1 - candidate_loss/baseline_loss` can aid comparison within a task; do not compare raw log loss with raw MSE or claim equal economic meaning across tasks. Mark relative skill undefined when baseline loss is zero.

Use chronological block resampling of paired losses. Freeze a horizon-aware block rule and a longer-block sensitivity check before scoring. As an initial design proposal, consider blocks of at least `max(7 days, 2H)` plus twice that length, subject to sufficient available blocks; this is not a universal guarantee of adequate dependence capture. If blocks are too few, mark uncertainty insufficient rather than shrinking blocks to manufacture precision. The 1,008-day release contains only about 80 nonoverlapping 300-hour windows per asset before splits and exclusions.

When comparing features or horizons, use aligned calendar resamples where possible to preserve their dependence. For any pooled multi-asset statistic, resample assets' shared calendar blocks together; BTC, ETH and SOL are correlated. Monthly rows and overlapping minute labels are not independent replications.

Maintain one visible search ledger for all features, targets, horizons, assets, model settings and restarts. Specify the multiplicity family and method before calling any result significant. Fixed-family simultaneous bands are descriptive for that declared family; they do not automatically cover earlier adaptive selection. A confirmatory claim requires an assessment design that includes selection, and genuinely unseen evidence.

### G. Verify, classify and stop

Check unique keys, exact comparator alignment, no train/validation label overlap, target formulas, fitted-transform provenance, finite predictions, probability normalization, model replay, metric recomputation and artifact hashes. Check stored CV selection against predictions. Separate implementation failures from predictive failures.

Use explicit statuses: `SKIPPED_QUALIFICATION`, `FAILED_IMPLEMENTATION`, `INSUFFICIENT_SUPPORT`, `NO_OBSERVED_IMPROVEMENT`, `EXPLORATORY_IMPROVEMENT`, or `DEVELOPMENT_CANDIDATE_UNCONFIRMED`. A documented evidence rule must distinguish the last two before the batch runs. A result passing development checks does not get a `VALIDATED_ALPHA` label.

Do not automatically launch pairs, interactions, trees, new thresholds or another grid from a favorable map. If a compact multivariate follow-up is justified, freeze one separate question, its training-only selection procedure and its fit budget. An all-feature regularized comparison may test joint information without requiring every constituent to win alone; that is a distinct experiment, not a conclusion from the standalone screen.

## 8. Automation shape and saved outputs

The following is workflow pseudocode, not a runnable script or a completed result:

```text
freeze batch contract + task manifest + fit budget
verify source identities and measurement contracts
for asset in declared_assets:
    for horizon in packaged_horizons:
        for target in declared_targets:
            freeze target-specific eligible keys and chronological folds
            for fold in folds:
                build earlier training and later assessment views; purge by H
                qualify candidates from training information
                fit target-specific M0
                for eligible feature in declared universe:
                    fit training-only preprocessing
                    fit/reuse missingness control
                    fit the one-feature task model
                    save all matched predictions, failures and fit counts
            verify saved forecasts and aggregate paired losses
            record coverage, uncertainty, stability and evidence status
publish the complete map, including negative and unsupported cells
stop at the declared batch budget
```

Use the simplest scripts and files sufficient for this workflow. No database, vector store, agent orchestration or background service is needed. Chunk source reads, cache only hash-bound views, and prevent memory limits from changing which rows are scored. Any runtime fix preserves the scientific contract or becomes an explicitly new version.

The consolidated result table should contain at least:

```text
run_id, asset, target_id, horizon_minutes, origin_stride_minutes,
feature_id, parent_concept, family, layer, lookback, transform,
fold_id, training_rows, assessment_rows, event/class counts when relevant,
training_coverage, assessment_coverage, missing_pattern,
baseline_loss, missingness_control_loss, candidate_loss,
delta_vs_baseline, delta_vs_missingness_control, relative_skill,
time_block_count, uncertainty_method, interval_bounds,
monthly_results_path, source_binding_results_path,
fit_status, evidence_status, attempted_fits, artifact_paths
```

Produce separate panels for each outcome, with horizons as columns and features/concepts as rows. Show uncertainty/support flags beside scores. Preserve a full machine-readable table rather than only a top-ten image. A missing cell must explain whether it was skipped, failed or unsupported; it must not be mistaken for no predictive effect.

Minimum completed run artifacts: frozen contract and input manifest, eligible/excluded keys, fold manifests, qualification records, preprocessing and models, row predictions, losses, uncertainty and diagnostics, verification receipt, decision, hashes and append-only progress log. A reproducible fitted run and a statistically convincing result remain different deliverables.

## 9. Holdout, replication and economic meaning

**No untouched final holdout is verified.** Handoff 2's default access window ends with paths by 2024-02-02 UTC, but this thread has already inspected later BTC history through January 2025. A loader access policy controls intentional access; it is not proof of historical secrecy. Record a named policy for expanded research access. Audit ETH/SOL exposure rather than assume those assets are untouched.

Before final confirmation, freeze the entire candidate pipeline: source definitions, asset universe, target/horizon, feature selection, preprocessing, model, training/refitting schedule, uncertainty and decision rule. Evaluate on a verified untouched later period or prospectively collected data. Confirm labels have fully matured before reading them. Do not use the withheld 112 days unless actually supplied and their prior exposure is audited. Their finite length is particularly restrictive for 200/300-hour targets.

The intended interpretation is task-specific: direction and signed return concern directional/expected-return information; excursions, magnitude and timing describe aspects of future path risk. These are inputs to investment decisions, not positions or strategies. Cost, expected return and risk remain parallel inputs to portfolio construction. Reference-price prediction alone establishes neither executable profit nor net alpha; execution delay, spread, fees, funding, slippage, overlapping exposure and portfolio rules belong to a separately defined later study.

## 10. Documentation and change discipline

During active work, record a UTC timestamp at least every ten minutes, plus each result, failure, decision and change of direction. Include the question, exact action, source/run paths, actual fit count, result, interpretation, open issue and next step. This is an active-work practice, not a scheduled background task.

Freeze a run before fitting; preserve failed attempts; never overwrite historical predictions or rewrite an expected hash to hide a mismatch. A changed target, horizon policy, cadence, feature universe, threshold, fold, metric or model setting must be visible in the experiment history. Reopening a stopped branch requires a new explicit research question, not silently extending the old one.

This document is a process note and evidence guide, not a replacement source catalogue or another editable source inventory. Existing manifests remain authoritative for the packages, and existing run artifacts remain authoritative for numerical results.

## 11. Evidence locations and limits of this review

### Source-package passages

Within **Handoff 1**: `READ_ME_FIRST.txt` (coverage, withheld days, price/join semantics); `reference/F1_PRICE_FORMULA.json` (VWAP definition); `PACKAGE_MANIFEST.json` (release metadata). Its one-hour helper is historical; the canonical multi-horizon Y comes from Handoff 2.

Within **Handoff 2**, all paths are under `HANDOFF_2_CANONICAL_XY_V1_1/`:

- `START_HERE.md` and `RELEASE_MANIFEST.json`: scope, counts, versions and review status.
- `team_tools/schema.json` and `team_tools/return_schema.json`: exact fields, keys, codes, horizons and null rules.
- `team_tools/TEAM_DATASET_SPEC.md`, sections 3–7: formulas, timing, signed returns, validity, binding change and feature semantics; sections 8 and 10: loader and chronology.
- `team_tools/build_dataset.py`: path formulas and Decimal oracle; `team_tools/canonical_dataset.py`: access, maturity and binding gates.
- `research_access.json`: default development access policy; `qa/counts.json`: per-asset/horizon label counts.
- `HANDOFF_2_MANIFEST.json`: delivered-file hashes. Manifest QA claims are not a fresh full audit by this document.

### Research history

The historical workflow root is `C:\Users\REDACTED_USER\Desktop\MFT ML\REPEATABLE PLAN\48H_ML`.

- [Baseline contract](</C:/Users/REDACTED_USER/Desktop/MFT ML/REPEATABLE PLAN/48H_ML/baseline_contract.json>)
- [Standalone decision](</C:/Users/REDACTED_USER/Desktop/MFT ML/REPEATABLE PLAN/48H_ML/06_TOURNAMENT_DECISION.md>)
- [Additive decision](</C:/Users/REDACTED_USER/Desktop/MFT ML/REPEATABLE PLAN/48H_ML/08_ADDITIVE_PAIR_DECISION.md>)
- [Locked interaction path](</C:/Users/REDACTED_USER/Desktop/MFT ML/REPEATABLE PLAN/48H_ML/09_LOCKED_PAIR_PATH.md>)
- [Compact tree result](</C:/Users/REDACTED_USER/Desktop/MFT ML/REPEATABLE PLAN/48H_ML/10_COMPACT_TREE_RESULT.md>)
- [Broader ridge result](</C:/Users/REDACTED_USER/Desktop/MFT ML/REPEATABLE PLAN/48H_ML/11_BROADER_RIDGE_RESULT.md>)
- [Final audit and parking decision](</C:/Users/REDACTED_USER/Desktop/MFT ML/REPEATABLE PLAN/48H_ML/12_FINAL_AUDIT_AND_DECISION.md>)

Exact run identifiers: `M1_extreme_20261031T081946_513623Z`; `M1_recheck_20261031T082237_869616Z`; `extreme_all522_20261031T083619_479454Z`; `additive71_20261031T085407_882154Z`; `locked_pair_20261031T090729_719529Z`; `compact12_20261031T092024_478891Z`; `concept31_20261031T093520_429829Z`; final audit `logistic12_20261031T095052_061640Z`.

Magnitude records: `C:\Users\REDACTED_USER\Desktop\MFT ML\REPEATABLE PLAN\48H_ML\magnitude`, especially `02_STANDALONE_RESULT.md`, `03_ADDITIVE_PAIRS_RESULT.md`, `tournaments/standalone256_20261031T095717_608161Z/scoreboard.csv`, and `pair_tournaments/top20_20261031T100905_374149Z/pair_scoreboard.csv`.

Earlier checks in this conversation independently replayed the interaction comparisons and all 25 ridge models without refitting. Other historical findings here are drawn from the saved reports and cited artifacts; this document does not claim to have independently rerun the entire 2,772-fit history. The final audit's machine-readable result was inspected directly. No external practitioner has approved this work.

Methodological background: [Cawley and Talbot, model-selection overfitting](https://www.jmlr.org/papers/v11/cawley10a.html); [Bailey et al., probability of backtest overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf); [scikit-learn, leakage and preprocessing](https://scikit-learn.org/stable/common_pitfalls.html). These support the separation of development, selection and confirmation; they do not endorse these particular results or every proposed rule.

**Next implementation deliverable:** a frozen 80-cell BTC task manifest and a small verified runner implementing the declared task families, folds, baselines, missingness controls and complete scoreboard. Do not launch the full matrix until the measurement checks and bounded implementation smoke checks pass. Record it as a new Y-by-horizon discovery study, with the old 48-hour extreme search preserved as completed and unconfirmed.


</details>

### Extreme final audit

[Original document](<C:/Users/REDACTED_USER/Desktop/MFT ML/REPEATABLE PLAN/48H_ML/12_FINAL_AUDIT_AND_DECISION.md>) ? SHA-256 `e670d8fbde0b52c5ff41ab063116f5e4f1cc1f074c614b23eb21b4375ec9fb3f`

<details>
<summary>Read complete preserved document</summary>

# Final audit: predictive value not established; park this model search

Updated 2026-10-31T09:53:13.823360+00:00. **Zero new fits.**

The saved twelve-feature logistic has the lowest observed development log loss among the models tried, and it improves on M0 and the best pair in the aggregate. Its final audit nevertheless fails the declared uncertainty and concentration requirements. **Stable predictive value at this BTC 48h +/-500-log-bps endpoint target is not established. Park the current model-search branch and preserve all evidence.**

This is a decision about the present evidence and candidate pipeline. It does not prove the target is intrinsically unpredictable or that all possible models would fail. The twelve-feature model remains a saved development candidate; it is not promoted as a validated forecast or trading signal. No untouched holdout has been verified.

## What was frozen and checked

The audit reuses the saved twelve-feature C=1 L2 logistic, M0 and the saved best pair. There are no new fits, preprocessing estimates, thresholds, calibration steps or feature changes. Primary scores retain all 18,481 development observations. All twelve values being valid is a matched diagnostic subset, not a new test set.

Before calculating these diagnostics, the [audit contract](final_audits/logistic12_20261031T095052_061640Z/CONTRACT.json) fixed the following criteria: both metrics improve against both comparators on full and all-valid rows; a majority of full calendar months improve log loss against both; both metrics have negative paired uncertainty upper bounds at both 7-day and 14-day block lengths; and both metrics retain improvement under March, March+August and the two largest contribution-month exclusions against both comparators.

The first two criteria pass. The uncertainty and concentration criteria fail. This audit cannot undo earlier model selection on these same outcomes. The intervals are conditional development diagnostics, not corrections for the entire research search.

## Average performance and missing values

Lower log loss is better:

| Population | Rows | M0 | Best pair | Twelve-feature logistic |
| --- | ---: | ---: | ---: | ---: |
| Full development | 18,481 | 0.449294381 | 0.444267760 | 0.443678457 |
| All twelve values valid | 17,318 | 0.450094507 | 0.444319579 | 0.444221205 |
| At least one missing | 1,163 | 0.437379858 | 0.443496137 | 0.435596510 |

On full development, Brier is 0.136616599 for the twelve-feature model, versus 0.138122795 for M0 and 0.136807454 for the pair. Both metrics also improve on the all-valid subset. The average improvement is therefore not confined entirely to missing-value rows.

However, **approximately 84.4% of the total incremental log-loss gain over the pair comes from the 1,163 rows with at least one missing feature**. Those rows are only 6.29% of development. On the 17,318 all-valid rows, the twelve-feature model improves log loss over the pair by just 0.000098374.

That concentration matters. It is not proof that missingness flags alone caused the improvement: other observed feature values and fitted coefficients also differ between the models. The supported conclusion is that much of the incremental advantage appears on rows with missing inputs. [All subset scores](final_audits/logistic12_20261031T095052_061640Z/subset_scores.csv), [feature coverage](final_audits/logistic12_20261031T095052_061640Z/feature_coverage.csv) and [saved audit predictions with original missing flags](final_audits/logistic12_20261031T095052_061640Z/audit_predictions.parquet) preserve the calculation.

## Monthly stability and concentration

The model beats M0 on log loss in **7 of 13 months**, and also beats the best pair in **7 of 13 months**. This passes a simple majority-month check, but the size of gains and losses still matters.

Against M0, March and August 2024 are the two largest favorable contribution months. Removing both leaves a tiny log-loss gain and makes Brier slightly worse. Against the pair, the largest favorable contribution months are December and July; removing both reverses improvement in both metrics. The main sources of improvement differ depending on the comparator.

Negative differences below favor the twelve-feature model. Exclusions are diagnostic only; the original model and primary evaluation remain unchanged.

| Comparator | Months excluded | Remaining rows | Log-loss difference | Brier difference |
| --- | --- | ---: | ---: | ---: |
| M0 | None | 18,481 | -0.005615923 | -0.001506196 |
| M0 | 2024-03 | 17,089 | -0.003029304 | -0.000776906 |
| M0 | 2024-03, 2024-08 | 15,601 | -0.000243401 | +0.000039367 |
| M0 | 2024-03, 2024-08 | 15,601 | -0.000243401 | +0.000039367 |
| best_pair | None | 18,481 | -0.000589303 | -0.000190855 |
| best_pair | 2024-03 | 17,089 | -0.001380921 | -0.000427557 |
| best_pair | 2024-03, 2024-08 | 15,601 | -0.001219561 | -0.000367888 |
| best_pair | 2024-12, 2024-07 | 15,505 | +0.000733585 | +0.000185485 |

Removing March alone does not reverse this model's M0 gain, unlike the earlier pair-versus-parent finding. The March+August check is the relevant failure here. The table also shows that excluding March and August does not remove its incremental advantage over the pair; that advantage depends on different months. [Monthly scores](final_audits/logistic12_20261031T095052_061640Z/monthly_scores.csv) and [full/all-valid concentration calculations](final_audits/logistic12_20261031T095052_061640Z/concentration.json) keep these distinctions explicit.

## Uncertainty

The audit uses the same paired chronological 7-day and 14-day blocks, 2,000 resamples per length and seed rule 48+days. Each comparison resamples matched observations, preserving local dependence from overlapping 48h targets. The block lengths do not prove that every source of market dependence is captured.

Every full-sample interval below includes zero:

| Comparator | Metric | Block length | Observed difference | Nominal 95% interval |
| --- | --- | --- | ---: | --- |
| M0 | log_loss | 7 days | -0.005615923 | [-0.013810, +0.002039] |
| M0 | brier | 7 days | -0.001506196 | [-0.003837, +0.000661] |
| M0 | log_loss | 14 days | -0.005615923 | [-0.015251, +0.002809] |
| M0 | brier | 14 days | -0.001506196 | [-0.004237, +0.000902] |
| best_pair | log_loss | 7 days | -0.000589303 | [-0.003156, +0.001856] |
| best_pair | brier | 7 days | -0.000190855 | [-0.000900, +0.000475] |
| best_pair | log_loss | 14 days | -0.000589303 | [-0.003238, +0.001947] |
| best_pair | brier | 14 days | -0.000190855 | [-0.000947, +0.000516] |

Thus even the unadjusted conditional intervals do not establish an improvement over M0 or the pair. This is not evidence of exactly zero benefit; it is insufficient precision to claim a reliable gain. [All intervals, including both missingness subsets](final_audits/logistic12_20261031T095052_061640Z/paired_uncertainty.csv), are preserved.

## Verification and decision

The audit first rebuilt original training preprocessing and replayed the saved twelve-feature and pair coefficients. A separate verification script then matched every saved comparator, recomputed subset scores with a separate metrics implementation, checked monthly and concentration calculations, independently reconstructed all uncertainty intervals and reproduced the final decision. [Verification receipt](final_audit_verification/logistic12_20261031T095052_061640Z.json): PASS; maximum interval discrepancy below 1e-15. This is a separate code-path check by the assistant, not independent-person signoff.

The 48h fit ledger is unchanged at **2,772 feature-model fits: 2,770 primary fits and two earlier audit refits**. No model was fitted by this audit or its verification.

- [Audit code](../tools/audit_48h_logistic12.py) and [separate verifier](../tools/verify_48h_logistic12_audit.py).
- [Machine-readable result](final_audits/logistic12_20261031T095052_061640Z/RESULT.json), [frozen contract](final_audits/logistic12_20261031T095052_061640Z/CONTRACT.json) and [artifact hashes](final_audits/logistic12_20261031T095052_061640Z/ARTIFACT_HASHES.json).
- [Workflow state](research_state.json) and [timestamped process log](experiment_log.jsonl).

**Decision: predictive value not established; park model search for the current 48h extreme target.** Preserve M0, the twelve-feature candidate and all negative results. Do not continue optimizing this development period or relabel any of it as untouched. No new horizon, target, model branch or deployment is initiated by this decision. The separate direction and magnitude baselines remain preserved.


</details>

### Magnitude standalone results and top candidates

[Original document](<C:/Users/REDACTED_USER/Desktop/MFT ML/REPEATABLE PLAN/48H_ML/magnitude/02_STANDALONE_RESULT.md>) ? SHA-256 `8f116e286c644e07cfdd5506aa45b9170a217b23f4ccb0e725f74b4de0e5d66b`

<details>
<summary>Read complete preserved document</summary>

# 48h magnitude: standalone tournament complete

Updated 2026-10-31T10:03:59.328059+00:00. All **256 qualified features** were fitted and separately verified; zero failures. All 522 original fields are accounted for, including 266 qualification skips.

**32 features lower development MAE versus M0. Three pass the frozen development-lead rule, representing two distinct concepts.** These are development candidates, not untouched confirmation. The extreme-target branch is closed and its artifacts remain pinned by [the closure manifest](../EXTREME_BRANCH_CLOSURE.json).

## Baseline and fixed process

Target: absolute 48h endpoint log return, in log bps. M0 predicts the training median **161.461839 log bps** for every row; its development **MAE is 187.831072 log bps**. The forecast and its average error are different numbers.

Each candidate receives the same unpenalized median-regression model with one standardized, training-median-filled feature and its missing indicator. Coefficients minimize absolute training error. Predictions are projected to nonnegative values under a rule fixed before evaluation. This matches the median/MAE target. No mean/RMSE or classification objective is mixed in.

The original 29,615 training rows, 96-row boundary purge and all 18,481 development rows are unchanged. Missing inputs do not remove rows. Existing input qualification is reused; the magnitude candidate list is not restricted to the extreme branch's 71 improvers.

[Step-by-step procedure](01_RESEARCH_PROCESS.md), [executed contract](tournaments/standalone256_20261031T095717_608161Z/CONTRACT.json) and [baseline replay](tournaments/standalone256_20261031T095717_608161Z/BASELINE_REPLAY.json) record the specification. All data remain exposed development; no untouched holdout is verified.

## Best average scores

| Feature | MAE, log bps | Improvement vs M0 | Months better | Result |
| --- | ---: | ---: | ---: | --- |
| `F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__15m__value` | 182.284669 | 2.953% | 10/13 | EXPLORATORY_MAE_GAIN |
| `F4Z__liquidation__liq_activity_severity_historical_active_rate__15m__value` | 182.486396 | 2.845% | 10/13 | EXPLORATORY_MAE_GAIN |
| `F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__1h__value` | 183.077047 | 2.531% | 10/13 | EXPLORATORY_MAE_GAIN |
| `F4Z__liquidation__liq_activity_severity_historical_active_rate__5m__value` | 184.120051 | 1.976% | 8/13 | EXPLORATORY_MAE_GAIN |
| `F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__5m__value` | 184.300412 | 1.880% | 10/13 | EXPLORATORY_MAE_GAIN |
| `F4Z__liquidation__liq_activity_severity_historical_active_rate__1h__value` | 184.446013 | 1.802% | 10/13 | DEVELOPMENT_MAGNITUDE_LEAD |
| `F4Z__liquidation__venue_fragmentation_percentile__15m__value` | 186.626948 | 0.641% | 11/13 | EXPLORATORY_MAE_GAIN |
| `F4Z__liquidation__liq_activity_severity_active_severity_percentile__15m__value` | 186.653197 | 0.627% | 11/13 | EXPLORATORY_MAE_GAIN |

The best average score is **cross-asset liquidation historical active rate at 15m**: MAE **182.284669**, lower by **5.546402 log bps (2.953%)**, with 10/13 months better. It remains exploratory because both family-adjusted uncertainty upper bounds include zero. It retains a 4.454167-log-bps improvement after excluding March and August, and 3.935869 after excluding its own largest contribution months, April and August. These diagnostics do not overcome the failed uncertainty rule.

## Three development leads, two concepts

The frozen rule requires improved MAE, a majority of months better, negative simultaneous uncertainty upper bounds at both 7-day and 14-day block lengths, and nonconstant development predictions. These three fields pass:

| Feature | MAE | Months better | 7-day simultaneous upper bound | 14-day simultaneous upper bound |
| --- | ---: | ---: | ---: | ---: |
| `F4Z__liquidation__liq_activity_severity_historical_active_rate__1h__value` | 184.446013 | 10/13 | -0.492838 | -0.653284 |
| `F4Z__liquidation__liq_fast_slow_lift_rz__15m__1h__value` | 187.314266 | 12/13 | -0.099498 | -0.072602 |
| `F4Z__liquidation__liq_fast_slow_lift_percentile__15m__1h__value` | 187.332386 | 12/13 | -0.093673 | -0.066074 |

The two fast/slow liquidation-lift transforms are the **same concept**, not two independent discoveries. Thus the shortlist contains BTC liquidation historical active rate at 1h and liquidation fast/slow lift at 15m/1h. Their gains remain positive after the declared March/August and largest-two-contributor exclusions. The lift effects are small: approximately half a log bp of MAE improvement.

The BTC activity 1h model's development forecasts vary only from approximately 177.14 to 183.07 log bps. It passes the declared nonconstant rule, but much of its benefit could still be a change in overall forecast level. That distinction matters in later comparisons; this result does not establish strong within-period magnitude discrimination.

## Correction to the original leading screen

The quoted pooled Spearman **+0.194962** for cross-asset liquidation historical active rate at **1h** reproduces. However, its development feature value is **1.0 on every row**. Its median-regression forecast is therefore a constant **194.233 log bps**. It improves MAE to 183.077047, but cannot distinguish future magnitudes within this development period and is not promoted as a development lead. [Screen/saturation check](tournaments/standalone256_20261031T095717_608161Z/ANCHOR_SCREEN_CHECK.json).

## Verification and saved evidence

The separate verifier rebuilt every feature's training preprocessing, replayed all 256 models on all development rows and checked the linear-program dual witnesses for numerical optimality. It also reconstructed both uncertainty families, monthly and missingness results, concentration checks and the source hashes that freeze the extreme branch. [Receipt](verification/standalone256_20261031T095717_608161Z.json): PASS; zero prediction discrepancy and no extra fits.

- [Complete scoreboard](tournaments/standalone256_20261031T095717_608161Z/scoreboard.csv), [all 522 outcomes](tournaments/standalone256_20261031T095717_608161Z/all_522_outcomes.csv), and [concept groups](tournaments/standalone256_20261031T095717_608161Z/concept_scoreboard.csv).
- [Monthly results](tournaments/standalone256_20261031T095717_608161Z/monthly_scores.csv), [present/missing results](tournaments/standalone256_20261031T095717_608161Z/present_missing_scores.csv), [uncertainty](tournaments/standalone256_20261031T095717_608161Z/paired_uncertainty.csv), and [concentration](tournaments/standalone256_20261031T095717_608161Z/concentration.json).
- [Saved models](tournaments/standalone256_20261031T095717_608161Z/models.json), [all predictions](tournaments/standalone256_20261031T095717_608161Z/development_predictions.parquet), [optimizer certificates](tournaments/standalone256_20261031T095717_608161Z/optimizer_certificates.csv), and [hashes](tournaments/standalone256_20261031T095717_608161Z/ARTIFACT_HASHES.json).
- [Runner](../../tools/run_48h_magnitude_standalone.py) and [separate verifier](../../tools/verify_48h_magnitude_standalone.py).

This stage adds exactly **256 fits**. The combined 48h ledger is **3,028 feature-model fits: 3,026 primary fits and two earlier audit refits**. The closed extreme branch remains at 2,772 fits; magnitude has 256. Verification adds zero.

## Next controlled step

The magnitude standalone stage is complete. The defensible addition shortlist starts with the two distinct concepts that passed the development rule. Any pair must beat its better standalone parent on the same rows. Preserve the 15m cross-asset activity model as the strongest average-score comparator, without relabeling its uncertainty result. No pair, interaction or tree fit has yet been made for magnitude. Direction remains queued.


</details>

### Magnitude addition tournament

[Original document](<C:/Users/REDACTED_USER/Desktop/MFT ML/REPEATABLE PLAN/48H_ML/magnitude/03_ADDITIVE_PAIRS_RESULT.md>) ? SHA-256 `e3060703251d63b7cb5b3b4ce9ea5c417b5351058355bf49534934422ec7d9e2`

<details>
<summary>Read complete preserved document</summary>

# Magnitude: all top20 additive pairs completed

Updated 2026-10-31T10:12:05.173936+00:00. **190 unique pairs fitted, zero failures, all separately verified.** Scope is the top20 features printed immediately before the request, not all32 standalone improvers.

## Fixed procedure

1. Freeze the exact top20 manifest and all 190 unordered pair identities before fitting. No self-pairs or reversed duplicates.
2. Reuse each parent's training-only median imputation, mean/std and missing indicator. The model has intercept, Z1, M1, Z2 and M2. No products or tuning.
3. Fit the same unpenalized median regression by minimizing absolute training errors on all29,615 original training rows. Use the same HiGHS interior-point solver and tolerances. Four single-thread workers shorten execution without changing the model objective.
4. Apply the same nonnegative prediction projection and retain every one of the18,481 development rows. The target remains absolute48h endpoint log return in log bps; M0 remains the training median161.461839, with development MAE187.831072.
5. Score MAE against M0, X1, X2, the better single parent and the globally best single. Fix the better parent by its saved full-development MAE; do not select a new parent in each month.
6. Save monthly and both-valid/any-missing scores, correlation/rank/duplicate-flag diagnostics and all predictions. Use the same paired7-day/14-day blocks,2,000 draws and seed48+days. Save nominal intervals for all comparisons and simultaneous bands across190 pairs for the primary best-parent comparison.
7. Check concentration for leading pairs, verify every saved model and LP certificate, and record the result. No interaction or tree is fitted.

[Contract](pair_tournaments/top20_20261031T100905_374149Z/CONTRACT.json), [candidate manifest](pair_tournaments/top20_20261031T100905_374149Z/candidate_top20.csv) and [pair manifest](pair_tournaments/top20_20261031T100905_374149Z/pair_manifest.csv) preserve the exact settings. Pair selection uses exposed development results. These are development comparisons; no untouched holdout is verified.

## Results

| Outcome | Count |
| --- | ---: |
| Requested / successful fits | 190 / 190 |
| Failures | 0 |
| Pairs below M0 MAE | 190 |
| Pairs below their better parent's MAE | 108 |
| Pairs with no incremental MAE gain | 82 |
| Passed the frozen additive development-lead rule | 3 |

Beating M0 is common here because the inputs were selected as standalone improvers. The relevant addition question is whether a pair improves beyond its better parent.

## Lowest absolute MAE

**MP0004: cross-asset liquidation historical active rate15m +5m.** These are two windows of the same concept.

| Forecast | MAE, log bps |
| --- | ---: |
| M0 | 187.831072 |
| 15m single, also globally best single | 182.284669 |
| 5m single | 184.300412 |
| Additive pair | **182.036977** |

The pair improves by **0.247692 log bps, or0.136%, beyond its better parent**. It improves in8/13 months, with both input values valid on every development row. Its simultaneous uncertainty upper bounds remain positive at both block lengths, so it is **EXPLORATORY_INCREMENTAL_GAIN**, not a development lead.

## Largest incremental improvement

**MP0071: cross-asset liquidation historical active rate5m + BTC liquidation historical active rate1h.** MAE improves from184.300412 for its better parent to183.411173: a reduction of0.889239 log bps, or0.482%. It improves in11/13 months and both values are valid on every development row. Its7-day simultaneous upper bound includes zero, so it also remains exploratory.

This pair has a larger incremental gain but a worse absolute score than the best standalone feature. Incremental and absolute rankings answer different questions.

## Pairs passing the development rule

The rule requires an incremental MAE gain, a majority of months better than the fixed parent, nonconstant forecasts and negative simultaneous upper bounds at both block lengths.

| Pair | Features | Pair MAE | Better-parent MAE | MAE reduction | Months better |
| --- | --- | ---: | ---: | ---: | ---: |
| MP0114 | `F4Z__liquidation__liq_activity_severity_active_severity_percentile__15m__value` + `F4Z__liquidation__liq_acceleration_rz__5m__15m__value` | 186.583333 | 186.653197 | 0.069864 | 10/13 |
| MP0125 | `F4Z__liquidation__liq_activity_severity_active_severity_rz__15m__value` + `F4Z__liquidation__liq_acceleration_rz__5m__15m__value` | 186.610895 | 186.657887 | 0.046992 | 10/13 |
| MP0102 | `F4Z__liquidation__venue_fragmentation_percentile__15m__value` + `F4Z__liquidation__liq_acceleration_rz__5m__15m__value` | 186.595031 | 186.626948 | 0.031916 | 10/13 |

These three pairs reduce to **two concept combinations**: active liquidation severity plus acceleration, and venue fragmentation plus acceleration. Their incremental reductions are only0.031916 to0.069864 log bps. All have worse absolute MAE than the globally best single. Passing the uncertainty rule does not make a tiny gain practically meaningful or establish future performance. All three have16,892 both-valid rows; missingness subset diagnostics are saved.

## Concentration diagnostics

Differences below are pair minus its fixed better parent; negative favors the pair. Original evaluation rows and models are unchanged.

| Pair | Months excluded | MAE difference |
| --- | --- | ---: |
| MP0004 | None | -0.247692 |
| MP0004 | 2024-03, 2024-08 | -0.661136 |
| MP0004 | 2024-02, 2024-07 | +0.110013 |
| MP0071 | None | -0.889239 |
| MP0071 | 2024-03, 2024-08 | -1.103875 |
| MP0071 | 2024-09, 2024-07 | -0.527252 |
| MP0102 | None | -0.031916 |
| MP0102 | 2024-03, 2024-08 | -0.027984 |
| MP0102 | 2024-04, 2024-07 | -0.025040 |
| MP0114 | None | -0.069864 |
| MP0114 | 2024-03, 2024-08 | -0.060960 |
| MP0114 | 2024-04, 2024-12 | -0.054505 |
| MP0125 | None | -0.046992 |
| MP0125 | 2024-03, 2024-08 | -0.041043 |
| MP0125 | 2024-04, 2024-07 | -0.037070 |

## Verification and artifacts

The separate verifier rebuilt original preprocessing, replayed every pair and parent, checked190 unique combinations, checked the LP primal/dual witnesses, reconstructed all nominal and primary simultaneous intervals, and verified monthly, missingness and concentration results. [Receipt](pair_verification/top20_20261031T100905_374149Z.json): PASS. Maximum prediction discrepancy is zero; maximum mean duality gap is below1e-12 log bps. No verification refits were made.

- [Incremental scoreboard](pair_tournaments/top20_20261031T100905_374149Z/pair_scoreboard.csv), [absolute scoreboard](pair_tournaments/top20_20261031T100905_374149Z/absolute_scoreboard.csv), [all190 outcomes](pair_tournaments/top20_20261031T100905_374149Z/all_190_outcomes.csv), and [concept groups](pair_tournaments/top20_20261031T100905_374149Z/concept_scoreboard.csv).
- [Monthly scores](pair_tournaments/top20_20261031T100905_374149Z/monthly_scores.csv), [valid/missing subsets](pair_tournaments/top20_20261031T100905_374149Z/valid_missing_scores.csv), [uncertainty](pair_tournaments/top20_20261031T100905_374149Z/paired_uncertainty.csv), and [concentration](pair_tournaments/top20_20261031T100905_374149Z/concentration.json).
- [Models](pair_tournaments/top20_20261031T100905_374149Z/models.json), [predictions](pair_tournaments/top20_20261031T100905_374149Z/development_predictions.parquet), [certificates](pair_tournaments/top20_20261031T100905_374149Z/optimizer_certificates.csv), and [artifact hashes](pair_tournaments/top20_20261031T100905_374149Z/ARTIFACT_HASHES.json).
- [Runner](../../tools/run_48h_magnitude_pairs.py) and [separate verifier](../../tools/verify_48h_magnitude_pairs.py).

This stage adds190 fits. Magnitude now has446 fits; the combined48h ledger is3,218 fits, comprising3,216 primary fits and two earlier audit refits. The closed extreme branch remains unchanged at2,772 fits.

The requested pair tournament is complete. Addition gives some observed gains, but the best absolute pair's incremental improvement remains uncertain and the statistically passing pairs have tiny gains from weaker starting scores. Preserve these distinctions before considering any further complexity. No next model class is run automatically.


</details>

### Room offsets V1

[Original document](<C:/Users/REDACTED_USER/Desktop/MFT ML/REPEATABLE PLAN/48H_ML/magnitude/04_ROOM_CONTEXT_V1.md>) ? SHA-256 `568a0db3b1b9db805f07b5f27bc2aedd11919fa0bd3961f6ad6d50642a10c423`

<details>
<summary>Read complete preserved document</summary>

# 48h magnitude — 1d room context v1

**Decision: adding this 1d room context did not improve the best fast-feature pair. Keep M2; do not advance this version to room interactions.**

| Forecast | Development MAE, log bps | Difference vs M2 |
|---|---:|---:|
| M0: training median | 187.831072 | +5.794094 |
| M2: saved MP0004 pair | 182.036977 | 0 |
| M3 v1: pair + 1d room | 183.870886 | +1.833909 |

M3 is **1.0074% worse than M2**, improving only **4/13 months**. Its improvement over M0 does not establish incremental room value: M2 already does better. No M4, 4h context, threshold search or tree was run.

## Version history

| Version | What changed | Fits | Status |
|---|---|---:|---|
| M0 | Training median magnitude, original baseline contract | baseline | Frozen |
| Standalone v1 | 256 qualified one-feature median regressions | 256 | Preserved |
| Pair v1 / M2 | 190 pairs of displayed top20; MP0004 lowest MAE | 190 | Preserved |
| Room v1, proposed plan | Fixed-pair residual room adjustment discussed | 0 | Superseded by user before fitting |
| Room v1, build attempt 1 | Joint M3 contract saved; source loading stopped | 0 | Failed before fit; receipt preserved |
| Room v1, build attempt 2 | Same research contract; initialize pandas index before concurrent reads | 1 | Completed, independently verified, rejected for advancement |
| M4 | Pair-by-room slopes | 0 | Not triggered |

Version `1.0` identifies the research specification; timestamped directories identify execution attempts. The first failure was a pandas lazy-index initialization race, not duplicate population times. An independent check confirmed all 48,192 timestamps unique. No outcomes were scored in that attempt. Both runner snapshots remain. A future research change must use v2; v1 outputs are immutable.

## Step 1 — pin the comparison

Target remains absolute BTC 48h endpoint **log** return in bps. This is magnitude/median prediction scored by MAE, not direction, threshold crossing or path excursion. Same 29,615 training rows, 96 boundary-purge rows and 18,481 development rows. Training origins: 2022-04-22 00:30 through 2023-12-29 23:30 UTC. Development: 2024-01-01 00:00 through 2025-01-22 00:00 UTC. All training labels finish before development starts.

M2 is frozen MP0004: cross-asset liquidation historical active rate **15m + 5m**. Its prior training imputation, standardization, missing indicators, coefficients and development predictions were replayed before the new fit. Selection of this pair already used development outcomes.

## Step 2 — read the room definitions

Read `ALPHA PRO V2/LEARN/V2/NESTED_IDEA/IDEA.html`, then its referenced `GROUP/V2/CONTRACT_FINAL.md` and `build.py`. The illustration describes 27 directional rooms; the underlying source preserves four FLOW/LIQ words, separating quiet from churn/chop. For this user-requested three-way room, valid quiet and balanced map to neutral; raw four-way words are retained in `room_states.parquet`. Missing is always code 0 and never neutral.

Six **raw trailing 1d F4** sensors are used: price displacement; flow activity, CVD path efficiency and flow pressure; liquidation side pressure and activity severity. No 1d F4Z percentile is required. Exported values need finite value, valid flag and nonnull `available_after <= decision_time`. No forward-fill or backfill. All 3,024 source partitions were hash-checked. A separate verifier reread nine deterministic source partitions and reconciled 576 scalar values. Export validity and timing are checked; the complete upstream F2 measurement construction has not been rebuilt.

## Step 3 — freeze coding and training-only thresholds

- PRICE: flat when absolute displacement is below its training Q1/3, **0.006200838294942 log return = 62.008383 log bps**. Otherwise positive is UP and negative DOWN. Exactly zero valid displacement is flat.
- FLOW: activity below training Q1/4 (**12196710854.830570 quote notional**) is quiet/neutral. Otherwise efficiency >=0.5 plus positive pressure gives BUY; efficiency >=0.5 plus negative pressure gives SELL; valid remaining cases are churn/neutral. Required unavailable inputs remain missing.
- LIQ: valid side pressure >+1/3 gives SHORT-LIQ; <-1/3 gives LONG-LIQ; the middle is neutral. Invalid side permits quiet/neutral only with valid zero severity; otherwise it is missing. This follows the source builder precedence.
- PRICE x FLOW x LIQ gives 27 physical room IDs. Missing in any required coordinate gives separate code0.
- Legacy discovery-window CUTS and STATE tables were not reused; both quantile thresholds use only current training observations.
- The known source-binding change at 2024-10-29 00:01 UTC makes the next 24h of context unavailable: 48 sampled rows use fallback. No observed 1d window spanning that boundary is interpreted as a continuous regime.

## Step 4 — support, missingness and model

The support rule was fixed before fitting: a physical room needs >=200 training rows across >=20 distinct UTC training dates. This is a basic support filter, not a claim of 200 independent observations. The outcome horizon overlaps.

There are 25 physical rooms observed in training, 26 in development and **14 supported rooms**. Model M3 jointly fits one common intercept, the two standardized fast-feature slopes, their two missing indicators, and 14 room offsets: 19 coefficients with training design rank19. No feature-by-room products are present. Unsupported/missing rows remain in the training fit with zero room indicators; at prediction they explicitly use the saved M2 forecast. They are never silently assigned to a neutral room.

On development, 18,031 rows (97.57%) use M3. The remaining 450 retain M2 exactly: 96 have missing context (including the binding mask) and 354 have a known but insufficiently supported room. All 18,481 rows are scored. This is a support-gated additive room test, not 27 independently fitted regressions.

Optimization remains unpenalized median regression: minimize sum absolute training residuals using HiGHS IPM with the same tolerances as the pair tournament. Predict max(0, linear forecast), subject to the frozen fallback above. One new fit; no tuning. Dual certificate and stationarity checks passed.

## Step 5 — compare on the unchanged development rows

Negative change means context helps; positive means it hurts.

| Month | Rows | M2 MAE | M3 MAE | M3 minus M2 |
|---|---:|---:|---:|---:|
| 2024-01 | 1,488 | 178.429 | 182.015 | +3.586 |
| 2024-02 | 1,392 | 238.099 | 229.937 | -8.162 |
| 2024-03 | 1,392 | 231.568 | 237.009 | +5.442 |
| 2024-04 | 1,440 | 178.029 | 182.213 | +4.184 |
| 2024-05 | 1,488 | 169.180 | 170.092 | +0.912 |
| 2024-06 | 1,440 | 114.818 | 119.535 | +4.718 |
| 2024-07 | 1,488 | 192.270 | 191.822 | -0.448 |
| 2024-08 | 1,488 | 238.827 | 241.924 | +3.096 |
| 2024-09 | 1,440 | 151.554 | 156.886 | +5.332 |
| 2024-10 | 1,488 | 132.973 | 131.107 | -1.866 |
| 2024-11 | 1,440 | 225.637 | 228.969 | +3.332 |
| 2024-12 | 1,488 | 149.605 | 155.911 | +6.306 |
| 2025-01 | 1,009 | 164.919 | 160.106 | -4.813 |

## Step 6 — dependence and concentration checks

Paired block bootstrap preserves chronological stretches: 2,000 draws each, 7-day and 14-day blocks, row-count weighted. The intervals are conditional on previous development selection; they do not undo earlier model searches.

| Comparison | Blocks | 95% interval for MAE change |
|---|---|---:|
| M3 minus M2 | 7 days, 56 blocks | [+0.0130, +3.6601] |
| M3 minus M2 | 14 days, 28 blocks | [-0.1113, +3.6774] |

The 14-day interval includes zero. The evidence supports rejecting advancement of this version, not a universal claim that room context cannot help. Removing March and August still leaves M3 **1.3916 bps worse**. Removing its two largest gain months (February 2024 and January 2025) leaves it **3.1163 bps worse**.

## Step 7 — independent verification and decision

A separate verifier reconstructed every room label and every prediction from saved raw inputs and original pair preprocessing, checked source samples, solver dual feasibility/stationarity, monthly scores, concentration and all block intervals. **PASS**; 48,192 labels and predictions replayed; zero uncertainty discrepancy; no verification refit.

The fixed advancement gate required lower aggregate MAE than M2, a majority of months improving, both block intervals below zero and improvement after dropping the two largest gain months. v1 fails. Keep the saved M2 for this comparison, park this room specification and do not automatically expand to M4 or faster nested contexts. This does not validate M2 as an investable forecast. No untouched holdout is verified.

## Artifacts and reproduction

- [Frozen v1 contract](room_overlays/v1_20261031T102650_855172Z/CONTRACT.json).
- [Machine-readable result](room_overlays/v1_20261031T102650_855172Z/RESULT.json).
- [All 27 room census, plus missing](room_overlays/v1_20261031T102650_855172Z/room_census.csv).
- [Room-level scores](room_overlays/v1_20261031T102650_855172Z/room_scores.csv), [monthly scores](room_overlays/v1_20261031T102650_855172Z/monthly_scores.csv), [coverage](room_overlays/v1_20261031T102650_855172Z/input_coverage.csv).
- [Prediction file](room_overlays/v1_20261031T102650_855172Z/predictions.parquet), [coefficients](room_overlays/v1_20261031T102650_855172Z/model.json), [room cutoffs](room_overlays/v1_20261031T102650_855172Z/room_definition.json).
- [Input and output hashes](room_overlays/v1_20261031T102650_855172Z/ARTIFACT_HASHES.json), [verification receipt](room_verification/v1_20261031T102650_855172Z.json).
- Run snapshot: `executed_runner.py` inside each attempt. Source loader, training and scoring are in `../../tools/run_48h_magnitude_room_v1.py`; verifier in `../../tools/verify_48h_magnitude_room_v1.py`.
- First attempt: `room_overlays/v1_20261031T102631_371575Z/FAILURE.json`.

Fit ledger after this stage: **3,219 total = 2,772 extreme workflow + 447 magnitude**. Magnitude =256 standalone +190 pairs +1 room model. Audit refits remain2 overall. No M4 fit.


</details>

### Frozen room offsets V2

[Original document](<C:/Users/REDACTED_USER/Desktop/MFT ML/REPEATABLE PLAN/48H_ML/magnitude/05_ROOM_CONTEXT_V2.md>) ? SHA-256 `fb74625fe3bfd730107310f50256de2da7e47d91472c3c50935fcdd0088c6973`

<details>
<summary>Read complete preserved document</summary>

# 48h magnitude — room overlay v2

**The frozen pair improves in four supported rooms on development data. None passes the simultaneous room-uncertainty rule. Across all rows, adding all supported room adjustments increases MAE.**

## What changed from v1

v1 jointly refitted the common pair coefficients with room offsets. v2 keeps the winning MP0004 pair completely frozen and fits only a training residual median for each supported room. The user specifically requested the exact room-by-room comparison. v1 was a different valid comparison; its pooled loss did not establish that every room failed. Indeed, its saved room table also had four descriptive gains.

No pair coefficient, imputation, standardization, room cutoff, target, train/development row or scoring metric changed. This is not a room-by-feature slope interaction.

## Step-by-step specification

1. Freeze the source v1 artifact hashes and new v2 contract before fitting offsets.
2. Reuse the exact 27 physical PRICE/FLOW/LIQ rooms, plus separate missing code0. PRICE flat threshold and FLOW quiet threshold were learned from training only. No percentile computed using development is introduced.
3. Reuse training support: >=200 rows and >=20 UTC dates. Fourteen rooms qualify. Sparse/missing/unseen rooms stay separate and retain M2 unchanged.
4. Replay the frozen MP0004 forecast on all original rows. Calculate each training residual as actual magnitude minus M2 forecast.
5. For each supported room, learn `offset[r] = median(training residuals in room r)`. This is a median/MAE correction; it leaves the pair coefficients fixed. One context model with14 offsets, no tuning.
6. Freeze offsets. Forecast `max(0, M2 + offset[room])`. Missing/sparse/unseen room offset is zero.
7. Score both forecasts on exactly the same development rows within each room. Retain pooled scoring of all18,481 rows. A room having larger returns or a lower raw MAE than a different room is not itself an incremental prediction gain.
8. Report every room, occupied months and training/development coverage. Use paired 7d/14d chronological block bootstrap with2,000 draws, seed48+days, and simultaneous14-room bands.
9. Independently reconstruct offsets, all48,192 predictions, room scores, month scores and simultaneous intervals. Replay passed; zero refits.
10. Save v2 as a separate immutable run. Do not build a forecast using only the four development winners: choosing those rooms now would add another development-selected model. Confirmation requires later untouched data.

## Within-room results

Gain = M2 MAE minus v2 MAE. Positive helps; negative hurts. Months count only months containing that room; small monthly counts remain visible in the CSV.

| PRICE | FLOW | LIQ | Dev rows | Pair MAE | v2 MAE | Gain bps | Gain % | Months improved |
|---|---|---|---:|---:|---:|---:|---:|---:|
| UP | BUY | NEUTRAL | 570 | 158.1385 | 150.5286 | +7.6099 | +4.81% | 8/13 |
| FLAT | NEUTRAL | SHORT-LIQ | 279 | 221.9796 | 221.0195 | +0.9601 | +0.43% | 6/13 |
| UP | BUY | SHORT-LIQ | 2,684 | 194.0047 | 193.5445 | +0.4602 | +0.24% | 6/13 |
| DOWN | NEUTRAL | NEUTRAL | 1,009 | 192.4140 | 192.1414 | +0.2726 | +0.14% | 7/13 |
| DOWN | SELL | LONG-LIQ | 2,994 | 181.7606 | 181.9494 | -0.1888 | -0.10% | 7/13 |
| UP | NEUTRAL | SHORT-LIQ | 1,392 | 177.0898 | 177.3476 | -0.2578 | -0.15% | 3/13 |
| DOWN | SELL | NEUTRAL | 1,765 | 172.3784 | 172.7641 | -0.3857 | -0.22% | 6/13 |
| UP | NEUTRAL | NEUTRAL | 2,422 | 183.9946 | 184.6833 | -0.6887 | -0.37% | 6/13 |
| FLAT | NEUTRAL | NEUTRAL | 3,135 | 179.8829 | 181.9253 | -2.0424 | -1.14% | 3/13 |
| DOWN | NEUTRAL | LONG-LIQ | 380 | 133.6527 | 135.9570 | -2.3043 | -1.72% | 5/12 |
| FLAT | BUY | NEUTRAL | 80 | 249.1334 | 251.6079 | -2.4746 | -0.99% | 5/12 |
| FLAT | NEUTRAL | LONG-LIQ | 281 | 143.7212 | 149.2785 | -5.5573 | -3.87% | 5/11 |
| FLAT | SELL | LONG-LIQ | 131 | 138.8159 | 147.0505 | -8.2346 | -5.93% | 5/11 |
| FLAT | SELL | NEUTRAL | 909 | 193.9846 | 212.3668 | -18.3822 | -9.48% | 2/13 |

The remaining13 physical rooms have insufficient training support; they are all retained in the census and scoreboard, including the room with zero development rows. Missing is an additional separate category. No unsupported room was relabeled neutral.

Across all rows: M2 MAE **182.036977**, v2 **183.334720**, change **+1.297743 log bps**. Four of13 months improve overall. The room-level gains and the pooled deterioration are both part of the result.

## Uncertainty and interpretation

Four rooms have lower development MAE; zero pass both simultaneous interval tests plus a majority of occupied-month wins. These adjustments need not help on later observations just because they improve training MAE. Examining many rooms also creates opportunities for apparent wins by chance. The selected parent and this development period were already examined repeatedly.

| Improving room | 14d nominal95% interval, v2 minus pair | 14d simultaneous95% interval |
|---|---:|---:|
| UP / BUY / NEUTRAL | [-20.072, +5.778] | [-27.309, +12.089] |
| FLAT / NEUTRAL / SHORT-LIQ | [-3.308, +1.630] | [-4.668, +2.747] |
| UP / BUY / SHORT-LIQ | [-6.366, +5.611] | [-9.558, +8.638] |
| DOWN / NEUTRAL / NEUTRAL | [-6.167, +5.122] | [-8.652, +8.106] |

**Decision:** preserve the four descriptive room gains as candidates. No confirmed room edge, no automatic M4, no winners-only deployment. The frozen pair remains the full-population comparison benchmark. No verified untouched holdout.

## Reproducibility and versions

- [v2 contract](room_overlays/v2_20261031T103411_915449Z/CONTRACT.json), [training offsets](room_overlays/v2_20261031T103411_915449Z/model.json), [result](room_overlays/v2_20261031T103411_915449Z/RESULT.json).
- [Every room scoreboard](room_overlays/v2_20261031T103411_915449Z/room_scoreboard.csv), [room uncertainty](room_overlays/v2_20261031T103411_915449Z/room_uncertainty.csv), [monthly room scores](room_overlays/v2_20261031T103411_915449Z/monthly_by_room.csv).
- [All predictions](room_overlays/v2_20261031T103411_915449Z/predictions.parquet), [artifact hashes](room_overlays/v2_20261031T103411_915449Z/ARTIFACT_HASHES.json), [verification](room_verification/v2_20261031T103411_915449Z.json).
- [v1 is preserved](04_ROOM_CONTEXT_V1.md); [version history](ROOM_VERSION_HISTORY.json).
- Runner: `REPEATABLE PLAN/tools/run_48h_magnitude_room_v2.py`; exact executed copy saved in run. Verifier/documenter: `REPEATABLE PLAN/tools/verify_document_48h_room_v2.py`.

Fit ledger: 3,220 total =2,772 extreme workflow +448 magnitude. Magnitude=256 singles+190 pairs+one joint room v1+one frozen-pair room v2. Two overall historical audit refits remain included; current verification adds zero.


</details>

### Compact trees V3

[Original document](<C:/Users/REDACTED_USER/Desktop/MFT ML/REPEATABLE PLAN/48H_ML/magnitude/06_NESTED_TREE_V3.md>) ? SHA-256 `c40fa01a3f615aa05da7c0e7c885526d4da12c33125d5f9de34dfc5b23542662`

<details>
<summary>Read complete preserved document</summary>

# 48h magnitude — nested context and tree v3

**Four fixed fits completed and independently verified. None beat the saved best pair. Adding1d context helped the tree only0.0471 log bps; adding4h context changed no predictions.**

| Model | Development MAE, log bps |
|---|---:|
| M0: Training median baseline | 187.831072 |
| M2: Saved best15m+5m pair | 182.036977 |
| L20: 20-feature additive median regression | 182.362546 |
| T20: 20-feature median tree | 184.046292 |
| T20_D1: Tree +1d room | 183.999201 |
| T20_D1_H4: Tree +1d and4h rooms | 183.999201 |

The last two models produce exactly identical predictions on all48,192 original rows. Inspection of serialized trees confirms **zero4h features used**, zero paths conditioning on both1d and4h, and140 paths combining fast feature values with1d context. These counts are paths across150 trees, not independent observations or unique market events.

## Version and scope

- v1: jointly fitted best pair plus additive1d room offsets. One fit, preserved.
- v2: frozen best pair plus training residual-median offsets; four exploratory room gains. One fit, preserved.
- v3: user-authorized broader top20 features, nonlinear trees, and1d/4h room context. One additive control and three fixed trees, four fits.
- Explicit linear feature-by-room multiplication terms have not been fitted. The tree was allowed to learn conditional effects; it did not choose4h splits. No automatic additional sweep was run.

## Step1 — freeze the research question

Use exactly the prior displayed top20 magnitude features, without another ranking or changing variants. Those features were selected using development data earlier; this is not an untouched validation exercise. Target remains the absolute48h BTC endpoint log return in bps. Same29,615 training rows,96-row boundary purge and18,481 development rows; every row retained. Train origins2022-04-22 00:30 through2023-12-29 23:30 UTC; development2024-01-01 00:00 through2025-01-22 00:00 UTC. Training label ends precede development start.

The requested1d and4h contexts are trailing windows ending at the same decision time: the4h interval lies within the1d interval. Context uses no future outcome.

## Step2 — qualify the4h room inputs

Read the six raw F4 sensors for PRICE/FLOW/LIQ from all3,024 source partitions; check every partition checksum against the canonical source registry. Gate finite values with valid flags and nonnull availability no later than decision time. Reuse the verified1d state construction. No forward-fill, no precomputed full-history room thresholds. Upstream F2 sensor construction remains outside this exported-data replay.

4h PRICE flat threshold: training Q1/3 of absolute displacement,0.002129934802053188 log return (21.299348 log bps). 4h FLOW quiet threshold: training Q1/4 of activity,1,462,942,098.4058452 quote notional. Remaining source rules are unchanged: efficiency>=0.5, signed flow pressure, liquidation side thresholds +/-1/3. Known quiet/churn/chop map to neutral; missing is separate0. All raw four-way states are saved.

The known source-binding transition at2024-10-29 00:01 UTC masks the following4h of4h context (8 sampled rows); the existing1d transition mask remains24h (48 rows). Source invalidity and missing inputs are preserved.

## Step3 — define room inputs before fitting

A physical room needs>=200 training rows across>=20 distinct UTC training dates. Fourteen1d rooms and fifteen4h rooms meet this fixed rule. Unsupported known rooms receive OTHER code28; missing stays code0; neither is silently neutral. Each clock is encoded as29 nominal one-hot columns (0..28), including absent training categories. No ordinal interpretation of a room number is introduced.

The census includes all729 physical1d/4h combinations plus combinations containing missing. Training observes399 physical nested cells; development observes376. Forty-two joint cells meet the same basic200-row/20-date support rule. These joint counts are an audit, not729 independently fitted models. No explicit729 joint dummy expansion was added.

The tree can split on a1d room, then a4h room, then a fast value (maximum depth3). This makes conditional nesting available to the learner. It does not force the learner to choose it.

## Step4 — preserve feature preprocessing and separate effects

All20 existing continuous features reuse their original training median imputation, training mean/std and separate original-missing indicator. All models receive40 feature columns. D1 adds29 columns; D1+H4 adds58. Missing and sparse context remain explicit model inputs; rows are not dropped.

| Contrast | Question |
|---|---|
| L20 vs M2 | Does adding the remaining top20 values help additively? |
| T20 vs L20 | Does this nonlinear tree help on identical feature inputs? |
| T20_D1 vs T20 | Does1d context add information to the tree? |
| T20_D1_H4 vs T20_D1 | Does4h context add beyond1d? |
| T20_D1_H4 vs T20 and M2 | Does the full context model improve either benchmark? |

## Step5 — fit exactly four models

L20 uses the same unpenalized median/LAD regression as prior magnitude experiments, with the same solver tolerances. The independent dual certificate passes.

All three trees use identical fixed XGBoost3.4.1 settings: median quantile objective (`reg:quantileerror`, alpha0.5),150 trees, depth3, learning rate0.03, minimum child weight100, lambda10, max_bin64, full row/column sampling, histogram CPU algorithm, seed48, one thread. Initial prediction is the original training median. No development eval_set, early stopping, parameter search or calibration. Predictions are projected to nonnegative magnitudes. Saved models include UBJ serialization, full JSON tree dumps and booster configs.

## Step6 — evaluate paired losses, months and uncertainty

All models are evaluated on the same18,481 development rows. Negative MAE difference helps.

| Model minus comparator | MAE difference | Months improved | 14d simultaneous95% interval |
|---|---:|---:|---:|
| L20 minus M2 | +0.325569 | 5/13 | [-1.0100, +1.6612] |
| T20 minus L20 | +1.683746 | 4/13 | [-1.5156, +4.8831] |
| T20_D1 minus T20 | -0.047091 | 8/13 | [-0.5425, +0.4483] |
| T20_D1_H4 minus T20_D1 | +0.000000 | 0/13 | [+0.0000, +0.0000] |
| T20_D1_H4 minus T20 | -0.047091 | 8/13 | [-0.5425, +0.4483] |
| T20_D1_H4 minus M2 | +1.962224 | 5/13 | [-1.4985, +5.4229] |

Uncertainty uses2,000 paired calendar-block resamples at7d and14d, weighted by row counts, seed48+days. Simultaneous bands cover the six predeclared contrasts. They do not erase earlier feature/model selection. Identical-model contrast intervals [0,0] are an arithmetic consequence of identical predictions; they do not establish that4h context has no possible information.

Monthly scores, fixed March/March+August exclusions, and exclusions of the largest one/two gain months are saved for every comparison. Full per1d-room and per nested-cell scores are also saved. In every nested cell, the added4h model versus1d-only model has exactly zero incremental difference. Any cell looking better than the old pair therefore cannot be attributed to4h context in this fit.

## Step7 — inspect learned nesting and verify

The tree with1d context has140 leaf paths combining a fast value and1d context. The proposed nested tree uses the same predictions and no4h feature. It learned no1d-by4h branching. This is a substantive outcome, not merely an average loss hiding strong4h improvements.

A separate verifier reconstructed every4h state using independent branch logic, training cutoffs/support/one-hot encodings, all feature preprocessing, the linear fit certificate and serialized tree predictions. It checked nine source partitions (576 scalar values), all48,192 predictions per model, month and nested-cell scores, paired uncertainty and tree paths. Verification PASS; zero verification fits.

## Decision and limits

Keep M2 as the comparison benchmark. L20 is0.1788% worse; the context tree is1.0779% worse. The1d addition to the feature-only tree gives a tiny0.0256% improvement with uncertainty including zero. The4h addition gives exactly zero improvement. No model passes the fixed advancement rule.

This fixed tree configuration did not extract a nested4h effect. Greedy trees can fail to discover interactions whose individual splits offer too little gain; this result does not settle every possible explicit product or hierarchical model. Those alternatives would be new versioned experiments, not reinterpretations of this run. No tuning or further automatic fit was made. No verified untouched holdout.

## Features used

The full top20 is retained, including correlated variants. This directly follows the requested broader feature test; no claim of20 independent concepts is made.

| Order | Canonical feature |
|---:|---|
| 1 | `F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__15m__value` |
| 2 | `F4Z__liquidation__liq_activity_severity_historical_active_rate__15m__value` |
| 3 | `F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__1h__value` |
| 4 | `F4Z__liquidation__liq_activity_severity_historical_active_rate__5m__value` |
| 5 | `F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__5m__value` |
| 6 | `F4Z__liquidation__liq_activity_severity_historical_active_rate__1h__value` |
| 7 | `F4Z__liquidation__venue_fragmentation_percentile__15m__value` |
| 8 | `F4Z__liquidation__liq_activity_severity_active_severity_percentile__15m__value` |
| 9 | `F4Z__liquidation__liq_activity_severity_active_severity_rz__15m__value` |
| 10 | `F4Z__liquidation__liq_acceleration_rz__5m__15m__value` |
| 11 | `F4Z__liquidation__liq_side_pressure_percentile__15m__value` |
| 12 | `F4Z__liquidation__liq_temporal_concentration_percentile__15m__value` |
| 13 | `F4Z__flow__flow_confirmation_rz__15m__value` |
| 14 | `F4Z__flow__flow_confirmation_percentile__15m__value` |
| 15 | `F4Z__liquidation__liq_side_pressure_rz__15m__value` |
| 16 | `F4Z__liquidation__liq_acceleration_percentile__5m__15m__value` |
| 17 | `F4Z__liquidation__liq_peak_age_seconds_percentile__15m__value` |
| 18 | `F4Z__liquidation__cross_asset_liq_activity_raw_active_severity_rz__15m__value` |
| 19 | `F4Z__liquidation__cross_asset_liq_activity_raw_active_severity_percentile__15m__value` |
| 20 | `F4Z__liquidation__cross_asset_liq_activity_raw_active_severity_rz__5m__value` |

## Saved artifacts

- [Frozen v3 contract](nested_context/v3_20261031T104224_906997Z/CONTRACT.json), [result](nested_context/v3_20261031T104224_906997Z/RESULT.json), [all model scores](nested_context/v3_20261031T104224_906997Z/model_scores.csv).
- [All predictions](nested_context/v3_20261031T104224_906997Z/predictions.parquet), [monthly comparisons](nested_context/v3_20261031T104224_906997Z/monthly_comparisons.csv), [uncertainty](nested_context/v3_20261031T104224_906997Z/uncertainty.csv).
- [729-cell census plus missing](nested_context/v3_20261031T104224_906997Z/nested_census.csv), [nested cell results](nested_context/v3_20261031T104224_906997Z/nested_scores.csv), [1d room results](nested_context/v3_20261031T104224_906997Z/one_day_scores.csv).
- [Actual learned paths](nested_context/v3_20261031T104224_906997Z/tree_path_audit.json), [feature importance](nested_context/v3_20261031T104224_906997Z/tree_importance.csv), [input rules](nested_context/v3_20261031T104224_906997Z/room_rules.json).
- [Artifact checksums](nested_context/v3_20261031T104224_906997Z/ARTIFACT_HASHES.json), [independent receipt](nested_verification/v3_20261031T104224_906997Z.json).
- Runner `REPEATABLE PLAN/tools/run_48h_nested_tree_v3.py`; immutable executed copy inside the run. Verifier `verify_48h_nested_tree_v3.py` in the same tools directory.
- [v1](04_ROOM_CONTEXT_V1.md), [v2](05_ROOM_CONTEXT_V2.md), [version history](ROOM_VERSION_HISTORY.json).

Fit ledger:3,224 total =2,772 extreme workflow +452 magnitude. Magnitude=256 singles+190 pairs+2 room overlays+1 broader additive model+3 trees. Two historical audit refits are included overall; this verification added zero.


</details>

### All pair-room interactions V4

[Original document](<C:/Users/REDACTED_USER/Desktop/MFT ML/REPEATABLE PLAN/48H_ML/magnitude/07_ROOM_INTERACTIONS_V4.md>) ? SHA-256 `dee1feddac65ed5c0ae01895f49be427988ff6da66aa17d8ec55bb220943a718`

<details>
<summary>Read complete preserved document</summary>

# 48h magnitude — V4 Step 1: categorical room interactions across all top20 pairs

**Completed and verified: 190/190 pairs; 379 new fits, 1 reused control, 0 failed fits.**

0 pairs improve over their own saved global pair. 0 improve over both their own global pair and the matched room-offset control. 0 beat the common best M2 benchmark. 0 pass the fixed pair credibility gate; 0 additionally qualify against best M2.

Across 2,660 eligible pair-room cells, 978 beat their own global pair, 435 beat both comparators, and 0 pass the simultaneous room-level evidence rule.

**Stopped after Step 1. No 4h fit, tree, additional horizon, parameter sweep or new feature search was run.**

## Versions and exact research question

v1 jointly fitted room offsets for the best pair. v2 held that pair frozen and learned training residual-median offsets. v3 tested top20 median trees with 1d/4h room context; the tree ignored 4h. These runs remain unchanged.

V4 follows the user’s explicit extension: use every unordered pair from the same top20, and allow each pair’s feature slopes to differ by categorical 1d room. The count is 20 choose 2 =190. This is not a fresh selection of the best20 pairs, not a 20-feature multivariate interaction model, and not a new feature ranking.

The question is whether changing the X1/X2 relationship by room improves forecasts on later rows. Comparing raw MAE across different rooms does not answer that question. All stated lifts compare forecasts on identical rows within the same room or on the unchanged full development population.

## Step 1 — pin data, target and comparators

Target: absolute BTC 48h endpoint log return in bps. Primary metric: MAE, hence median/LAD regression. Same 29,615 training observations, 96 boundary-purge rows and 18,481 development observations. Train origins span 2022-04-22 00:30 through 2023-12-29 23:30 UTC; development 2024-01-01 00:00 through 2025-01-22 00:00 UTC. Training label ends precede development.

Each pair has three comparisons: its own saved global M2, its matched room-offset M3, and the common saved best pair MP0004 (MAE182.036977). Own-pair gains alone can favor weak starting models; the best-M2 comparison makes that visible. All prior source manifests, frozen predictions and baseline contracts are checked before fitting.

## Step 2 — retain the original training-only preprocessing

Reuse each feature’s training median imputation, training centering/std and separate original-missing indicator. No dropping development rows, recomputing preprocessing on later data or changing features. The two missing indicators enter as global main effects; there are no missing-indicator-by-room products. Both-values-valid and any-missing results are reported separately.

## Step 3 — define categorical room and support

Reuse audited v1 1d PRICE/FLOW/LIQ states and training-only cutoffs. Twenty-seven physical rooms plus separate missing code0; room IDs are never treated as continuous. Quiet/balanced states use the established neutral mapping; missing is never neutral.

The original14 rooms pass >=200 training rows and >=20 distinct UTC dates. Each pair-room additionally needs >=100 both-feature-valid training observations spanning >=20 dates, >=3 distinct raw values for each feature, and rank3 for [1,Z1,Z2] on those valid training rows. These checks use no development outcomes. All190 pairs passed in all14 rooms, producing2,660 eligible pair-room cells.

Unsupported, missing and unobserved rooms remain explicit in every census and scoreboard. They use the own saved pair forecast exactly; no implicit neutral assignment. Eligibility and raw value-rank diagnostics are saved before any new fit.

## Step 4 — fit the explicit model and a necessary control

For each pair, the matched additive control is:

`M3 = b0 + b1*Z1 + bm1*M1 + b2*Z2 + bm2*M2 + sum_r(a_r*I(room=r))`

The requested interaction model is:

`M4 = b0 + b1*Z1 + bm1*M1 + b2*Z2 + bm2*M2 + sum_r(I(room=r)*(a_r + g1_r*Z1 + g2_r*Z2))`

The effective value slopes inside room r are `b1+g1_r` and `b2+g2_r`. Main effects and products are fitted jointly. This is explicit multiplication of standardized feature values by categorical room indicators. It is not merely a room offset or an X1*X2-only model.

The saved M2 comparator remains frozen. M3 and M4 jointly learn their own parameters from all original training rows; M4 is allowed to change slopes. All unsupported training rows enter with zero room columns. At prediction, supported rooms use the fitted nonnegative forecast; unsupported/missing rooms fall back to saved M2. This is identical for the M3 control and M4.

Each M3 has19 columns; each M4 has47. No regularization or hyperparameter tuning was introduced: same HiGHS IPM median-regression settings and tolerances. The fixed top20 include correlated forms and some identical missing indicators; the numerical design rank is recorded. Individual coefficients in a deficient design can be nonunique. Room slopes are reported with the pair-room value-rank support checks and are descriptive fitted relationships, not causal effects.

M3 is needed to distinguish a slope gain from an offset gain. There are190 M4 fits and190 controls evaluated. The exact MP0004 M3 from v1 is reused after eligibility/design compatibility is checked;189 other controls are new. Total new fits379, no redundant refit of the reused model.

## Step 5 — score full population and every room

Save overall and monthly scores for every pair; every physical room plus missing; room-by-month results; both-valid/any-missing results; original global and effective room slopes; and concept group summaries. Forecasts for both M3 and M4 are retained on every development row.

Positive gain in the next table means lower M4 MAE than that comparator. Rankings use gain versus the own global pair; all190 outcomes are available in the linked full scoreboard.

| Rank | Pair | Features | Global MAE | M4 MAE | Gain vs global | Gain vs offsets | Rooms beating both |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | MP0058 | BTC-liq activity 5m + BTC-liq severity pct 15m | 183.9892 | 185.0414 | -1.0521 | +0.2002 | 3/14 |
| 2 | MP0062 | BTC-liq activity 5m + Liq temporal concentration pct 15m | 184.0573 | 185.1988 | -1.1416 | +0.1342 | 2/14 |
| 3 | MP0059 | BTC-liq activity 5m + BTC-liq severity RZ 15m | 183.9831 | 185.1525 | -1.1694 | +0.0916 | 2/14 |
| 4 | MP0057 | BTC-liq activity 5m + Venue fragmentation pct 15m | 183.9990 | 185.2404 | -1.2414 | +0.0072 | 2/14 |
| 5 | MP0068 | BTC-liq activity 5m + Cross-liq severity RZ 15m | 184.0521 | 185.2940 | -1.2420 | +0.0827 | 2/14 |
| 6 | MP0069 | BTC-liq activity 5m + Cross-liq severity pct 15m | 184.0537 | 185.3264 | -1.2728 | +0.0547 | 2/14 |
| 7 | MP0061 | BTC-liq activity 5m + Liq side pressure pct 15m | 184.0176 | 185.3071 | -1.2894 | +0.0297 | 2/14 |
| 8 | MP0065 | BTC-liq activity 5m + Liq side pressure RZ 15m | 184.0166 | 185.3180 | -1.3014 | +0.0122 | 3/14 |
| 9 | MP0055 | BTC-liq activity 5m + Cross-liq activity 5m | 184.3011 | 185.6042 | -1.3031 | -0.0674 | 4/14 |
| 10 | MP0066 | BTC-liq activity 5m + Liq acceleration pct 5m 15m | 184.0138 | 185.3325 | -1.3187 | +0.0230 | 2/14 |
| 11 | MP0067 | BTC-liq activity 5m + Liq peak age pct 15m | 184.0281 | 185.3734 | -1.3453 | -0.0560 | 3/14 |
| 12 | MP0064 | BTC-liq activity 5m + Flow confirmation pct 15m | 184.0086 | 185.3594 | -1.3508 | -0.0450 | 2/14 |
| 13 | MP0063 | BTC-liq activity 5m + Flow confirmation RZ 15m | 184.0133 | 185.4063 | -1.3929 | -0.0856 | 2/14 |
| 14 | MP0060 | BTC-liq activity 5m + Liq acceleration RZ 5m 15m | 184.0111 | 185.4490 | -1.4378 | -0.0939 | 2/14 |
| 15 | MP0143 | Liq acceleration RZ 5m 15m + Cross-liq severity RZ 15m | 186.6237 | 188.0737 | -1.4501 | +0.1740 | 3/14 |
| 16 | MP0144 | Liq acceleration RZ 5m 15m + Cross-liq severity pct 15m | 186.6097 | 188.1078 | -1.4982 | +0.1459 | 3/14 |
| 17 | MP0070 | BTC-liq activity 5m + Cross-liq severity RZ 5m | 184.1626 | 185.6648 | -1.5022 | -0.1850 | 2/14 |
| 18 | MP0160 | Liq temporal concentration pct 15m + Cross-liq severity RZ 15m | 186.6649 | 188.1950 | -1.5301 | +0.1314 | 2/14 |
| 19 | MP0158 | Liq temporal concentration pct 15m + Liq acceleration pct 5m 15m | 186.8062 | 188.4065 | -1.6002 | -0.0231 | 4/14 |
| 20 | MP0157 | Liq temporal concentration pct 15m + Liq side pressure RZ 15m | 186.7568 | 188.3625 | -1.6057 | -0.0215 | 3/14 |

Lowest absolute M4 MAE: **MP0038 = 184.604518**, versus own global 182.741888, offset control 183.700647, and common best M2 182.036977.

Original winning pair MP0004: M2 **182.036977**, M3 **183.870886**, M4 **185.502288**. Its M4 improves in 4/13 months over M2 and 3/14 rooms.

## Step 6 — look for actual within-room lifts

The following20 cells have the largest descriptive gain versus their own global pair, among eligible cells also improving over the matched offset model. These cells were selected using development results. Rows overlap in future48h paths; counts are not independent samples. Full tables include losing cells and cells with no fitted room model.

| Pair | 1d room | Dev rows | Global MAE | M4 MAE | Gain vs global | Gain vs offsets | Room evidence |
|---|---|---:|---:|---:|---:|---:|---|
| MP0120 | FLAT / BUY / NEUTRAL | 80 | 256.167 | 231.075 | +25.093 | +7.577 | EXPLORATORY_ROOM_INTERACTION_GAIN |
| MP0114 | FLAT / BUY / NEUTRAL | 80 | 255.826 | 231.426 | +24.400 | +7.132 | EXPLORATORY_ROOM_INTERACTION_GAIN |
| MP0121 | FLAT / BUY / NEUTRAL | 80 | 256.058 | 231.953 | +24.105 | +6.610 | EXPLORATORY_ROOM_INTERACTION_GAIN |
| MP0131 | FLAT / BUY / NEUTRAL | 80 | 256.253 | 233.649 | +22.604 | +5.064 | EXPLORATORY_ROOM_INTERACTION_GAIN |
| MP0132 | FLAT / BUY / NEUTRAL | 80 | 256.143 | 234.606 | +21.536 | +4.070 | EXPLORATORY_ROOM_INTERACTION_GAIN |
| MP0125 | FLAT / BUY / NEUTRAL | 80 | 256.008 | 235.090 | +20.918 | +3.617 | EXPLORATORY_ROOM_INTERACTION_GAIN |
| MP0117 | FLAT / BUY / NEUTRAL | 80 | 255.935 | 235.326 | +20.609 | +3.216 | EXPLORATORY_ROOM_INTERACTION_GAIN |
| MP0187 | FLAT / BUY / NEUTRAL | 80 | 257.432 | 236.860 | +20.572 | +3.032 | EXPLORATORY_ROOM_INTERACTION_GAIN |
| MP0113 | FLAT / BUY / NEUTRAL | 80 | 255.870 | 235.478 | +20.392 | +3.337 | EXPLORATORY_ROOM_INTERACTION_GAIN |
| MP0184 | FLAT / BUY / NEUTRAL | 80 | 257.431 | 237.211 | +20.220 | +2.771 | EXPLORATORY_ROOM_INTERACTION_GAIN |
| MP0115 | FLAT / BUY / NEUTRAL | 80 | 256.070 | 235.909 | +20.160 | +2.696 | EXPLORATORY_ROOM_INTERACTION_GAIN |
| MP0145 | FLAT / BUY / NEUTRAL | 80 | 257.341 | 237.270 | +20.070 | +2.787 | EXPLORATORY_ROOM_INTERACTION_GAIN |
| MP0135 | FLAT / BUY / NEUTRAL | 80 | 257.095 | 237.457 | +19.639 | +2.238 | EXPLORATORY_ROOM_INTERACTION_GAIN |
| MP0128 | FLAT / BUY / NEUTRAL | 80 | 256.104 | 236.590 | +19.514 | +2.174 | EXPLORATORY_ROOM_INTERACTION_GAIN |
| MP0100 | FLAT / BUY / NEUTRAL | 80 | 256.262 | 236.760 | +19.502 | +1.986 | EXPLORATORY_ROOM_INTERACTION_GAIN |
| MP0124 | FLAT / BUY / NEUTRAL | 80 | 256.899 | 237.425 | +19.474 | +2.028 | EXPLORATORY_ROOM_INTERACTION_GAIN |
| MP0129 | FLAT / BUY / NEUTRAL | 80 | 255.658 | 236.315 | +19.343 | +2.223 | EXPLORATORY_ROOM_INTERACTION_GAIN |
| MP0175 | FLAT / BUY / NEUTRAL | 80 | 256.947 | 237.732 | +19.214 | +2.135 | EXPLORATORY_ROOM_INTERACTION_GAIN |
| MP0151 | FLAT / BUY / NEUTRAL | 80 | 256.860 | 237.812 | +19.048 | +1.526 | EXPLORATORY_ROOM_INTERACTION_GAIN |
| MP0181 | FLAT / BUY / NEUTRAL | 80 | 256.955 | 237.913 | +19.042 | +1.285 | EXPLORATORY_ROOM_INTERACTION_GAIN |

## Where the room-specific lifts occur

To avoid presenting many similar pairs in one small room as separate discoveries, this table shows the largest descriptive gain in each room among cells beating both global and offset controls. These leaders are selected after viewing development outcomes; they remain exploratory.

| 1d room | Pair | Features | Rows | Own global MAE | M4 MAE | Gain vs own global | Gain vs offsets | Gain vs best M2 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| FLAT / BUY / NEUTRAL | MP0120 | BTC-liq severity pct 15m + Liq acceleration pct 5m 15m | 80 | 256.167 | 231.075 | +25.093 | +7.577 | +18.059 |
| UP / BUY / NEUTRAL | MP0141 | Liq acceleration RZ 5m 15m + Liq acceleration pct 5m 15m | 570 | 169.359 | 152.532 | +16.826 | +0.271 | +5.606 |
| UP / BUY / SHORT-LIQ | MP0055 | BTC-liq activity 5m + Cross-liq activity 5m | 2,684 | 200.991 | 192.260 | +8.731 | +6.187 | +1.745 |
| FLAT / NEUTRAL / SHORT-LIQ | MP0055 | BTC-liq activity 5m + Cross-liq activity 5m | 279 | 231.016 | 226.226 | +4.790 | +0.995 | -4.246 |
| DOWN / NEUTRAL / NEUTRAL | MP0127 | BTC-liq severity RZ 15m + Liq temporal concentration pct 15m | 1,009 | 195.267 | 190.680 | +4.587 | +0.627 | +1.734 |
| DOWN / NEUTRAL / LONG-LIQ | MP0055 | BTC-liq activity 5m + Cross-liq activity 5m | 380 | 137.379 | 134.868 | +2.510 | +6.089 | -1.216 |
| DOWN / SELL / LONG-LIQ | MP0144 | Liq acceleration RZ 5m 15m + Cross-liq severity pct 15m | 2,994 | 184.291 | 182.382 | +1.909 | +0.917 | -0.621 |
| UP / NEUTRAL / SHORT-LIQ | MP0056 | BTC-liq activity 5m + BTC-liq activity 1h | 1,392 | 180.796 | 179.556 | +1.240 | +0.535 | -2.466 |
| DOWN / SELL / NEUTRAL | MP0090 | BTC-liq activity 1h + Liq side pressure pct 15m | 1,765 | 173.706 | 172.795 | +0.911 | +1.360 | -0.416 |

### A larger-sample candidate: MP0055 in UP / BUY / SHORT-LIQ

This pair combines BTC liquidation historical active rate5m and cross-asset liquidation historical active rate5m. On2,684 development rows in this1d room, global-pair MAE200.9910 falls to192.2600; the matched offset model scores198.4469. The gain is8.7310 log bps (4.3440%) versus the global pair and6.1869 beyond the offset model. It wins12/13 occupied months against each comparator. This is a more substantial support base than the largest raw room gain, which uses only80 development rows.

The14-day nominal95% interval for M4-minus-global MAE is[-13.8984,-3.6645]; versus offsets it is[-10.1005,-2.1902]. After simultaneous adjustment across the room search, the intervals become[-20.1998,+2.7377] and[-15.0581,+2.6843]. The positive upper bounds prevent a confirmed room lead under the frozen rule.

Per one training standard deviation, the fitted BTC5m slope changes from+45.6068 globally to-36.4220 in this room, while the cross-asset5m slope changes from-12.8472 to+41.6055. These are conditional coefficients of correlated measurements, not causal effects; their signs alone are not a validation result.

Against the previously best MP0004 model on these same2,684 rows, the comparison is194.0047 ->192.2600: a smaller1.7447 log-bps improvement (about0.90%). The larger4.34% gain above uses MP0055 own global pair as comparator. These are distinct benchmarks and must not be interchanged.

The full-population MP0055 interaction model still worsens MAE184.3011 ->185.6042. A new policy that uses its interaction only in this room would be selected from development and has not been independently confirmed or deployed.

## Step 7 — uncertainty and concentration

Use2,000 paired7-day and14-day chronological block resamples, seed48+days, weighting by observation count. Overall simultaneous bands cover570 pair-comparator contrasts (190 pairs x3). A separate room family covers5,320 pair-room-comparator contrasts (2,660 cells x2). Nominal intervals are also saved. Searching across rooms/pairs is explicitly counted; these corrections do not remove all prior development selection.

A credible pair must improve both its own global pair and matched M3; improve in a majority of months against both; have both block-length simultaneous upper bounds below zero for both; retain gains on both-values-valid rows; and retain gains after separately removing its two strongest gain months and two strongest gain rooms. A full advancement candidate must additionally beat best M2 with both simultaneous bands below zero.

A credible room cell must improve both own M2 and M3, win in a majority of occupied months against each, and have all four room-family simultaneous upper bounds below zero. Room leads remain development candidates; they do not authorize a forecast assembled by choosing development winners.

Concentration reports include largest one/two gain months, fixed March and March+August exclusions, and largest one/two gain rooms. These are zero-refit diagnostics; official scoring always retains all18,481 rows.

## Step 8 — independent replay and stop

Verification **PASS**: 380 fitted/reused models replayed; every prediction, room slope, training dual certificate, aggregate/month/room score, concentration statistic and both simultaneous uncertainty families reconstructed. Maximum prediction discrepancy 5.68e-14; maximum uncertainty discrepancy 8.53e-14. No verification fit.

**Step1 decision:** 0 credible pair candidates, 0 candidates against the common best M2, 0 credible room cells under the frozen rules. Stop and review these results. No4h or faster nested state was fitted.

All evidence is from previously exposed development. No verified untouched holdout exists. Even a passing candidate requires confirmation; a failed credibility gate does not erase a descriptive room gain.

## Saved artifacts and reproduction

- [Frozen contract](interaction_tournaments/v4_step1_20261031T105948_813637Z/CONTRACT.json), [fit plan](interaction_tournaments/v4_step1_20261031T105948_813637Z/FIT_PLAN.json), [result](interaction_tournaments/v4_step1_20261031T105948_813637Z/RESULT.json).
- [All190 pair scores](interaction_tournaments/v4_step1_20261031T105948_813637Z/pair_scoreboard.csv), [absolute ranking](interaction_tournaments/v4_step1_20261031T105948_813637Z/absolute_scoreboard.csv), [all outcomes](interaction_tournaments/v4_step1_20261031T105948_813637Z/all190_outcomes.csv), [concept groups](interaction_tournaments/v4_step1_20261031T105948_813637Z/concept_scoreboard.csv).
- [All room scores](interaction_tournaments/v4_step1_20261031T105948_813637Z/room_scoreboard.csv), [slopes by room](interaction_tournaments/v4_step1_20261031T105948_813637Z/room_slopes.csv), [training support checks](interaction_tournaments/v4_step1_20261031T105948_813637Z/pair_room_eligibility.csv).
- [Monthly scores](interaction_tournaments/v4_step1_20261031T105948_813637Z/monthly_scores.csv), [monthly room scores](interaction_tournaments/v4_step1_20261031T105948_813637Z/monthly_room_scores.csv), [valid/missing scores](interaction_tournaments/v4_step1_20261031T105948_813637Z/valid_missing_scores.csv), [concentration](interaction_tournaments/v4_step1_20261031T105948_813637Z/concentration.json).
- [Pair uncertainty](interaction_tournaments/v4_step1_20261031T105948_813637Z/uncertainty.csv), [room uncertainty](interaction_tournaments/v4_step1_20261031T105948_813637Z/room_uncertainty.csv), [optimizer certificates](interaction_tournaments/v4_step1_20261031T105948_813637Z/optimizer_certificates.csv).
- [M4 predictions](interaction_tournaments/v4_step1_20261031T105948_813637Z/interaction_predictions.parquet), [M3 predictions](interaction_tournaments/v4_step1_20261031T105948_813637Z/offset_predictions.parquet), [model coefficients](interaction_tournaments/v4_step1_20261031T105948_813637Z/models.json), [checksums](interaction_tournaments/v4_step1_20261031T105948_813637Z/ARTIFACT_HASHES.json).
- [Verification receipt](interaction_verification/v4_step1_20261031T105948_813637Z.json), [version history](ROOM_VERSION_HISTORY.json).
- Runner `REPEATABLE PLAN/tools/run_48h_room_interactions_v4.py`; exact executed copy in the run. Verifier `verify_48h_room_interactions_v4.py`.


Verification note: the first verifier attempt encountered decimal-string room IDs in saved exclusion lists (NumPy IDs serialized with JSON default=str). The verifier now decodes room identifiers using the recorded axis; every exclusion score replays exactly. Models, forecasts and results were unchanged. The initial note is preserved beside the final verification receipt.


</details>

### Incremental features V5 and top25

[Original document](<C:/Users/REDACTED_USER/Desktop/MFT ML/REPEATABLE PLAN/48H_ML/magnitude/08_INCREMENTAL_Z_V5.md>) ? SHA-256 `4356c49a6d6a7609348a35eb440972dd934434f06994df3873156e310c468f44`

<details>
<summary>Read complete preserved document</summary>

# 48h magnitude — V5: incremental Z across the522-feature universe

**Completed and independently verified: all522 accounted for; 254 qualified incremental features tested, 509 new fits, 0 failures.**

The fixed research setting is1d **UP / BUY / SHORT-LIQ**, with the selected BTC5m + cross-asset5m liquidation-activity model as the saved starting forecast. This room and pair were selected after earlier development searches.

Starting MAE **192.259952**. The matched two-feature room refit reproduces **192.259952**. The best product-interaction model adds **flow activity RZ5m**, scoring **190.903227**: a **1.356725 log-bps / 0.7057%** improvement. Its simple addition control scores **191.961900**.

98 additions beat the starting forecast; 82 interaction models beat it; 90 interactions beat their matched addition model; 54 beat anchor,refit and addition together. **0 pass the full frozen uncertainty/stability gate.**

## Version and authorization

V4 selected an exploratory pair-room candidate. The user then requested every incremental Z from the522 universe and a top25 report. V5 tests one Z at a time, not a combined25-feature model. V1–V4 remain unchanged. No4h room, new target horizon, tree, parameter sweep or automatic next experiment was run.

## Step1 — define the exact comparison

X1 is BTC liquidation historical active rate5m; X2 is cross-asset liquidation historical active rate5m. The saved V4 MP0055 forecast is the fixed comparator. Target stays absolute48h BTC endpoint log return in bps, scored by MAE.

The original chronological split is unchanged:29,615 training origins,96 purged origins,18,481 development origins. Within the fixed room, there are3,858 training and2,684 development origins. New parameters apply only inside this room. Training on its rows is mathematically equivalent to retaining fixed outside-room loss terms that cannot change those parameters. All2,684 room development rows are scored. Outside this room, saved V4 MP0055 forecasts remain exactly unchanged; full18,481-row prediction files are saved.

The room uses the same previously audited training-only cutoffs, availability gates and missing-context handling. Training labels finish before development begins. No new regime thresholds were fitted.

## Step2 — account for every candidate

| Classification | Features |
|---|---:|
| Existing qualification passed; incremental Z tested |254|
| Already one of the two X features |2|
| Failed the existing timing/coverage/variation qualification |266|
| Total accounted for |522|

Every excluded feature retains its original audit reason in `all522_outcomes.csv`. The request to test every feature did not silently turn unreplayed, constant or insufficient-history fields into qualified predictors.

For a previously qualified Z, the new room-specific gate requires at least50% valid training coverage,1,000 valid rows,20 distinct valid UTC dates,3 distinct values, rank4 for [1,X1,X2,Z] on all-three-present training rows and at least one additional identifiable product direction. All254 non-base qualified features passed. No development outcome was used for eligibility.

## Step3 — preserve preprocessing and missingness

Every X and Z reuses its original training-only imputation median, mean and standard deviation fitted on the full29,615 training rows. Each has a separate original-missing indicator. Missing observations remain in both train and development.

Products X1*Z and X2*Z use the standardized imputed values. For numerical conditioning only, each product is centered and scaled using the3,858 active training rows. With a free intercept and unpenalized coefficients, this is a reparameterization of the same product model. Product transforms, raw input support and all-three-present diagnostics are retained.

## Step4 — fit controls and explicit interactions

1. Preserve the saved V4 anchor forecast.
2. Fit one matched room-only base: `b0 + b1*X1 + bm1*M1 + b2*X2 + bm2*M2`. This reveals whether mere refitting causes a gain. Its development MAE matches the saved anchor.
3. For every Z, fit addition: `base main effects + bZ*Z + bmZ*MZ`.
4. For the same Z, fit interaction: `base main effects + bZ*Z + bmZ*MZ + g1*(X1*Z) + g2*(X2*Z)`.
5. Fit all coefficients jointly using the same unpenalized median/LAD solver and tolerances. No tuning, no development early stopping. Predict nonnegative magnitude.
6. Total fits:1 matched base +254 additions +254 interactions =509. Zero fit failures.

A change in pair weights is allowed in the augmented models; the addition and matched-base controls keep that from being mislabeled an interaction gain. All main effects are retained. Saved conditional slopes show how X1/X2 coefficients change at training Z10th/50th/90th percentiles. These are fitted relationships, not causation.

## Step5 — top25 interaction models

Ranked by interaction MAE on the same2,684 room development rows. Lower MAE is better. Gain is starting192.259952 minus interaction MAE; positive helps. The addition column exposes whether products earned their place.

| Rank | Z feature | Addition MAE | Interaction MAE | Gain vs start | Gain beyond addition | Months better vs start |
|---:|---|---:|---:|---:|---:|---:|
| 1 | flow activity RZ 5m | 191.9619 | 190.9032 | +1.3567 | +1.0587 | 7/13 |
| 2 | basis range RZ 15m | 192.0899 | 190.9750 | +1.2849 | +1.1148 | 9/13 |
| 3 | basis range pct 15m | 192.0460 | 191.0369 | +1.2230 | +1.0091 | 9/13 |
| 4 | flow activity RZ 15m | 192.2271 | 191.1653 | +1.0946 | +1.0618 | 7/13 |
| 5 | flow activity pct 15m | 192.0703 | 191.2602 | +0.9998 | +0.8101 | 7/13 |
| 6 | flow activity pct 5m | 191.9974 | 191.4273 | +0.8327 | +0.5701 | 7/13 |
| 7 | venue breadth pct 5m | 192.0519 | 191.6246 | +0.6354 | +0.4273 | 7/13 |
| 8 | range expansion pct 5m / 15m | 192.0069 | 191.6538 | +0.6062 | +0.3531 | 9/13 |
| 9 | range expansion pct 5m / 1h | 192.3171 | 191.6589 | +0.6011 | +0.6582 | 6/13 |
| 10 | range expansion RZ 5m / 15m | 192.0214 | 191.6879 | +0.5721 | +0.3335 | 9/13 |
| 11 | flow acceleration pct 5m / 15m | 192.0155 | 191.7119 | +0.5480 | +0.3036 | 10/13 |
| 12 | basis range RZ 1h | 192.2598 | 191.7141 | +0.5459 | +0.5457 | 6/13 |
| 13 | liq acceleration pct 5m / 15m | 192.1179 | 191.7236 | +0.5363 | +0.3943 | 11/13 |
| 14 | flow acceleration RZ 5m / 15m | 192.0217 | 191.7438 | +0.5162 | +0.2779 | 8/13 |
| 15 | price excursion range RZ 5m | 192.2078 | 191.7868 | +0.4732 | +0.4210 | 6/13 |
| 16 | liq acceleration pct 5m / 1h | 192.3329 | 191.8394 | +0.4205 | +0.4935 | 6/13 |
| 17 | vol acceleration RZ 15m / 1h | 192.1395 | 191.8569 | +0.4031 | +0.2826 | 7/13 |
| 18 | funding change pct 5m | 192.3454 | 191.9022 | +0.3577 | +0.4431 | 6/13 |
| 19 | price displacement RZ 15m | 191.7799 | 191.9089 | +0.3511 | -0.1289 | 7/13 |
| 20 | oi range position pct 15m | 192.0105 | 191.9109 | +0.3490 | +0.0996 | 9/13 |
| 21 | vol acceleration pct 15m / 1h | 192.1968 | 191.9177 | +0.3423 | +0.2792 | 7/13 |
| 22 | vol shock recency pct 15m | 192.3957 | 191.9190 | +0.3409 | +0.4767 | 10/13 |
| 23 | flow activity pct 1h | 192.1491 | 191.9239 | +0.3361 | +0.2253 | 6/13 |
| 24 | range expansion pct 15m / 1h | 192.2669 | 191.9247 | +0.3352 | +0.3422 | 7/13 |
| 25 | basis change pct 15m | 191.9298 | 191.9252 | +0.3348 | +0.0046 | 9/13 |

Several rows are alternate normalizations/windows of the same concept. They are not25 independent discoveries. Flow activity and basis range lead this interaction ranking. Price displacement RZ15m ranks among the top25 interaction scores, but its addition-only model performs better than its product model.

## Step6 — compare with simpler additions and inspect constant features

The best addition using a Z that actually varies in development is **cross asset liq activity raw historical active rate 15m**, MAE **190.882727**. That is slightly lower than the best interaction MAE190.903227. The best interaction is therefore not the best score across all tested models.

The raw lowest addition score is cross-asset liquidation historical active rate1h, MAE190.006884. That Z is constant at its saturated value throughout this room’s development period. Its gain can come from the trained base-weight/intercept change; it does not demonstrate new time-varying Z information in development. The constant flag is preserved, not hidden or used to rewrite the original eligibility rule. Its product model worsens sharply to200.806455.

## Step7 — monthly, missingness, concentration and uncertainty

Monthly scores, all-three-present/any-missing comparisons, and fixed March/March+August plus strongest one/two gain-month exclusions are saved for every candidate. No rows are removed from official scoring.

Uncertainty uses2,000 paired7d/14d calendar-block resamples anchored at2024-01-01 UTC. Empty calendar blocks are retained; resampled losses are weighted by active row counts. The simultaneous family covers1,270 comparisons:254 features x5 contrasts (addition/refit, addition/anchor, interaction/addition, interaction/refit, interaction/anchor).

The leading flow-activity RZ5m interaction wins7/13 months against both the starting forecast and its addition control. Its14d nominal95% interval for MAE change versus the starting forecast is[-3.7990,+0.5255]; the simultaneous interval is[-5.4393,+2.7259]. Against addition, the nominal interval is[-3.2755,+0.7513] and simultaneous[-4.8248,+2.7075]. Zero lies inside even the nominal intervals.

A credible interaction must improve anchor,refit and addition; improve in a majority of months versus anchor/addition; have simultaneous upper bounds below zero at both block lengths for allthree; retain all-three-present gains; survive removing two strongest gain months; and have a varying observed Z in development. No candidate passes the full rule. These corrections remain conditional on earlier room/base selection; no untouched holdout exists.

## Step8 — independent verification, versioning and stop

Independent verification PASS: all522 candidate outcomes checked;254 candidate inputs/product transforms and509 models replayed; training dual certificates verified; all active and full-development forecasts reconciled; monthly/concentration and all1,270 uncertainty contrasts replayed. Maximum prediction error 0; maximum uncertainty discrepancy 1.25e-12. No verification fit.

Conclusion: flow activity and basis range are exploratory interaction candidates in this selected room. The average gains are modest and remain uncertain. Addition alone is at least competitive. Stop here; do not combine winners, deploy a selected forecast or add another clock automatically.

## Saved results and reproduction

- [Frozen contract](incremental_z/v5_20261031T112240_951821Z/CONTRACT.json), [result](incremental_z/v5_20261031T112240_951821Z/RESULT.json), [all522 outcomes and exclusions](incremental_z/v5_20261031T112240_951821Z/all522_outcomes.csv).
- [All254 scores](incremental_z/v5_20261031T112240_951821Z/scoreboard.csv), [top25 interactions](incremental_z/v5_20261031T112240_951821Z/top25_interactions.csv), [top25 additions](incremental_z/v5_20261031T112240_951821Z/top25_additions.csv).
- [Monthly scores](incremental_z/v5_20261031T112240_951821Z/monthly_scores.csv), [uncertainty](incremental_z/v5_20261031T112240_951821Z/uncertainty.csv), [concentration](incremental_z/v5_20261031T112240_951821Z/concentration.json), [valid/missing scores](incremental_z/v5_20261031T112240_951821Z/valid_missing_scores.csv).
- [Conditional slopes](incremental_z/v5_20261031T112240_951821Z/conditional_slopes.csv), [coefficients](incremental_z/v5_20261031T112240_951821Z/models.json), [product preprocessing](incremental_z/v5_20261031T112240_951821Z/product_preprocessing.json).
- [Full development interaction forecasts](incremental_z/v5_20261031T112240_951821Z/interaction_full_predictions.parquet), [full addition forecasts](incremental_z/v5_20261031T112240_951821Z/addition_full_predictions.parquet).
- [Artifact hashes](incremental_z/v5_20261031T112240_951821Z/ARTIFACT_HASHES.json), [verification](incremental_z_verification/v5_20261031T112240_951821Z.json), [version history](ROOM_VERSION_HISTORY.json).
- Runner `REPEATABLE PLAN/tools/run_48h_incremental_z_v5.py`; exact executed copy saved in the run. Independent verifier `verify_48h_incremental_z_v5.py`.


</details>

### Candidate and confirmation plan

[Original document](<C:/Users/REDACTED_USER/Desktop/MFT ML/REPEATABLE PLAN/48H_ML/magnitude/09_CANDIDATE_CONFIRMATION_V1.md>) ? SHA-256 `76595a3330e2541327e12b7a2f218d2c599f86c0e4481ff5056c9e69a3b01ca1`

<details>
<summary>Read complete preserved document</summary>

# 48h magnitude: research candidate registration v1

Status: **exploratory forecast candidate, unconfirmed**. This registers the existing V5 result for tracking; it does not promote it to stage-13 candidate alpha or create model V6. No model was refitted and no new outcomes were opened.

## 1. What is frozen

Candidate `BTC48H_MAG_R1_FLOW5M_RESEARCH_V1` is V5 `Z0056__interaction`. In the 1d **UP / BUY / SHORT-LIQ** room, it combines BTC liquidation historical active rate 5m, cross-asset liquidation historical active rate 5m, and flow activity RZ 5m, including both liquidation-feature ? flow products and all main effects. It predicts the conditional median absolute 48h BTC endpoint log return, with nonnegative output.

The JSON card preserves exact coefficients, design order, training-only imputations/scaling, product transforms, split and hashed source references. Original runs remain unchanged. Missing indicators remain separate. Room construction, source-transition masks and outside-room forecasts inherit the frozen V5 procedure; a room label is not itself a trade instruction.

Training origins: 2022-04-22 00:30 to 2023-12-29 23:30 UTC (29,615). Boundary purge: 96. Development: 2024-01-01 00:00 to 2025-01-22 00:00 UTC (18,481). Selected room: 3,858 training and 2,684 development rows. Overlapping 48h outcomes are dependent.

## 2. What the saved evidence establishes

| Same-room procedure | Development MAE, log bps |
|---|---:|
| Saved room-conditioned pair | 192.259952 |
| Pair plus flow, addition only | 191.961900 |
| Pair plus flow and products | 190.903227 |
| Best varying-Z addition from the broader search | 190.882727 |

The interaction improves the anchor by 1.356725 log bps (0.7057%) and its matched addition by 1.058673. It wins 7/13 months. Its nominal 14-day block interval for MAE difference versus anchor is approximately [-3.799, +0.525]; zero is included. All 2,684 selected-room observations have valid Z, so this particular improvement is not from missing-value rows.

The broader search tested 254 incremental Z features after selecting the pair and room using development results. Zero interactions passed the full V5 credibility gate. A simpler varying-Z addition scored slightly better than this interaction. Thus neither unique interaction value nor economic alpha has been established. MAE is forecast error, not trade profit. The old development period cannot become untouched by renaming it.

## 3. How this fits your own framework

The Knowledge Bank's active `INVESTMENT_PROCESS.md` keeps risk, expected returns and costs as parallel inputs to portfolio construction. The dated context package explicitly says an absolute-endpoint-move forecast describes movement size, without supplying direction or a covariance matrix.

The 13-stage map requires robustness/search accounting (stage 9), a forecast-to-decision contract (10), realistic execution/capture (11), a joined evidence dossier (12), then candidate-alpha registration and a shadow protocol (13). Its thresholds are task-specific, not universal. This candidate is still in predictive evaluation; it has not passed stages 9?13. Calling it a research candidate now preserves the hypothesis without claiming those gates passed.

The second folder's current-state and proposed next-experiment pages describe a separate 1h study. Their process guidance applies; their target, scores and proposed next model do not replace this 48h contract. Source identities and hashes are recorded in the candidate card. These are internal process documents, not independent proof of performance or a fresh verification of every underlying book passage.

## 4. Next: confirm the forecast without another search

1. Finish the outstanding upstream normalization/availability lineage review. Existing exported-field checks and partial replays do not prove the entire raw-data pipeline is point-in-time correct.
2. Audit exposure records to identify genuinely unused later data. If none can be established, collect prospective timestamped forecasts. Do not inspect possible reserve outcomes before agreeing the test. No untouched holdout is currently verified.
3. Before scoring, freeze sample dates/end rule, active-room eligibility, minimum useful MAE improvement, minimum support/power assessment and treatment of missing forecasts. Those numerical choices remain pending; this note is not an executable final confirmation protocol.
4. Compare this frozen candidate with the room pair, its matched flow-addition model and the preselected stronger simple addition on identical observations. Fix the comparison family and dependence-aware uncertainty before opening outcomes. Retain the 7/14-day block sensitivity and monthly/concentration diagnostics; do not select a favorable block length afterward.
5. Save forecasts before labels mature, including issue time, feature availability, model version and fit cutoff. Attach outcomes after 48h. A fresh incremental benefit exceeding the predeclared useful-effect threshold with adequate support supports the forecast claim. Inconclusive evidence remains inconclusive; tuning after seeing it creates a new version needing new confirmation.

## 5. Separately: establish whether a decision can use it

Magnitude alone gives no long/short choice and is not realized path volatility, expected signed return or an option payoff forecast. Do not multiply separate unconditional direction and magnitude outputs to invent expected return.

For a trading use, first specify the instrument/venue and one economic hypothesis: for example, whether this forecast improves a separately justified strategy's exposure decisions. Freeze the entry, side rule, sizing/caps, exits and handling of overlapping 48h positions. If options are chosen, their quoted prices and payoff distribution require their own test; a magnitude forecast is insufficient.

Compare that fixed decision policy against the same strategy without this forecast, using realistic fees, spread/slippage, funding, delay, fills and positions. Report net PnL, drawdown, turnover, exposure, capacity and concentration. Forecast quality and net economic capture are distinct gates; better MAE cannot substitute for this comparison. Risk and cost inputs may be prepared in parallel with forecast confirmation.

Only after the required predictive and economic evidence survives should the stage-13 dossier be promoted, with a frozen shadow protocol and review/retirement rules. Shadow records must reconcile issued forecasts, intended decisions and executable prices/fills. Capital deployment is a separate later decision; none is made here.

## 6. Version and process record

- Research registration v1; existing fitted model V5 unchanged.
- Saved exact model/preprocessing and hashed artifact/source bindings in the linked JSON card.
- Updated this project's navigation and experiment log; external knowledge folders untouched.
- No new fits, holdout scoring, live collection service, order or trading system changes.

[Exact candidate card](candidates/BTC48H_MAG_R1_FLOW5M_RESEARCH_V1/CANDIDATE_CARD.json) ? [V5 results and full search](08_INCREMENTAL_Z_V5.md)


</details>

### Direction standalone/addition and top20

[Original document](<C:/Users/REDACTED_USER/Desktop/MFT ML/REPEATABLE PLAN/48H_ML/REPEATABLE PLAN dir/48H_ML/direction/01_DIRECTION_RESULT.md>) ? SHA-256 `5f74e065cda263eacda7820a7ee3dc87119886e001fde2b071c5a5fbc6310adf`

<details>
<summary>Read complete preserved document</summary>

# BTC 48h direction: verified development result

Same machine, changed Y: positive endpoint log return = UP, negative = DOWN, exact ties excluded (none in this population).
All 522 fields accounted for: 256 qualified and 266 skipped. 256 standalone fits; 217 log-loss improvers; 0 standalone leads.
All 23,436 unique pairs of the 217 improvers attempted: 23,436 successful, 0 failures. 11,941 improved beyond their better single parent. Pair labels: {'EXPLORATORY_INCREMENTAL_GAIN': 11940, 'NO_INCREMENTAL_LOG_LOSS_GAIN': 11495, 'DEVELOPMENT_ADDITIVE_LEAD': 1}.

| Forecast | Log loss | Brier | AUC diagnostic |
|---|---:|---:|---:|
| M0, frozen training P(UP)=0.5057909843 | 0.692023 | 0.249438 | 0.500000 |
| Best single by log loss | 0.685546 | 0.246215 | 0.547272 |
| Best pair by log loss (P3908) | 0.684958 | 0.245925 | 0.551011 |

Best single: `F4Z__volatility__realized_variance_level_percentile__4h__value`.
Best pair: `F4Z__basis__basis_persistence_percentile__1h__value` + `F4Z__volatility__realized_variance_level_percentile__4h__value`.
Best pair improves log loss by 1.021% against M0 and 0.086% against its fixed better parent, with 9/13 months better. Its label is EXPLORATORY_INCREMENTAL_GAIN.

## Advancement decision

No pair satisfies the predeclared useful-addition gate (pair-family uncertainty/stability rule and both metrics better than the global best single). Stop here. Product interactions, 1d rooms, nested 4h rooms and tree models were NOT RUN. No failed gate was relaxed after seeing the results.
M0 remains the reference; the best measured single and pair remain exploratory candidates. This is no evidence that BTC direction is inherently unpredictable. It is a failure to establish the required upgrade with this fixed search and reused history.
No verified untouched final holdout exists. Conditional family bands address this stage of comparisons; they do not erase previous feature, pair, regime or target selection. Do not present the selected development optimum as an unbiased future-performance estimate.

## Pairs passing the incremental rule before the global-best comparison

1 pair(s) pass against their own parents. None improves both primary metrics over the global best single. Statistical incremental significance and best absolute forecasting quality are different comparisons.

| Pair | Feature 1 | Feature 2 | Log loss | Brier | LL improvement vs better parent | Months better | Status |
|---|---|---|---:|---:|---:|---:|---|
| P14171 | `F4Z__funding__funding_change_percentile__1h__value` | `F4Z__oi__leverage_unwind_rz__5m__value` | 0.691629 | 0.249239 | 0.015% | 11/13 | DEVELOPMENT_ADDITIVE_LEAD |

## Fixed-model diagnostics

The best pair has 18,456 both-valid rows; its paired log-loss difference versus the fixed better parent there is -0.00057722 (negative means improvement).
After removing the best one/two calendar months by summed loss contribution, remaining log-loss differences are -0.00016985 / 0.00012213. These are diagnostics, not fresh tests.

## Best 20 standalone forecasts by development log loss

| Rank | Exact feature | Log loss | Brier | LL improvement vs M0 | Months better | Status |
|---:|---|---:|---:|---:|---:|---|
| 1 | `F4Z__volatility__realized_variance_level_percentile__4h__value` | 0.685546 | 0.246215 | 0.936% | 9/13 | EXPLORATORY_IMPROVEMENT |
| 2 | `F4Z__liquidation__liq_activity_severity_active_severity_rz__4h__value` | 0.685558 | 0.246222 | 0.934% | 8/13 | EXPLORATORY_IMPROVEMENT |
| 3 | `F4Z__liquidation__liq_activity_severity_active_severity_percentile__4h__value` | 0.685835 | 0.246359 | 0.894% | 8/13 | EXPLORATORY_IMPROVEMENT |
| 4 | `F4Z__volatility__realized_variance_level_rz__4h__value` | 0.686094 | 0.246485 | 0.857% | 8/13 | EXPLORATORY_IMPROVEMENT |
| 5 | `F4Z__price__price_excursion_range_percentile__4h__value` | 0.686263 | 0.246569 | 0.832% | 7/13 | EXPLORATORY_IMPROVEMENT |
| 6 | `F4Z__price__price_excursion_range_rz__4h__value` | 0.686709 | 0.246797 | 0.768% | 9/13 | EXPLORATORY_IMPROVEMENT |
| 7 | `F4Z__basis__basis_range_rz__4h__value` | 0.687365 | 0.247120 | 0.673% | 7/13 | EXPLORATORY_IMPROVEMENT |
| 8 | `F4Z__flow__flow_activity_rz__4h__value` | 0.687378 | 0.247121 | 0.671% | 8/13 | EXPLORATORY_IMPROVEMENT |
| 9 | `F4Z__flow__flow_activity_percentile__4h__value` | 0.687384 | 0.247124 | 0.670% | 8/13 | EXPLORATORY_IMPROVEMENT |
| 10 | `F4Z__liquidation__cross_asset_liq_activity_raw_active_severity_rz__4h__value` | 0.687772 | 0.247324 | 0.614% | 6/13 | EXPLORATORY_IMPROVEMENT |
| 11 | `F4Z__basis__basis_range_percentile__4h__value` | 0.687834 | 0.247351 | 0.605% | 7/13 | EXPLORATORY_IMPROVEMENT |
| 12 | `F4Z__liquidation__cross_asset_liq_activity_raw_active_severity_percentile__4h__value` | 0.688048 | 0.247460 | 0.574% | 6/13 | EXPLORATORY_IMPROVEMENT |
| 13 | `F4Z__basis__basis_level_rz__4h__value` | 0.688145 | 0.247503 | 0.560% | 6/13 | EXPLORATORY_IMPROVEMENT |
| 14 | `F4Z__liquidation__liq_acceleration_percentile__5m__4h__value` | 0.688169 | 0.247514 | 0.557% | 9/13 | EXPLORATORY_IMPROVEMENT |
| 15 | `F4Z__basis__basis_level_percentile__4h__value` | 0.688174 | 0.247517 | 0.556% | 6/13 | EXPLORATORY_IMPROVEMENT |
| 16 | `F4Z__oi__remaining_recent_oi_fraction_percentile__4h__value` | 0.688242 | 0.247551 | 0.546% | 8/13 | EXPLORATORY_IMPROVEMENT |
| 17 | `F4Z__liquidation__venue_breadth_percentile__4h__value` | 0.688309 | 0.247584 | 0.537% | 9/13 | EXPLORATORY_IMPROVEMENT |
| 18 | `F4Z__liquidation__liq_acceleration_percentile__15m__4h__value` | 0.688434 | 0.247646 | 0.519% | 9/13 | EXPLORATORY_IMPROVEMENT |
| 19 | `F4Z__oi__leverage_unwind_rz__4h__value` | 0.688444 | 0.247653 | 0.517% | 9/13 | EXPLORATORY_IMPROVEMENT |
| 20 | `F4Z__flow__signed_flow_acceleration_rz__15m__4h__value` | 0.688522 | 0.247690 | 0.506% | 9/13 | EXPLORATORY_IMPROVEMENT |

## Best 20 additive forecasts by absolute development log loss

| Pair | Feature 1 | Feature 2 | Log loss | Brier | LL improvement vs better parent | Months better | Status |
|---|---|---|---:|---:|---:|---:|---|
| P3908 | `F4Z__basis__basis_persistence_percentile__1h__value` | `F4Z__volatility__realized_variance_level_percentile__4h__value` | 0.684958 | 0.245925 | 0.086% | 9/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P3819 | `F4Z__basis__basis_persistence_percentile__1h__value` | `F4Z__liquidation__liq_activity_severity_active_severity_rz__4h__value` | 0.684967 | 0.245930 | 0.086% | 9/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P18560 | `F4Z__liquidation__liq_temporal_concentration_percentile__4h__value` | `F4Z__volatility__realized_variance_level_percentile__4h__value` | 0.685019 | 0.245954 | 0.077% | 8/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P2207 | `F4Z__basis__basis_level_rz__15m__value` | `F4Z__liquidation__liq_activity_severity_active_severity_rz__4h__value` | 0.685207 | 0.246064 | 0.051% | 8/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P14277 | `F4Z__funding__funding_level_percentile__15m__value` | `F4Z__liquidation__liq_activity_severity_active_severity_rz__4h__value` | 0.685247 | 0.246076 | 0.045% | 6/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P3817 | `F4Z__basis__basis_persistence_percentile__1h__value` | `F4Z__liquidation__liq_activity_severity_active_severity_percentile__4h__value` | 0.685252 | 0.246072 | 0.085% | 9/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P6758 | `F4Z__flow__flow_acceleration_percentile__5m__4h__value` | `F4Z__volatility__realized_variance_level_percentile__4h__value` | 0.685273 | 0.246079 | 0.040% | 11/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P9050 | `F4Z__flow__flow_activity_rz__4h__value` | `F4Z__volatility__realized_variance_level_rz__4h__value` | 0.685277 | 0.246083 | 0.119% | 9/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P7835 | `F4Z__flow__flow_acceleration_rz__5m__4h__value` | `F4Z__volatility__realized_variance_level_percentile__4h__value` | 0.685294 | 0.246090 | 0.037% | 11/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P16896 | `F4Z__liquidation__liq_activity_severity_active_severity_rz__4h__value` | `F4Z__liquidation__liq_temporal_concentration_percentile__4h__value` | 0.685296 | 0.246093 | 0.038% | 8/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P2819 | `F4Z__basis__basis_level_rz__5m__value` | `F4Z__liquidation__liq_activity_severity_active_severity_rz__4h__value` | 0.685296 | 0.246091 | 0.038% | 7/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P2296 | `F4Z__basis__basis_level_rz__15m__value` | `F4Z__volatility__realized_variance_level_percentile__4h__value` | 0.685311 | 0.246117 | 0.034% | 8/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P4105 | `F4Z__basis__basis_persistence_percentile__4h__value` | `F4Z__volatility__realized_variance_level_percentile__4h__value` | 0.685330 | 0.246108 | 0.032% | 7/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P20090 | `F4Z__oi__oi_extreme_recency_percentile__4h__value` | `F4Z__volatility__realized_variance_level_percentile__4h__value` | 0.685341 | 0.246113 | 0.030% | 7/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P6020 | `F4Z__flow__flow_acceleration_percentile__15m__4h__value` | `F4Z__volatility__realized_variance_level_percentile__4h__value` | 0.685352 | 0.246118 | 0.028% | 9/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P16975 | `F4Z__liquidation__liq_activity_severity_active_severity_rz__4h__value` | `F4Z__volatility__realized_variance_level_rz__5m__value` | 0.685368 | 0.246128 | 0.028% | 8/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P2705 | `F4Z__basis__basis_level_rz__4h__value` | `F4Z__volatility__realized_variance_level_percentile__4h__value` | 0.685381 | 0.246133 | 0.024% | 5/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P2908 | `F4Z__basis__basis_level_rz__5m__value` | `F4Z__volatility__realized_variance_level_percentile__4h__value` | 0.685382 | 0.246135 | 0.024% | 7/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P7658 | `F4Z__flow__flow_acceleration_rz__5m__1h__value` | `F4Z__volatility__realized_variance_level_percentile__4h__value` | 0.685384 | 0.246135 | 0.024% | 9/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P3511 | `F4Z__basis__basis_path_retention_percentile__4h__value` | `F4Z__volatility__realized_variance_level_percentile__4h__value` | 0.685390 | 0.246137 | 0.023% | 9/13 | EXPLORATORY_INCREMENTAL_GAIN |

## Largest 20 incremental gains over each pair's fixed better parent

| Pair | Feature 1 | Feature 2 | Log loss | Brier | LL improvement vs better parent | Months better | Status |
|---|---|---|---:|---:|---:|---:|---|
| P23194 | `F4Z__volatility__realized_variance_level_rz__1h__value` | `F4Z__volatility__vol_acceleration_rz__1h__4h__value` | 0.686803 | 0.246836 | 0.272% | 9/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P22949 | `F4Z__volatility__range_expansion_rz__1h__4h__value` | `F4Z__volatility__realized_variance_level_rz__1h__value` | 0.686819 | 0.246844 | 0.268% | 8/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P23100 | `F4Z__volatility__realized_variance_level_percentile__1h__value` | `F4Z__volatility__vol_acceleration_rz__1h__4h__value` | 0.686859 | 0.246864 | 0.264% | 8/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P15816 | `F4Z__liquidation__liq_acceleration_percentile__1h__4h__value` | `F4Z__liquidation__liq_activity_severity_active_severity_percentile__1h__value` | 0.686787 | 0.246830 | 0.264% | 8/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P23188 | `F4Z__volatility__realized_variance_level_rz__1h__value` | `F4Z__volatility__vol_acceleration_percentile__1h__4h__value` | 0.686861 | 0.246865 | 0.263% | 9/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P16416 | `F4Z__liquidation__liq_acceleration_rz__1h__4h__value` | `F4Z__liquidation__liq_activity_severity_active_severity_percentile__1h__value` | 0.686809 | 0.246841 | 0.261% | 8/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P22945 | `F4Z__volatility__range_expansion_rz__1h__4h__value` | `F4Z__volatility__realized_variance_level_percentile__1h__value` | 0.686886 | 0.246878 | 0.258% | 8/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P22344 | `F4Z__price__price_excursion_range_percentile__1h__value` | `F4Z__volatility__vol_acceleration_rz__1h__4h__value` | 0.686908 | 0.246889 | 0.257% | 8/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P23094 | `F4Z__volatility__realized_variance_level_percentile__1h__value` | `F4Z__volatility__vol_acceleration_percentile__1h__4h__value` | 0.686904 | 0.246887 | 0.256% | 8/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P22324 | `F4Z__price__price_excursion_range_percentile__1h__value` | `F4Z__volatility__range_expansion_rz__1h__4h__value` | 0.686915 | 0.246893 | 0.254% | 8/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P22748 | `F4Z__volatility__range_expansion_percentile__1h__4h__value` | `F4Z__volatility__realized_variance_level_rz__1h__value` | 0.686935 | 0.246902 | 0.252% | 9/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P22338 | `F4Z__price__price_excursion_range_percentile__1h__value` | `F4Z__volatility__vol_acceleration_percentile__1h__4h__value` | 0.686955 | 0.246912 | 0.249% | 9/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P22744 | `F4Z__volatility__range_expansion_percentile__1h__4h__value` | `F4Z__volatility__realized_variance_level_percentile__1h__value` | 0.686962 | 0.246915 | 0.248% | 9/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P15818 | `F4Z__liquidation__liq_acceleration_percentile__1h__4h__value` | `F4Z__liquidation__liq_activity_severity_active_severity_rz__1h__value` | 0.686893 | 0.246882 | 0.248% | 9/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P23204 | `F4Z__volatility__realized_variance_level_rz__1h__value` | `F4Z__volatility__vol_shock_recency_percentile__4h__value` | 0.686972 | 0.246920 | 0.248% | 9/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P7304 | `F4Z__flow__flow_acceleration_rz__1h__4h__value` | `F4Z__volatility__realized_variance_level_rz__1h__value` | 0.686856 | 0.246863 | 0.248% | 9/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P1702 | `F4Z__basis__basis_level_percentile__1h__value` | `F4Z__basis__basis_level_rz__15m__value` | 0.689112 | 0.247998 | 0.247% | 8/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P16486 | `F4Z__liquidation__liq_acceleration_rz__1h__4h__value` | `F4Z__price__price_excursion_range_percentile__1h__value` | 0.686915 | 0.246892 | 0.246% | 9/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P22318 | `F4Z__price__price_excursion_range_percentile__1h__value` | `F4Z__volatility__range_expansion_percentile__1h__4h__value` | 0.686988 | 0.246929 | 0.245% | 9/13 | EXPLORATORY_INCREMENTAL_GAIN |
| P7279 | `F4Z__flow__flow_acceleration_rz__1h__4h__value` | `F4Z__price__price_excursion_range_percentile__1h__value` | 0.686881 | 0.246876 | 0.245% | 9/13 | EXPLORATORY_INCREMENTAL_GAIN |

## Evidence and limitations

Every fit retained 29,615 training origins and 18,481 development origins, with the same 96-origin boundary purge. The direction target was rebuilt from signed returns, not the old extreme `y` column. Preprocessing was training-only; class weights, C, solver and settings stayed fixed. No development tuning or changed-parameter retries.
Existing label-blind qualification was reused after checking hashes and identical keys/splits/returns. Exported metadata and bounded normalization replay do not fully establish actual exchange publication latency or reconstruct upstream F2. Original PROVISIONAL source limitations remain.
Separate verification rebuilt inputs, probabilities, every month, parent comparisons and the primary simultaneous block intervals. Convex objective residuals were checked. Verification made zero additional fits. Brier uncertainty for singles was added as a diagnostic without changing the inherited rule.
All original source-workspace artifacts are unchanged. New direction artifacts live in this linked branch because the original MFT ML folder is outside the current writable root.

[Standalone report](tournaments/direction_all522_20261031T112228_526278Z/REPORT.md) · [Pair report](pair_tournaments/direction_additive_20261031T112454_940677Z/REPORT.md) · [All pair outcomes](pair_tournaments/direction_additive_20261031T112454_940677Z/all_pair_outcomes.csv)
[Single verification](tournament_verification/direction_all522_20261031T112228_526278Z.json) · [Uncertainty reconstruction](tournament_verification/direction_all522_20261031T112228_526278Z_uncertainty.json) · [Pair verification](pair_verification/direction_additive_20261031T112454_940677Z.json)
[Forecast object contract](review/FORECAST_CONTRACT.json) · [P(UP)/P(DOWN) candidate forecasts](review/candidate_direction_forecasts.parquet) · [Process](00_PROCESS.md) · [Research state](research_state.json) · [Events](experiment_log.jsonl)


</details>

### Trading-rule replay

[Original document](<C:/Users/REDACTED_USER/Desktop/MFT ML/REPEATABLE PLAN/48H_ML/backtests/01_RULE_V1_RESULT.md>) ? SHA-256 `98e63a2dffff09c04f8059fed9df59ca409550e4fe17c31cb9c87b00a303e087`

<details>
<summary>Read complete preserved document</summary>

# 48h trading-rule development replay v1

**Decision: exploratory price-proxy result, not confirmed tradable alpha.** No model refits, threshold tuning, extremes filtering or holdout evaluation. The user asked to backtest the earlier rule using the available direction and magnitude candidates.

## What the three forecasts mean

Direction predicts P(positive 48h endpoint return). Magnitude predicts median absolute 48h endpoint log return. Extremes predicts P(absolute endpoint log return >500bps), combining up and down tails. None alone specifies a position or expected net profit. Do not multiply the unconditional direction and magnitude outputs to infer expected signed return. The extreme branch remains parked; it is not silently promoted into a trading filter.

The latest saved direction continuation was checked: P9050_I interaction log loss0.684953, its additive parent0.685277; neither earned advancement. T20 XGBoost scored0.707316, worse than M0 0.692023. The simulation uses the user-discussed P9050_I, not a new winner selected by trading PnL. Magnitude is the frozen V5 Z0056 interaction candidate.

## Frozen rule and shared opportunities

Use the existing 18,481 development origins from 2024-01-01 through 2025-01-22. Restrict to 1d UP/BUY/SHORT-LIQ. Earliest available room origin starts a shared 48h reservation; filter-rejected slots stay reserved, with no replacement during that interval. This makes four policies act on the same opportunities rather than allowing different entry schedules to obscure the filter comparison. It is a controlled opportunity replay, not unrestricted re-entry at every qualifying bar.

R = room long; RD = room long only if P(UP)>0.5; RM = room long only if predicted magnitude>300logbps; RDM = both filters. Side remains long for this experiment; probability below50% skips, never shorts. Thresholds were frozen before PnL access. The earlier example's 300bps is an illustrative hypothesis, not a calibrated profitable cutoff.

Entry uses the VWAP for the minute after the signal; exit proxy is exactly48h after that entry timestamp. Thus execution-proxy endpoints are shifted one minute from the model label. Allocate5% of current account equity at entry, one cash-long position, no leverage, no stop/TP, cash earns zero. Trading costs are assumed5bp fee plus5bp spread/slippage per side, charged to actual entry/exit notionals (about20bp round trip). These are not measured venue costs. No funding is included because this is a cash-long proxy; it is not a perpetual-futures simulation.

## Results

Returns are compounded account returns across the entire development interval, not annualized. Drawdown uses the minute marked-to-market equity path. Trade bps are simple net returns on entry notional; the forecast remains in log bps.

| Policy | Trades | Net account return | Maximum drawdown | Mean net trade bps | Average account exposure |
|---|---:|---:|---:|---:|---:|
| R | 92 | 1.330% | -2.033% | 29.10 | 2.368% |
| RD | 85 | 1.680% | -1.913% | 39.58 | 2.188% |
| RM | 6 | 1.175% | -0.378% | 390.73 | 0.157% |
| RDM | 6 | 1.175% | -0.378% | 390.73 | 0.157% |

Room opportunities:92. The direction filter skips7:4 losing trades and3 winning trades, with mean net return -98.14bps among the skipped trades. This improves the observed account return by about0.35percentage points. The descriptive14-day paired block interval for mean daily return improvement versus R includes zero: [-0.0500,+0.2425] account bps/day. It is not selection-adjusted or fresh confirmation.

The magnitude filter keeps6 trades,5 net winners and1 loser. Every one also passes direction, so RM and RDM are identical. The direction model adds no discrimination among those6 entries. The86 skipped opportunities contain44 winners and42 losers, with mean net trade return +3.87bps. Much lower exposure explains part of the lower drawdown; this is not an equal-risk comparison.

Two February2024 trades dominate the magnitude-filtered result. Removing those two as a concentration diagnostic leaves **-0.0520%** compounded account return. This diagnostic is not a newly selected strategy. There are no magnitude-filtered entries after July2024 in this shared calendar. Six trades cannot establish a stable combined edge.

## Data and execution limits

Canonical f066 is cross-venue minute trade VWAP, explicitly not a candle close or executable quote (team dataset specification, Price authority section). It does not establish that an order can fill at this price, especially across venues. Costs are scenarios and capacity is untested. No claim about live fills, venue-specific spot returns, funding or net perpetual PnL follows.

All required traded price paths were checked for valid, positive values and availability; none was silently excluded based on later outcomes. Price reads were bounded to the existing development interval plus the48h exit and1min delay. No reserved-period outcomes were accessed.

The original models, room and thresholds were selected or proposed after extensive development work. This replay is additional development research. No untouched holdout is verified. Statistical forecast gains and these hypothetical account returns do not establish tradable alpha. The CFTC also notes that simulated results are affected by hindsight and may misrepresent liquidity/execution effects: https://www.cftc.gov/LawRegulation/FederalRegister/FinalRules/e7-3122.html

## Verification and saved evidence

Saved contract before price/PnL access; exact source hashes; 18,481 aligned forecasts; shared opportunity ledger; trade ledger; minute equity; daily/monthly returns; filter diagnostics and descriptive block intervals. Checks confirm entries after signals,48h holds, no overlapping slots, and valid price paths. An independent compounded-trade calculation reconciles all four minute-equity terminal balances. No model fit was performed.

## Next decision

Keep this version frozen. The present result does not justify adding an extremes threshold or optimizing300bps. The next useful trading check needs a chosen venue/instrument with executable bid/ask and fee data, and a predeclared fresh-data evaluation. If there is no unexposed history, timestamped prospective shadow forecasts provide the clean route. Keep the existing model's conditional-median magnitude meaning; expected-return/position sizing requires a separate economic model.

[Run contract](rule_v1_20261031T115141_997832Z/CONTRACT.json) | [Trade ledger](rule_v1_20261031T115141_997832Z/trades.csv) | [Verification](rule_v1_20261031T115141_997832Z/VERIFICATION.json)


</details>

### Direction source relocation and later continuation

At compilation, the direction branch is now under `REPEATABLE PLAN/48H_ML/REPEATABLE PLAN dir/48H_ML/direction` in this workspace. Earlier notes refer to its former Knowledge Bank location. The current files were retrieved from the new location; preserved historical source links may still name the former path.

- [CONTRACT.json](<C:/Users/REDACTED_USER/Desktop/MFT ML/REPEATABLE PLAN/48H_ML/REPEATABLE PLAN dir/48H_ML/direction/complexity_runs/top20_20261031T114338_667303Z/CONTRACT.json>)
- [interaction_scores.csv](<C:/Users/REDACTED_USER/Desktop/MFT ML/REPEATABLE PLAN/48H_ML/REPEATABLE PLAN dir/48H_ML/direction/complexity_runs/top20_20261031T114338_667303Z/interaction_scores.csv>)
- [interaction_decisions.json](<C:/Users/REDACTED_USER/Desktop/MFT ML/REPEATABLE PLAN/48H_ML/REPEATABLE PLAN dir/48H_ML/direction/complexity_runs/top20_20261031T114338_667303Z/interaction_decisions.json>)
- [tree_scores.csv](<C:/Users/REDACTED_USER/Desktop/MFT ML/REPEATABLE PLAN/48H_ML/REPEATABLE PLAN dir/48H_ML/direction/complexity_runs/top20_20261031T114338_667303Z/tree_scores.csv>)
- [tree_decisions.json](<C:/Users/REDACTED_USER/Desktop/MFT ML/REPEATABLE PLAN/48H_ML/REPEATABLE PLAN dir/48H_ML/direction/complexity_runs/top20_20261031T114338_667303Z/tree_decisions.json>)

Compilation 2026-10-31T12:36:19.507735+00:00. Documentation-only; no new fits, no model changes, no new holdout access. This is not a fresh replay of every historical model.

## Later continuation: random forest full256 v2

One fixed magnitude comparison; no tuning. Scores: {"M0": 187.83107169705042, "pair": 182.03697721076213, "RF2": 189.64267684249788, "RF": 194.98150405121555}. Standard average of leaf medians; exploratory. [Full report](<C:/Users/REDACTED_USER/Desktop/MFT ML/REPEATABLE PLAN/48H_ML/magnitude/random_forest/v2_full256_20261031T161111Z/REPORT.md>).

## XGBoost controlled training-only tuning

{"M0": 187.83107169705042, "pair": 182.03697721076213, "XGB": 188.6110034053413}.[Report](<C:/Users/REDACTED_USER/Desktop/MFT ML/REPEATABLE PLAN/48H_ML/magnitude/xgb_training_tuned/v1_20261031T163853Z/REPORT.md>).19 fits, no new holdout.
