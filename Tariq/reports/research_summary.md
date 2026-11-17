# ScaleWave research summary

Prepared 17 November 2026. Internal company research handoff.

## What did we investigate?

Two different questions about BTCUSDT: how large the price change is after 48 hours, and how long it takes to reach the highest price within a future window.

## What improved?

The 48-hour magnitude model improved from MAE 182.036977 to 180.798518 log bps on the same 18,481 development rows. That is approximately 0.68% lower average error. The current Spearman rank correlation is 0.078527. Saved models and required pair cards are retained.

## What did not work well?

Many magnitude features and alternative models failed to improve the current benchmark. Shorter training windows also lost. Timing's selected-tree policy made small improvements at 15m-16h, but lost at 48h-300h. A simple baseline can be difficult to beat.

## What did timing research complete?

Three baselines at ten horizons; a sampled coverage audit; a 315-fit pilot; and a staged screen of 386 individual features at each of ten horizons. The staged method shortlisted features using earlier validation before choosing a single-feature tree. It was not an exhaustive search or a feature-combination model.

## What remains uncertain?

Small development gains do not prove future profits or statistical significance. Feature publication timing remains unresolved. Local timing checks verify saved receipts, not raw-data/model replay. The timing reserve was excluded from timing, but some of its dates were used by earlier magnitude research. Some old losing models were intentionally pruned.

## What should happen next?

Investigate why longer-horizon selection missed gains seen in individual cheap models. Prepare a small, explicit set of alternative models and transformations, especially for 300h. Do this before combinations or the next target. No follow-up training has started. Connected specialist models are future work and must use chronological out-of-sample upstream predictions.

## What can a colleague do with this repository?

Read the explanations, inspect source and experiment history, check saved timing scores, replay available magnitude models with approved inputs, and continue from the recorded next step. Original datasets stay outside GitHub and require separate permission.

See the root README for links to results, reproduction instructions and the beginner GitHub guide.
