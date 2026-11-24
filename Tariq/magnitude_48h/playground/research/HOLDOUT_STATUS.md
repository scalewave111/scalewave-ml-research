# Holdout status

**No verified untouched final holdout is available in this workflow.**

The 18,481 later observations beginning 2024-01-01 were excluded from fitting baseline constants, imputation and model coefficients. They were nevertheless included in prior discovery screens and subsequently inspected through scores, feature bins and model comparisons. They now serve explicitly as development assessment/model-selection data.

Calling them a final holdout or splitting already inspected history again would misstate their role.

## Withheld data mentioned by the source package

The supplied F4/F4Z package states that it contains 1,008 daily partitions from 2022-04-22 through 2025-01-23, with the last shared minute ending at 2025-01-24 00:00 UTC. Its README mentions 112 withheld daily partitions from 2025-01-24 through 2025-05-15.

Those withheld partitions are not supplied in that package. Their accessibility and previous research exposure have not been verified. A package exclusion alone is not proof of an untouched confirmation period.

The tournament does not open, infer, synthesize or score withheld labels. A future holdout should be registered before its outcomes are inspected.

## Before final confirmation

1. Establish the dataset, coverage, provenance and prior exposure of a genuinely later period using metadata first.
2. Lock the feature set, preprocessing, model family/settings, fitting policy, target, baseline, metrics and decision rule before opening its outcomes.
3. Apply timestamp-based label purging at the final boundary and specify the last admissible label endpoint.
4. Fit using only the allowed preceding data, then evaluate once under the locked protocol.
5. Record the outcome, including a failure. If the holdout is subsequently used to redesign the model, reclassify it as development and require a new confirmation period.

These are preparation rules. No final holdout has been registered or evaluated yet.
