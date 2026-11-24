# XGBoost training-only tuning:48h magnitude

Corrected admission plus bounded HP search. Three expanding training folds;48h boundary purge;6 settings;18 CV fits+1 final fit. No development early stopping or tuning. Median objective reg:quantileerror,alpha0.5.

```json
{
  "selection": {
    "config": 2,
    "selected_parameters": {
      "max_depth": 2,
      "reg_lambda": 200
    },
    "pooled_CV_MAE": 140.0023922430209,
    "pooled_CV_M0_MAE": 146.40722228515958,
    "selection_frozen_before_final_scoring": true
  },
  "final_features": 384,
  "scores": {
    "M0": 187.83107169705042,
    "pair": 182.03697721076213,
    "XGB": 188.6110034053413
  },
  "months_better_M0": 4,
  "months_better_pair": 4,
  "uncertainty": [
    {
      "days": 7,
      "comparator": "M0",
      "low": -0.20963446344339248,
      "high": 1.7133485390673762
    },
    {
      "days": 7,
      "comparator": "pair",
      "low": 2.254199829945522,
      "high": 11.165854093940148
    },
    {
      "days": 14,
      "comparator": "M0",
      "low": -0.30583260677112334,
      "high": 1.8453980853864342
    },
    {
      "days": 14,
      "comparator": "pair",
      "low": 2.2328024219774574,
      "high": 10.82018555300219
    }
  ],
  "new_fits": 19,
  "holdout": "NONE_VERIFIED",
  "verification": "Exact18481 keys and saved-model replay PASS",
  "status": "EXPLORATORY_DEVELOPMENT_RESULT"
}
```

Admission is recalculated from each fold, including final48h train;1h counts are not copied. Existing masked export cache reused; upstream funding-boundary uncertainty remains. New procedure changes both admission and HP selection relative to historical models. Saved pair remains selected on development, so comparisons are exploratory.

Independent verification: all 18 saved CV fits, 18,481 exact development keys and targets, finite forecasts, recomputed MAE, fold label-end separation and final admission counts passed. See VERIFICATION.json.

Conclusion: selected XGBoost did not improve later MAE. Training CV improvement did not transfer. It beat M0 and the saved pair in only 4/13 months. Keep the existing pair as the stronger development comparator; no untouched holdout is established.
