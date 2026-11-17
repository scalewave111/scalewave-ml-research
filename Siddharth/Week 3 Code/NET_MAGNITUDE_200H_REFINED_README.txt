200H NET MAGNITUDE: MAE-FOCUSED FEATURE/BLEND REFINEMENT, ALL FOUR FAMILIES

cd "C:\Users\Siddh\Desktop\ML\November\Week 3"
python Code/fit_net_magnitude_200h_refined.py

Defaults to independent HGB/ExtraTrees/MLP/GRU runs. To focus on the previous
candidate winners: add --families mlp hgb. No model families are mixed.
Input audit: --check-inputs. Non-fitting synthetic checks:
python Code/check_net_magnitude_200h_refined.py

CHANGES
Previous new-feature candidates: MLP MAE375.150209/RMSE534.426478,
HGB376.890379/531.025201; mean377.135789/499.616543. Both beat mean MAE
slightly, failed earlier-validation guards and lost mean RMSE. Those exposed
results motivate this exploratory refinement; no independent confirmation.

Original3packs retained: volatility_endpoint30,trend_range36,slow_market54.
Fourth endpoint_refined48 adds12new features to trend_range:
- Q50/Q75/Q90 of past200h absolute endpoint magnitudes over200/720completed
  hourly endpoints (6features). Historical overlapping blocks; no future Y.
- Hourly squared-return energy divided by5min squared-return energy across
  100/200/400/720h (4features); persistence/aggregation hypotheses.
- Per-hour5minRV100/720 and400/720 ratios (2features).
Total80distinct inputs. Exact formulas in net_200h_refinement_features.py
and saved feature dictionary; no outcome-driven individual feature ranking.

Finer head weights0/.25/.5/.75/1; model weights .05/.1/.2/.25/.35/.5/.65/
.75/.9/1 versus training mean/median. Blends reuse fitted predictions.
Rank candidate and accepted configurations by earlier pooled MAE, with the
worst relative MAE/RMSE score as tie-break. This explicitly changes the
previous joint-score ranking to match current MAE focus. RMSE still reported.
Acceptance requirements are unchanged: .1%gain versus EACH mean/median MAE
and mean RMSE,3/4windowwins/ties,no>2%windowloss for each comparison. If none
qualifies selected remains mean; candidate remains a separate diagnostic.
No retrospective promotion of previous candidates or guard relaxation.

Everything else retains the200h feature-search methodology: canonical12000-
minute absolute endpoint net target,checksum/local receipt,strict200h
training/validation maturity,4earlier3monthwindows15/12/9/6months,fulltimeline
GRU60/40/20/0h snapshots,train-onlypreprocessing,pairedL1/L2heads,fast2000cap,
same estimator budgets. See NET_MAGNITUDE_200H_FEATURE_SEARCH_README.txt.
512inner head fits across4families;up to64outer with one transform. Finer
blend grids add arithmetic,not fits;extra featurepack increases fit budget.
--families mlp hgb uses256inner head fits. Full budgets768across4families.
Optional --full-search and --transforms raw signedlog quadratic remain.
No automatic resume/runtime guarantee. No training started by assistant.

NEW OUTPUTS, OLD RESULTS PRESERVED
Code/experiments/net_magnitude_200h_<family>_refined_fast_v1/<timestamp>/
Report/net_magnitude_200h_<family>_refined_fast_v1/<timestamp>/
Full budgets use refined_v1. Same hashes,contracts,trials,feature metadata,
pairedmodels/serialization replay,predictions,metrics,COMPLETE workflow.

STATUS
Syntax/non-fitting synthetic endpoint/refinement causality and prefix,
formulas,quantile ordering,finite coverage,source loading,blend/guard checks
passed. Actual input audit status recorded in PROJECT_CONTEXT.md.
No real-data fitting/training/inference or improved score established.
Further MAE/RMSE improvement is unverified; user runs the experiment.

Actual --check-inputs PASSED:47088eligible/17473evaluation timestamps;
80distinct features,packs30/36/54/48,all16validation histories and actual
prefix/future checks passed. No model fitting performed.
