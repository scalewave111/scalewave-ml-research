RUN:
  python .\Code\test_mean_feature_linearity_48h.py --interactions

For curvature tests only, omit --interactions.

SCOPE
Current UP3features: liquidation activity15m, activity5m,4h/24hvolatility ratio.
Current DOWN7features: same3 plus168hvolatility,1hvenuebreadth,5mfunding
persistence,1hbasislevel. Full column names in CONTRACT.json.
This tests relationships to conditional mean MAGNITUDE, not signed-return
forecasting. UP and DOWN evaluated separately on their actual historical sides.
No direction model is needed. No trading or signed-return changes.

METHOD
Baseline: all current side features enter additively/linearly with missing flags.
For each feature: replace its linear term with a cubic spline; keep all other
features linear. Use3or5training-quantile knots; duplicates collapsed. A constant
feature falls back to its linear representation, recorded in FIT_AUDIT.json.
Optional: add each pair's product of standardized inputs to the additive model.
3UPpairs+21DOWNpairs. This detects this particular interaction form, not every
possible threshold, higher-order, or nonlinear interaction.

Training-only imputation/scaling and spline knots; derived design columns scaled
on training data to make Ridge penalties comparable. Alpha-per-row candidates
.001/.01/.1/1. Spline complexity and alpha selected using row-weighted MSE across
3earlier windows. The additive baseline is tuned independently using the SAME
windows. This diagnostic baseline is not necessarily identical to the previously
frozen alpha.01 Ridge models. Targets are ordinarybps after per-row conversion
from canonical log labels. Mean-target squared error; nonnegative predictions.

Quarterly outer evaluations Jan/Apr/Jul/Oct2024, same prepared development rows.
Training labels end strictly before each cutoff. Validation origins first2months
after cutoffs9/6/3months earlier; all validation labels mature before outer issue.
Feature sets themselves were previously researched on exposed history.

UNCERTAINTY / INTERPRETATION
Paired MSE gain = linear squared error minus challenger squared error, averaged
on IDENTICAL later-side observations. Positive means challenger improved.
Calendar7day blocks are resampled within each outer period, preserving paired
predictions and within-block dependence. Report1000bootstrap draws by default.
Intervals are pointwise95percent, exploratory and NOT multiple-test-adjusted.
Block length does not guarantee removal of all serial dependence.
--block-days14 can be used for sensitivity; --bootstrap controls draws.

Summary conclusion:
- evidence_of_useful_nonlinearity / interaction: lower interval bound>0 and
  improvement in at least3of4periods;
- linear_preferred_in_this_test: upper gain interval bound<0;
- inconclusive: otherwise.
None of these proves a feature's true relationship is exactly linear or causal.
Correlated features can substitute for one another. A marginal curved plot can
reflect other correlated inputs; the conditional model comparison is distinct.

PLOTS
For each feature, create4period panels. Bin boundaries come from eligible
side-specific training quantiles. Later actual mean magnitude and sample counts
are displayed per bin, with pointwise calendar-block uncertainty. Missing raw
values are excluded from plots but retained/imputed in model comparisons.
Tied discrete features may produce fewer occupied bins. These are marginal
plots, not partial-dependence plots or model predictions.

OUTPUT: Code/experiments/mean_feature_linearity_48h_v1/<timestamp>/
  SUMMARY.txt / linearity_summary.csv: main results and exploratory conclusions
  comparison_by_period.csv: fold-level wins/errors
  binned_relationships.csv / plots/*.png: curves, sample counts and uncertainty
  conditional_predictions.csv: matched actual and predicted side magnitudes
  validation_metrics.csv / SELECTIONS.json: earlier-only choices
  FIT_AUDIT.json / *.joblib: saved models, cutoff audits, hashes, replay checks
  CONTRACT.json / source snapshots / COMPLETE.json: reproducibility

Curvature-only:1056inner+48outer small Ridge/spline fits.
With all pair interactions:2208inner+144outer fits; CPU runtime depends on machine.
--threads2default. This runner has no resume mode. Every run creates a new folder.
No real-data execution is implied by creating the runner or synthetic tests.
