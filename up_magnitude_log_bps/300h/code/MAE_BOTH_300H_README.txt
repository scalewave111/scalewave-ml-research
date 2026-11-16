300h UP/DOWN MAE refinement

Run from PowerShell:
cd "C:\Users\Siddh\Desktop\ML\November\Week 2"
python .\Code\fit_mae_both_300h.py

Separate outputs: Code/experiments/mae_both_300h_v1/<timestamp>.
Old files/results preserved. No resume. This is a new experiment, not an
established improvement. No real-data fitting has been performed by assistant.

Search: leverage38, flow/liquidation30, combined62 packs, independently for
UP and DOWN and each quarterly issue. HGB absolute loss: depth2,4 leaves,
minleaf256,60iterations,lr.05,L2=10,bins63,3000 evenly spaced estimator rows.
GRU16 absolute loss: hidden16,lr.001,AdamWdecay.03,12epochs,batch512,
gradientclip1,2048 evenly spaced estimator rows. Seed48. Same train-only
preprocessing and bounded median residual as earlier GRU. Four exact causal
snapshots90/60/30/0h assembled before filtering future label validity.
All mature rows used for preprocessing, coverage and mean/median baselines.

Model blend weights .05/.1/.25/.5/.75/1 and constant anchor meanweights0/.5/1.
192 inner fits + up to16 final fits; runtime not benchmarked. No broad
Cartesian tuning. Compared with prior tests, this probes larger GRU capacity,
shallower regularized trees, flow features and additional shrinkage weights.

Four earlier three-month windows; strict300h maturity at all fit boundaries.
MAE-only selection: .1% pooled gain against BOTH mean and median MAE,
>=3/4 wins-or-ties against both and no window >2% worse against either.
Otherwise median fallback. Best validation candidate also saved, diagnostic
only when gate fails. RMSE and median absolute error are reported, not optimized
or gated. Missing input coverage<80% returns constant anchor.

Comparisons (hash/key/truth/issue checked, not used to select new models):
UP previous_reference = diagnostic candidate from mae_up_300h_v1/
20261114T163703375349Z, MAE455.734514; DOWN previous_reference = accepted
selected forecast from magnitude_models_300h_fast_v1/20261114T162602551164Z,
MAE459.144369. UP reference failed earlier validation, so is not an accepted
model. Previous forecasts are NOT automatically overwritten/promoted.

17473 real evaluation origins per target. Repeatedly examined historical data,
not independent confirmation. New results may worsen; no guaranteed gain.
Output metrics.csv includes MAE, median_absolute_error, RMSE and IC for each
forecast and period. Predictions, feature definitions/admitted columns,
validation trials, fit boundaries, models/replay, hashes and exclusions saved.

Checks:
python Code/fit_mae_both_300h.py --check-inputs
python Code/check_mae_both_300h.py
