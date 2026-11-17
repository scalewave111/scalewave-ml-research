120H CONTROLLED DIAGNOSIS: ORIGINAL FOUR FAMILIES

cd "C:\Users\Siddh\Desktop\ML\November\Week 3"
python Code/fit_net_magnitude_120h_controlled.py

Optional one family: --families mlp (or hgb,extra,gru).
--check-inputs checks features/labels without fitting.

Sixteen FIXED features: momentum,range position,efficiency,upvariance over
60/120/240h;168hdrawdown,336hjump concentration;720hmean/CV of completed past
120h endpoint magnitudes. No newfeature search or model-family mixing.

Four paired cells per family:
small_direct,small_residual,larger_direct,larger_residual.
Small2000estimator rows,HGB40rounds,Extra24trees,MLP16epochs,GRU12epochs.
Larger6000rows,HGB80rounds,Extra64trees,MLP48epochs,GRU24epochs.
Budgets jointly vary row count and iteration count; they do not isolate those
individual factors. All preprocessing and baselines use full mature history.
Architecture fixedMLP32hidden,GRU16hidden; trees depth/regularization fixed.
All models use L1 loss to hold objective fixed for controlled comparison.

Direct target median-centered/stdscaled; residual target mean-centered/stdscaled.
Residual neural output layer starts atzero,so initial prediction is trainingmean.
For trees,constant target shifts are translation-equivariant; centering alone is
not a new signal or fundamentally different model. Paired tree cells are controls.
Neural residual comparison also changes output initialization,explicitly.
Predictions nonnegative; no target clipping/logtransform or future mean used.

Same four earlier3monthwindows15/12/9/6months;strict120hmaturity,GRU36/24/12/0h.
Validation selects cell and modelweight .05/.1/.25/.5/.75/1 toward trainingmean.
MAE primary: .1%pooledgain vs mean,at least2/4windowwins/ties,no>2%windowloss;
otherwise meanfallback. RMSE reported,not gate. This MAE-only/2of4gate is a changed
selection policy relative to earlier joint MAE/RMSE/3of4feature searches.
Cell controls allow the change in policy to be separated from actual model scores.
Choices frozen before outer scoring; all four outer cell predictions saved for
paired analysis,not used to choose that issue's cell. Candidate diagnostic separate.
Historical evaluation exposed,no fresh confirmation or guaranteed gain.

training_validation_diagnostics.csv: comparable1000row mature training sample
and validation MAE/RMSE/bias and levels. Training sample can overlap estimator
rows,so it measures apparent fit,not independent training performance.
outer_cell_diagnostics.csv: allrows and actual>trainingP90 errors for all4cells.
Tail grouping retrospective diagnostic only,never a prediction input.
metrics.csv contains eachcell,selected,candidate,mean/median; predictions includes
allcell forecasts. Hashes,models,serialization replay and completion saved by run.

256innerfits plus64outerfits across all4families;largerbudgets may take longer.
No runtimepromise/resume. New net_magnitude_120h_<family>_controlled_v1 timestamped
Code/experiments and Report roots. Existing files/results preserved.
Uses existing localcanonicalmanifest,read-onlysource loaders and feature modules.
Status: syntax/imports,synthetic zero-residual neural forward,MAEguard,
selection/fallback and budgets checked. No model fitting or actualinputaudit.
