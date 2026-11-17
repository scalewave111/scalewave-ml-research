120H NET MAGNITUDE: NEURAL REFINEMENT ONLY

cd "C:\Users\Siddh\Desktop\ML\November\Week 3"
python Code/fit_net_magnitude_120h_neural_refined.py

Runs MLP and GRU independently; no trees. One family: --families mlp or gru.
Two feature packs market_new/all,2000estimator-row cap.
MLP32hidden16epochsAdamWdecay.1; GRU16hidden12epochsAdamWdecay.03.
Within each family compare original L1 loss and squared-error L2 alternative.
Weights .05/.1/.25/.4/.5/.6/.75/1 toward training mean or median.
Three earlier validation windows,strict120h maturity and train-only preprocessing.
48innerfits per family,up to8outerfits each. No measured runtime.
Require .1%pooled MAE gain vs both baselines,2/3wins/ties,no>2%loss each;
also .1%pooled RMSE gain vs mean,2/3RMSEwins/ties,no>2%RMSEwindowloss.
Rank accepted by MAE; otherwise mean fallback. Diagnostic candidate separate.
No guarantee guards transfer to later outcomes.
Uses local canonical checksum receipt by default; --canonical-manifest overrides.
Previous pure and fast policies are aligned comparison controls,not tuning data.
Outputs Code/experiments/net_magnitude_120h_<family>_neural_refined_v1/<timestamp>/
and Report/net_magnitude_120h_<family>_neural_refined_v1/<timestamp>/.
Generated/syntax/static dispatch checked. No input audit,model fitting or inference.
Historical exposed adaptive evaluation; no improvement established yet.
