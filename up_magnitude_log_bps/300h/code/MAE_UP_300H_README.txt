UP300h MAE-only refinement (2026-11-14)

Run:
cd "C:\Users\Siddh\Desktop\ML\November\Week 2"
python .\Code\fit_mae_up_300h.py

UP only; DOWN results unchanged. Separate timestamp outputs:
Code/experiments/mae_up_300h_v1/<timestamp>

Compared with the joint fast search: selection now optimizes MAE, with no RMSE
acceptance requirement. Both metrics are still reported. Two feature packs:
leverage38 and combined62; HGB/MLP16/GRU8 with absolute-error loss only.
96 inner fits and up to8 final fits. Neural12 epochs/2048 estimator rows;
HGB40 iterations/3000 estimator rows. Other architectures, preprocessing,
missingness rules, blend weights and causal sequences inherit the300h code.
All training rows used for preprocessing and mean/median baselines.
Four earlier three-month validation windows, strict300h maturity, quarterly
fits, all17473 UP evaluation origins. Acceptance: .1% pooled MAE improvement
against BOTH mean/median, >=3/4 wins-or-ties against both, no window >2% worse
against either. Else median. Best candidate retained separately.

Previous fast selected and candidate predictions are hash/key/truth checked
and comparison-only. The prior UPcandidate457.499983 is diagnostic; this run
must demonstrate an improvement, and does not automatically promote it.
Historical evaluation is repeatedly exposed; gains are not independent
confirmation. No guarantee of gain and no wall-clock benchmark. No resume.

Outputs include predictions, period/pooled metrics, actual feature lists,
validation trials, serialized models/replay, exclusions and provenance.
dual_baseline_checks.csv still reports all four MAE/RMSE comparisons; joint
success is NOT required for this MAE-only selection.

Checks:
python Code/fit_mae_up_300h.py --check-inputs
python Code/check_mae_up_300h.py
