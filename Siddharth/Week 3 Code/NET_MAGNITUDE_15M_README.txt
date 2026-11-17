15-MINUTE CANONICAL NET MAGNITUDE - FOUR INDEPENDENT MODELS

cd "C:\Users\Siddh\Desktop\ML\November\Week 3"
python Code/fit_net_magnitude_15m_separate.py

Runs HGB, ExtraTrees, MLP16, GRU16 separately and sequentially. Each family
covers every period with no other-model or median fallback in its primary
forecast. Earlier validation chooses a feature pack within that family.

Individual run:
python Code/fit_net_magnitude_15m.py --pure-model hgb
Replace hgb with extra/mlp/gru as needed. All-model launcher stops if a run fails.
Optional read-only input audit:
python Code/fit_net_magnitude_15m.py --check-inputs

TARGET: 10000*abs(ln(P(t+15min)/P(t))), read directly from checksum-verified
canonical horizon_minutes=15 Y. This is not UP minus DOWN. Valid finite
nonnegative labels only, endpoint-within-excursion/direction checks, explicit
source-binding crossing exclusions. Outcomes are never used as X predictors.

Five packs:
- market: all522 original masked F4Z value candidates, not forced48h winners.
- prior_price:51 causal price features from prior15m UP/DOWN feature builder.
- new_price:57 new NET15 features.
- all: union630 candidates before per-fit training-quality filtering.
- market_new:579 candidates before per-fit quality filtering.

NEW FEATURES:9 measurements at15/30/60/240/1440/10080minute lookbacks:
RV, signed net, absolute net, sum absolute returns, trend efficiency, upside
variance share, jump share, RMS return, positive-return fraction. Three
normalized volatility ratios15/60,30/240,60/1440. Uses canonical5m returns
strictly before origin; every rolling window requires complete observations.
All57 exact names/formulas saved. Prior51 definitions remain in the recorded
Week2 improve_excursion_features_15m.py source.

TIMING: labels mature strictly before fitting cutoff with15minute horizon.
Three earlier two-month validation windows offset9/6/3months; Jan/Apr/Jul/Oct
2024 outer issues. Original half-hour X grid retained: this is15minute target
prediction every30minutes, not a newly fabricated15minute X timeline.
GRU four exact-time snapshots90/60/30/0minutes; sequence timeline formed before
label filtering. Missing snapshots use trained imputation and missing flags.

Same estimator budgets as48h: cap6000 evenly spaced mature rows; HGB60 rounds,
ExtraTrees48trees, MLP16/40epochs, GRU16/24epochs. Full mature preprocessing,
scaling and baselines.60inner and up to8outer fits per family. Broader market
candidate set may take more time than48h. CPU; fixed seed48; no resume.
Median/mean are comparison baselines. Guarded policy is separately recorded;
pure-family predictions do not imply validation acceptance. MAE is the
feature-pack ranking objective; RMSE/bias also reported.

Code output: Code/experiments/net_magnitude_15m_<family>_only_v1/<timestamp>/
Reports: Report/net_magnitude_15m_<family>_only/<timestamp>/
Includes predictions, model artifacts/replay, metrics, exact feature lists,
validation trials, target population audit, formulas, provenance and versions.
Shared model engine: Code/fit_net_magnitude_48h.py (keep this file available).
Week2/Week1 source files used read-only. Requires existing sklearn/pandas/numpy/
duckdb/joblib/threadpoolctl installation; GRU requires torch.

STATUS: Generated for user execution.15m loader, models, feature checks and
complete pipeline have NOT been executed/tested. No new performance claim.
Historical evaluation has already been exposed; no independent confirmation.
