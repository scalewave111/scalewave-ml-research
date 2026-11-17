SCALEWAVE BTCUSDT / 48H MAGNITUDE HANDOFF

START HERE
Extract the ZIP into its own folder. Open PowerShell in that folder.
Use Python 3.10 and a virtual environment:
  python -m venv .venv
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt
  .\.venv\Scripts\python.exe model.py --verify
Expected: PASS, 18481 rows, MAE 180.798518098, Spearman 0.078526513.
Verification fits no models and writes no prediction files.

PREDICT / IMPORT
  .\.venv\Scripts\python.exe model.py --input replay_inputs.parquet --output predictions.parquet
Or import model.predict(features_dataframe). Input requires the seven exact
columns in features.csv and a unique timezone-aware DatetimeIndex of origins.
The code selects the snapshot by timestamp, restores saved preprocessing,
computes the liquidation anchor and averages three 128-tree forests, then blends.
NaN/infinite values use saved training medians and missing flags. Missing columns
are rejected. Do not substitute raw exchange fields for the engineered features.
This package consumes precomputed project features; it is not a raw-feed feature builder.

SCOPE / LIMITATIONS
These are four historical snapshots covering 2024-01-01 through
2025-01-22 00:00 UTC (exclusive endpoint 00:30). Other origins are rejected.
There is no fitted 2026 live model here. Fresh-data confirmation remains pending.
The output is median-oriented absolute endpoint movement in log bps, not a signed
conditional mean, direction forecast, excursion, ordinary-bps return or trade signal.
MAE is average prediction error, not the forecast itself.

SOURCE CODE
model.py is the self-contained inference/replay implementation.
research_source contains the original forest training, pair training, preprocessing,
split and verification source for inspection. These original scripts refer to the
full repository layout and are NOT standalone retraining commands. Full original
training data and research configurations are deliberately not duplicated here.
The saved artifacts reproduce predictions without retraining or original datasets.
The config retains original repository paths for provenance; portable inference
reads models/ and does not use those paths. Historical fit/verification evidence
describes its original run, not the availability of every original research file.

ARCHITECTURE / TRAINING
Pair: QuantileRegressor(quantile=0.5, alpha=0, solver='highs') using the first two
features. Forest: absolute_error, bootstrap=True, max_features=1.0, 128 trees,
three equally averaged seeds 48/17/91. Clip component forecasts at zero.
Final = (1-weight)*pair + weight*forest average.
Jan/Apr/Jul/Oct 2024 depth: 5/5/5/3; min leaf: 64/64/128/128;
bootstrap draws: 16384/16384/4096/4096; forest weights: .5/.5/.25/0.
Expanding history with quarterly refits; strict label_end < prediction cutoff;
preprocessing learned from training only. Earlier three-month inner validation
selects blend weight from 0,.1,.25,.5; ties prefer lower weight. Model settings
and features were chosen through repeated development experiments, not fresh testing.

FILES / INTEGRATION
models/: all forests, pair coefficients, preprocessing and blend settings.
replay_inputs.parquet: only seven features on the 18481 evaluation origins.
expected.parquet: targets and verified forecasts. This is development data.
features.csv: exact feature order/names. evidence/: compact original evidence.
manifest.json: SHA256 checksums verified before model loading by --verify.
Only load joblib files from a trusted sender; checksums check integrity, not identity.
The ZIP includes a trader-friendly PDF. No dataset or package is uploaded by the builder.
For forecast integration use prediction as overall_magnitude_log_bps with horizon
2880 minutes, BTCUSDT, UTC origin, and median-oriented statistic metadata.
Keep it separate from UP/DOWN conditional mean returns in ordinary bps.
