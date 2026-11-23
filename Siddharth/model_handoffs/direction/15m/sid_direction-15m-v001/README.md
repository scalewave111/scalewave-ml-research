# sid_direction-15m-v001

Owner: Sid. BTCUSDT, 15 minutes. Frozen quarter starting January 1, 2025 UTC. Tested with Python 3.12.5.

Probability of a positive native 15-minute BTCUSDT endpoint return from t to t+15. The reported execution-target comparison evaluates the sign from t+1 to t+16.

## Run the packaged example

Open PowerShell in this folder after extracting the entire bundle. Install the pinned requirements into your Python 3.12.5 environment, then run:

```powershell
python -m pip install -r requirements.txt
python predict.py --input example_input.json --output actual_output.json --artifact-dir "../../../../model_files/sid_direction-15m-v001"
Get-Content actual_output.json
```

Compare the output with expected_output.json. The synthetic example produces `p_up = 0.492407167581`. Match numeric outputs with absolute and relative tolerance of 1e-8. After uploading, download the manifest-listed files to a local artifact folder and pass that folder to --artifact-dir.

## Inputs

Provide one UTC half-hour feature row. Use the feature keys in feature_schema.json; it also gives the exact component order. example_input.json illustrates the complete interface with artificial feature values. FEATURES.md includes feature calculations and data guards. The caller supplies engineered features at the stated origin.

Use only origin-available market features and trailing short-window BTC inputs. Apply the validity masks and release checks described in FEATURES.md.

Ensembling uses 75% shallow histogram boosting and 25% ridge logistic probabilities. The columns are aligned by name to the per-estimator order before clipping. Null encodes missing data. Boosting handles missing values natively and the ridge pipeline uses its fitted imputer. When an entire component row is missing, its saved prior is used before applying the saved probability transform.

## Evaluation and provenance

evaluation.json includes the matched walk-forward metrics, quarterly breakdown and a separate check of the final checkpoint. manifest.json records the training cutoff, data-catalog and artifact hashes, and the available source provenance. The original training commit has not been established. SHA-256 checks run before loading fitted files. Load trusted artifacts only.

- Quarterly walk-forward evaluation reuses a historical calendar examined in model selection. Evaluation ends 2025-01-23 23:30 UTC. Current market performance is unverified.
- Primary comparison covers successive historical fitted checkpoints. The included frozen January 2025 checkpoint is scored separately below.
- Integration requires the upstream causal feature pipeline and data availability checks. No execution, live feed or quarterly retraining is included.
- Native training sign and one-minute-delayed execution sign can differ.
