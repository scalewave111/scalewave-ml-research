# sid_sol-net-magnitude-1h-v001

Forecast of 10000 * abs(VWAP[t+61] / VWAP[t+1] - 1) for SOLUSDT. All-feature LightGBM mean absolute normalized return, restored to bps.

## Run the example

Use Python 3.12.5. Place the MLflow artifacts together. From the model directory run:

```powershell
python -m pip install -r requirements.txt
python predict.py --input example_input.json --artifact-dir . --output actual_output.json
```

Compare the prediction with expected_output.json to 1e-8 absolute tolerance. For this local release pass the model_files/sid_sol-net-magnitude-1h-v001 directory to --artifact-dir.

## Input and evaluation

Supply the engineered features listed in manifest.json as of the forecast minute. JSON null represents missing observations. Use the causal EWM(span=180, adjust=False, min_periods=30) of absolute one-minute SOL simple returns in bps, floored and filled at 0.01, times sqrt(60). The release distributes the October 2024 checkpoint. evaluation.json reports native forecast errors on eligible labeled minutes and includes the historical, validation and test periods. Each quarter of historical forecasts uses its own scheduled checkpoint. Validation and test have been reviewed previously.

The included fit source is a verbatim copy of Code/sol60_train.py. It uses the original Week 4 research workspace and training data at their original paths. Use that workspace for refitting. The prediction CLI uses the distributed checkpoint independently.
