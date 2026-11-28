# sid_sol-net-magnitude-5m-v001

Forecast of 10000 * abs(VWAP[t+6] / VWAP[t+1] - 1) for SOLUSDT. Wide LightGBM mean of the clipped absolute normalized return, restored to bps.

## Run the example

Use Python 3.12.5. Place the MLflow artifacts together. From the model directory run:

```powershell
python -m pip install -r requirements.txt
python predict.py --input example_input.json --artifact-dir . --output actual_output.json
```

Compare the prediction with expected_output.json to 1e-8 absolute tolerance. For this local release pass the model_files/sid_sol-net-magnitude-5m-v001 directory to --artifact-dir.

## Input and evaluation

Supply the engineered features listed in manifest.json as of the forecast minute. JSON null represents missing observations. Use the same scale as the engineered scale feature, SOL_scale clipped below at 0.2 bps. The release distributes the October 2024 checkpoint. evaluation.json reports native forecast errors on eligible labeled minutes and includes the historical, validation and test periods. Each quarter of historical forecasts uses its own scheduled checkpoint. Validation and test have been reviewed previously.

The included fit source is a verbatim copy of Report/SOL_5M_WIDE_ROUTER_V02/wide.py. It uses the original Week 4 research workspace and training data at their original paths. Use that workspace for refitting. The prediction CLI uses the distributed checkpoint independently.
