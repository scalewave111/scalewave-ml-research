# sid_sol-direction-5m-v001

Probability that SOLUSDT minute VWAP at t+6 exceeds VWAP at t+1. Wide LightGBM binary head.

## Run the example

Use Python 3.12.5. Place the MLflow artifacts together. From the model directory run:

```powershell
python -m pip install -r requirements.txt
python predict.py --input example_input.json --artifact-dir . --output actual_output.json
```

Compare the prediction with expected_output.json to 1e-8 absolute tolerance. For this local release pass the model_files/sid_sol-direction-5m-v001 directory to --artifact-dir.

## Input and evaluation

Supply the engineered features listed in manifest.json as of the forecast minute. JSON null represents missing observations.  The release distributes the October 2024 checkpoint. evaluation.json reports native forecast errors on eligible labeled minutes and includes the historical, validation and test periods. Each quarter of historical forecasts uses its own scheduled checkpoint. Validation and test have been reviewed previously.

The included fit source is a verbatim copy of Report/SOL_5M_WIDE_ROUTER_V02/wide.py. It uses the original Week 4 research workspace and training data at their original paths. Use that workspace for refitting. The prediction CLI uses the distributed checkpoint independently.
