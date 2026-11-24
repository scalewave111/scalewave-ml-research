# sid_net-magnitude-15m-v002

Conditional median of 10000 * abs(P[t+16 minutes] / P[t+1 minute] - 1), measured in simple bps. Minute-issued volatility-scaled absolute-error LightGBM using BTC/ETH activity and time-of-day features.

## Run the example

Download every artifact in the matching MLflow run into one folder. From that folder, using Python 3.12.5, run the following commands.

```powershell
python -m pip install -r requirements.txt
python predict.py --input example_input.json --artifact-dir . --output actual_output.json
```

Compare with expected_output.json to 1e-8 absolute tolerance. The synthetic example produces net_magnitude_bps = 7.983212754702. In the local handoff bundle, pass the directory containing net-magnitude.joblib as --artifact-dir.

## Input and evaluation

Supply one whole-minute BTCUSDT feature row, using the exact names in manifest.json. The standalone build_features function in predict.py specifies the upstream transforms. Supply BTC and ETH minute prices as a UTC minute-indexed frame with BTC and ETH columns. Mask invalid or late prices. Send the returned scale for the forecast minute as scale_bps, plus the manifest feature columns. The feature builder uses trailing prices and prior-day slot history.

The packaged October 1, 2024 checkpoint was trained through September 2024. The metrics in evaluation.json cover seven quarterly refits; frozen_checkpoint_test gives this checkpoint?s own result. Earlier work informed feature design on the same historical period.
