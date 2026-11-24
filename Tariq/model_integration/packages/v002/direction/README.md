# sid_direction-15m-v002

Probability that the BTCUSDT reference price at t+16 minutes exceeds the reference price at t+1 minute. Minute-issued execution-direction classifier combining 65% logistic and 35% shallow boosting with prior-quarter ZIP calibration.

## Run the example

Download every artifact in the matching MLflow run into one folder. From that folder, using Python 3.12.5, run the following commands.

```powershell
python -m pip install -r requirements.txt
python predict.py --input example_input.json --artifact-dir . --output actual_output.json
```

Compare with expected_output.json to 1e-8 absolute tolerance. The synthetic example produces p_up = 0.507160336911. In the local handoff bundle, pass the directory containing direction.joblib as --artifact-dir.

## Input and evaluation

Supply one whole-minute BTCUSDT feature row, using the exact names in manifest.json. The standalone build_features function in predict.py specifies the upstream transforms. Supply valid BTC F4 price, flow and liquidation features for 5m, 15m and 1h in the market frame. Mask invalid values and observations published after the decision. Use BTC and ETH reference minute prices available by their minute end.

The packaged October 1, 2024 checkpoint was trained through September 2024. The metrics in evaluation.json cover seven quarterly refits; frozen_checkpoint_test gives this checkpoint?s own result. Earlier work informed feature design on the same historical period.
