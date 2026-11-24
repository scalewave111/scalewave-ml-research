"""Predict one BTCUSDT 15-minute execution outcome from engineered features."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit

def build_features(prices, market, idx):
    # load_data enforces source validity and publication by each price minute-end.
    market = market.reindex(idx).copy()
    px = np.log(prices)*10000
    for asset in ["BTC", "ETH"]:
        path = px[asset]
        vol = path.diff().abs().ewm(halflife=60, min_periods=15).mean().clip(lower=.05)
        for n in (1, 5, 15, 60):
            market[f"{asset}_return_{n}m_scaled"] = (path.diff(n)/(vol*np.sqrt(n))).reindex(idx).clip(-8,8)
    market["btc_eth_relative_15m"] = market.BTC_return_15m_scaled - market.ETH_return_15m_scaled
    market["flow_change"] = market.flow_pressure__5m - market.flow_pressure__1h
    market["liquidation_change"] = market.liq_side_pressure__5m - market.liq_side_pressure__1h
    market["weekend"] = (idx.dayofweek >= 5).astype(float)
    return market.replace([np.inf,-np.inf],np.nan).astype("float32")

def linear_inputs(x, center, scale):
    return np.tanh(((x-center)/scale).fillna(0).to_numpy(dtype="float64"))

def raw_probability(model, x):
    data = x[model["columns"]]
    linear = model["linear"].predict_proba(linear_inputs(data,model["center"],model["scale"]))[:,1]
    tree = model["tree"].predict_proba(data)[:,1]
    return .65*linear + .35*tree
def forecast(model, x, row):
    raw = raw_probability(model, x)[0]
    base = logit(model["training_up_rate"])
    return expit(base + model["calibration"] * (logit(raw) - base))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    row = json.loads(args.input.read_text(encoding="utf-8-sig"))
    timestamp = pd.Timestamp(row["forecast_time_utc"])
    if timestamp.tzinfo is None or timestamp != timestamp.floor("min"):
        raise ValueError("A timezone-aware whole-minute timestamp is required")
    if row["symbol"] != "BTCUSDT" or row["horizon"] != "15m":
        raise ValueError("Expected BTCUSDT and 15m")
    fields = manifest["input_features"]
    if set(row["features"]) != set(fields):
        raise ValueError("Feature names must match the manifest")
    values = [row["features"][name] for name in fields]
    if any(v is not None and (type(v) not in (float, int) or not np.isfinite(v)) for v in values):
        raise ValueError("Expected finite numbers or null in features")
    x = pd.DataFrame([values], columns=fields, dtype="float32")
    item, = manifest["artifacts"]
    path = args.artifact_dir/item["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        raise ValueError("Checkpoint hash differs from manifest")
    value = forecast(joblib.load(path), x, row)
    if not np.isfinite(value):
        raise ValueError("Non-finite forecast")
    result = {"model_version":manifest["model_version"], "symbol":"BTCUSDT", "horizon":"15m",
              "forecast_time_utc":timestamp.tz_convert("UTC").isoformat(),manifest["output_key"]:float(value)}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")

if __name__ == "__main__":
    main()
