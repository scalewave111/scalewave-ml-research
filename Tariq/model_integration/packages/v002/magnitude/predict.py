"""Predict one BTCUSDT 15-minute execution outcome from engineered features."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit

def build_features(prices):
    """All price-dependent fields use strictly earlier published minute prices."""
    log_price = np.log(prices).shift(1) * 10000
    features = {}
    groups = {}

    def add(group, name, values):
        features[name] = np.asarray(values, dtype="float32")
        groups.setdefault(group, []).append(name)

    for asset in ["BTC", "ETH"]:
        path = log_price[asset]
        group = "btc" if asset == "BTC" else "cross_asset"
        for step, half_lives in [(1, (8, 30, 180, 1440)), (5, (30, 180)), (15, (180, 1440))]:
            movement = path.diff(step).abs() * np.sqrt(15 / step)
            for half_life in half_lives:
                level = movement.ewm(halflife=half_life, min_periods=6).mean()
                add(group, f"{asset}_move_{step}m_decay_{half_life}m", np.log1p(level))
        for minutes in [15, 60, 240]:
            window = path.rolling(minutes, min_periods=max(5, int(minutes*.8)))
            add(group, f"{asset}_range_{minutes}m", np.log1p(window.max() - window.min()))

    def level(key):
        return pd.Series(np.expm1(features[key]), index=prices.index)

    fast = level("BTC_move_1m_decay_30m")
    slow = level("BTC_move_15m_decay_1440m")
    scale = (.40*fast + .35*level("BTC_move_5m_decay_180m") + .25*slow).clip(lower=1).fillna(12)
    add("btc", "log_scale", np.log1p(scale))
    add("shape", "btc_fast_over_slow", np.log((1+fast)/(1+slow)))
    trend = log_price.BTC.diff(15).abs()
    travel = log_price.BTC.diff().abs().rolling(15, min_periods=12).sum()
    add("shape", "btc_trend_fraction_15m", trend/(.1+travel))
    add("cross_asset", "eth_to_btc_activity", np.log((1+level("ETH_move_1m_decay_30m"))/(1+fast)))

    idx = prices.index
    ny = idx.tz_convert("America/New_York")
    add("clock", "ny_hour", ny.hour + ny.minute / 60)
    add("clock", "utc_hour", idx.hour + idx.minute / 60)
    add("clock", "utc_weekday", idx.dayofweek)
    add("clock", "minute_in_hour", idx.minute)
    add("clock", "minute_in_quarter_hour", idx.minute % 15)
    # Past days' realized activity in the same upcoming 15-minute clock interval.
    # At t this group statistic ends at t minus 1 day plus 15 minutes.
    activity = log_price.BTC.diff().abs().rolling(15, min_periods=12).mean()*np.sqrt(15)
    normalized = (activity / scale.shift(15)).clip(0, 8).shift(1440-15)
    weekday = (idx - pd.Timedelta(days=1)).dayofweek < 5
    slot = idx.hour*60 + idx.minute
    # Separate the historical weekday and weekend observations before forward filling.
    for is_weekday in [True, False]:
        history = normalized.where(weekday == is_weekday).groupby(slot).transform(
            lambda s: s.ewm(halflife=12, ignore_na=True, min_periods=2).mean())
        if is_weekday:
            season = history
        else:
            season = season.where(idx.dayofweek < 5, history)
    add("clock", "historical_slot_activity_ratio", season)
    return pd.DataFrame(features, index=idx), scale, groups


def forecast(model, x, row):
    scale = row["scale_bps"]
    if type(scale) not in (float, int) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale_bps must be positive and finite")
    return max(0.0, model.predict(x)[0] * scale)


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
