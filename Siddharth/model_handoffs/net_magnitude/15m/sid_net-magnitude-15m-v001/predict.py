"""Frozen BTC 15-minute absolute execution return in simple basis points."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent

INPUT_COLUMNS = (
    "BTC_abs_1m_ew15",
    "BTC_abs_1m_ew60",
    "BTC_abs_1m_ew360",
    "BTC_abs_1m_ew1440",
    "BTC_abs_5m_ew15",
    "BTC_abs_5m_ew60",
    "BTC_abs_5m_ew360",
    "BTC_abs_5m_ew1440",
    "BTC_abs_15m_ew15",
    "BTC_abs_15m_ew60",
    "BTC_abs_15m_ew360",
    "BTC_abs_15m_ew1440",
    "BTC_range_15",
    "BTC_range_60",
    "BTC_range_240",
    "ETH_abs_1m_ew15",
    "ETH_abs_1m_ew60",
    "ETH_abs_1m_ew360",
    "ETH_abs_1m_ew1440",
    "ETH_abs_5m_ew15",
    "ETH_abs_5m_ew60",
    "ETH_abs_5m_ew360",
    "ETH_abs_5m_ew1440",
    "ETH_abs_15m_ew15",
    "ETH_abs_15m_ew60",
    "ETH_abs_15m_ew360",
    "ETH_abs_15m_ew1440",
    "ETH_range_15",
    "ETH_range_60",
    "ETH_range_240",
    "log_scale",
    "weekend",
    "ny_slot",
    "ny_weekday",
    "ny_sin1",
    "ny_cos1",
    "ny_sin2",
    "ny_cos2",
    "ny_slot_ratio",
    "utc_slot",
    "utc_weekday",
    "utc_sin1",
    "utc_cos1",
    "utc_sin2",
    "utc_cos2",
    "utc_slot_ratio",
    "minute_trend_15",
    "minute_trend_60",
    "up_energy_60",
    "short_risk_ratio",
    "day_risk_ratio",
    "minute_risk_5",
    "minute_abs_5",
    "minute_trend_5",
    "up_energy_5",
    "minute_risk_15",
    "minute_abs_15",
    "up_energy_15",
    "minute_risk_60",
    "minute_abs_60",
    "minute_risk_240",
    "minute_abs_240",
    "minute_trend_240",
    "up_energy_240",
    "minute_risk_1440",
    "minute_abs_1440",
    "minute_risk_10080",
    "minute_abs_10080",
    "utc_hour",
    "near_utc_0h",
    "near_utc_8h",
    "near_utc_13h",
    "near_utc_14h",
    "near_utc_16h",
    "near_utc_20h",
    "vol_change_5",
    "abs_to_rms_5",
    "vol_change_15",
    "abs_to_rms_15",
    "vol_change_60",
    "abs_to_rms_60",
    "vol_change_240",
    "abs_to_rms_240",
    "same_slot_risk_14d",
    "slot_relative_risk",
    "f4z__flow__flow_activity_percentile__4h",
    "f4z__flow__flow_activity_percentile__1d",
    "f4z__flow__flow_pressure_percentile__4h",
    "f4z__flow__flow_pressure_percentile__1d",
    "f4z__flow__flow_cvd_slope_percentile__4h",
    "f4z__flow__flow_cvd_slope_percentile__1d",
    "f4z__price__price_displacement_percentile__4h",
    "f4z__price__price_displacement_percentile__1d",
    "f4z__price__price_excursion_range_percentile__4h",
    "f4z__price__price_excursion_range_percentile__1d",
    "f4z__volatility__realized_variance_level_percentile__4h",
    "f4z__volatility__realized_variance_level_percentile__1d",
    "risk_2h",
    "trend_over_risk_2h",
    "upside_variance_fraction_2h",
    "risk_8h",
    "trend_over_risk_8h",
    "upside_variance_fraction_8h",
    "risk_16h",
    "trend_over_risk_16h",
    "upside_variance_fraction_16h",
    "risk_48h",
    "trend_over_risk_48h",
    "upside_variance_fraction_48h",
    "risk_168h",
    "trend_over_risk_168h",
    "upside_variance_fraction_168h",
    "risk_672h",
    "trend_over_risk_672h",
    "upside_variance_fraction_672h",
    "imbalance_4h",
    "imbalance_change_4h",
    "flow_price_4h",
    "imbalance_1d",
    "imbalance_change_1d",
    "flow_price_1d",
    "funding_normalized_rate_per_8h",
    "spread_funding_normalized_rate_per_8h",
    "mark_basis_bps",
    "spread_mark_basis_bps",
    "oi_quote_oi_value_usdt_equiv",
    "spread_oi_quote_oi_value_usdt_equiv",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_models(directory, manifest):
    models = {}
    for item in manifest["artifacts"]:
        path = directory / item["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Artifact SHA-256 mismatch: {path.name}")
        if path.suffix == ".joblib":
            models[path.stem] = joblib.load(path)
    return models


def predict_frame(models, x):
    state = models["magnitude"]
    scale = np.expm1(x["log_scale"].to_numpy())
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("log_scale must encode a finite positive adjusted scale")
    prediction = {}
    for name, pipeline in state["models"].items():
        raw = pipeline.predict(x[state["features"][name]])
        prediction[name] = np.maximum(.03, raw) * scale
    w = state["tree_weight"]
    return state["multiplier"] * (w * prediction["boost"] + (1 - w) * prediction["mlp_deep"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(HERE / "manifest.json")
    row = read_json(args.input)
    if row["symbol"] != "BTCUSDT" or row["horizon"] != "15m":
        raise ValueError("Expected BTCUSDT and 15m")
    when = pd.Timestamp(row["forecast_time_utc"])
    if when.tzinfo is None or pd.isna(when):
        raise ValueError("Supply a timezone-aware forecast timestamp")
    when = when.tz_convert("UTC")
    if when != when.floor("30min"):
        raise ValueError("Expected a timestamp on the UTC half-hour grid")
    values = row["features"]
    if not isinstance(values, dict) or set(values) != set(INPUT_COLUMNS):
        raise ValueError("Feature names must match INPUT_COLUMNS exactly")
    for name, value in values.items():
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError(f"Invalid feature {name}: use a finite number or null")
    x = pd.DataFrame([values], columns=INPUT_COLUMNS, dtype=float)
    with threadpool_limits(limits=1):
        predictions = predict_frame(load_models(args.artifact_dir, manifest), x)
    value = float(np.asarray(predictions)[0])
    if not math.isfinite(value):
        raise ValueError("Prediction is nonfinite")
    output = {"model_version": manifest["model_version"], "symbol": "BTCUSDT", "horizon": "15m",
              "forecast_time_utc": when.isoformat(), "net_magnitude_bps": value}
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
