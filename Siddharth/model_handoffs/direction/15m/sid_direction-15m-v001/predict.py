"""Frozen BTC 15-minute direction blend."""
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
    "carry_agreement",
    "f2_basis_change",
    "f2_excursion_balance",
    "f2_funding_change",
    "f2_liq_concentration",
    "f2_liq_peak_fraction",
    "f2_liq_share",
    "f2_oi_change",
    "f2_oi_price_alignment",
    "f2_range_spike",
    "f2_taker_balance",
    "f2_taker_share",
    "f2_variance_turnover_gap",
    "f4__basis__basis_level__4h",
    "f4__flow__flow_activity__1d",
    "f4__flow__flow_cvd_slope__1d",
    "f4__flow__flow_pressure__1d",
    "f4__flow__flow_pressure__4h",
    "f4__funding__funding_level__4h",
    "f4__liquidation__liq_activity_severity__4h",
    "f4__liquidation__liq_side_pressure__4h",
    "f4__liquidation__venue_fragmentation__4h",
    "f4__oi__leverage_unwind__4h",
    "f4__oi__oi_range_position__4h",
    "f4__price__price_displacement__1d",
    "f4__price__price_displacement__4h",
    "f4__price__price_excursion_range__1d",
    "f4__price__price_move_retention__4h",
    "f4__volatility__realized_variance_level__1d",
    "f4__volatility__realized_variance_level__4h",
    "f4z__flow__flow_activity_percentile__4h",
    "f4z__flow__flow_cvd_slope_percentile__4h",
    "f4z__flow__flow_pressure_percentile__4h",
    "f4z__price__price_displacement_percentile__4h",
    "f4z__price__price_excursion_range_percentile__4h",
    "f4z__volatility__realized_variance_level_percentile__4h",
    "f5__flow_price__flow_price_alignment__1d_leaf_2",
    "f5__flow_price__flow_price_alignment__1d_leaf_3",
    "f5__flow_price__flow_price_alignment__1d_leaf_5",
    "f5__flow_price__flow_price_alignment__1d_leaf_6",
    "f5__flow_price__flow_price_alignment__4h_leaf_2",
    "f5__flow_price__flow_price_alignment__4h_leaf_3",
    "f5__flow_price__flow_price_alignment__4h_leaf_5",
    "f5__flow_price__flow_price_alignment__4h_leaf_6",
    "f5__flow_price__price_response_per_flow__1d_leaf_2",
    "f5__flow_price__price_response_per_flow__1d_leaf_3",
    "f5__flow_price__price_response_per_flow__1d_leaf_5",
    "f5__flow_price__price_response_per_flow__1d_leaf_6",
    "f5__liq_oi__oi_response_per_liq__4h_leaf_2",
    "f5__liq_oi__oi_response_per_liq__4h_leaf_3",
    "f5__liq_oi__oi_response_per_liq__4h_leaf_5",
    "f5__liq_oi__oi_response_per_liq__4h_leaf_6",
    "f5__liq_price__liq_price_alignment__4h_leaf_2",
    "f5__liq_price__liq_price_alignment__4h_leaf_3",
    "f5__liq_price__liq_price_alignment__4h_leaf_5",
    "f5__liq_price__liq_price_alignment__4h_leaf_6",
    "f5__price_vol__price_displacement_to_vol__4h_leaf_2",
    "f5__price_vol__price_displacement_to_vol__4h_leaf_3",
    "f5__price_vol__price_displacement_to_vol__4h_leaf_5",
    "f5__price_vol__price_displacement_to_vol__4h_leaf_6",
    "f5z__flow_price__flow_price_alignment_percentile__4h_leaf_2",
    "f5z__flow_price__flow_price_alignment_percentile__4h_leaf_3",
    "f5z__flow_price__flow_price_alignment_percentile__4h_leaf_5",
    "f5z__flow_price__flow_price_alignment_percentile__4h_leaf_6",
    "f5z__liq_oi__oi_response_per_liq_percentile__4h_leaf_2",
    "f5z__liq_oi__oi_response_per_liq_percentile__4h_leaf_3",
    "f5z__liq_oi__oi_response_per_liq_percentile__4h_leaf_5",
    "f5z__liq_oi__oi_response_per_liq_percentile__4h_leaf_6",
    "f5z__liq_price__liq_price_alignment_percentile__4h_leaf_2",
    "f5z__liq_price__liq_price_alignment_percentile__4h_leaf_3",
    "f5z__liq_price__liq_price_alignment_percentile__4h_leaf_5",
    "f5z__liq_price__liq_price_alignment_percentile__4h_leaf_6",
    "f5z__price_vol__price_displacement_to_vol_percentile__4h_leaf_2",
    "f5z__price_vol__price_displacement_to_vol_percentile__4h_leaf_3",
    "f5z__price_vol__price_displacement_to_vol_percentile__4h_leaf_5",
    "f5z__price_vol__price_displacement_to_vol_percentile__4h_leaf_6",
    "flow_price_agreement",
    "forced_flow_unwind",
    "move_per_range",
    "move_per_volatility",
    "positive_days_7d",
    "pressure_change_72h",
    "share_acceleration",
    "volatility_share",
    "taker_imbalance_15m",
    "imbalance_shift_two_15m_buckets",
    "log_vwap_shift_two_15m_buckets",
    "flow_price_alignment_15m",
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
    probabilities = {}
    for name, state in models.items():
        frame = x[state["columns"]].clip(state["low"], state["high"], axis=1)
        p = state["estimator"].predict_proba(frame)[:, 1]
        p = np.where(frame.notna().any(axis=1), p, state["prior"])
        probabilities[name] = np.clip(p * state["probability_scale"] + state["probability_offset"],
                                      1e-6, 1 - 1e-6)
    return .75 * probabilities["boost"] + .25 * probabilities["ridge"]


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
              "forecast_time_utc": when.isoformat(), "p_up": value}
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
