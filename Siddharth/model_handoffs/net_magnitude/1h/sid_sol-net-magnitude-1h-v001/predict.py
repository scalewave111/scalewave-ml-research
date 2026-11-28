"""Run one frozen SOL forecast from a whole-minute feature row."""
import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np


def predict(payload, manifest, artifact_dir):
    for key in ("symbol", "horizon"):
        if payload[key] != manifest[key]:
            raise ValueError(f"Unexpected {key}")
    stamp = datetime.fromisoformat(payload["forecast_time_utc"].replace("Z", "+00:00"))
    if stamp.utcoffset() is None or stamp.utcoffset().total_seconds() or stamp.second or stamp.microsecond:
        raise ValueError("Expected a whole UTC minute")
    features = payload["features"]
    names = manifest["input_features"]
    if set(features) != set(names):
        raise ValueError("Feature keys differ from the manifest")
    x = np.array([features[n] for n in names], dtype="float32")
    if np.isinf(x).any():
        raise ValueError("Use null for missing observations")
    item, = manifest["artifacts"]
    path = Path(artifact_dir) / item["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    model = joblib.load(path)
    direction = manifest["target"] == "direction"
    if model["type"] == "gru":
        import torch
        from torch import nn
        net = nn.GRU(12, 18, batch_first=True)
        linear = nn.Linear(18, 3)
        for prefix, layer in (("gru.", net), ("output.", linear)):
            layer.load_state_dict({k[len(prefix):]: v for k, v in model["state"].items() if k.startswith(prefix)})
        z = np.nan_to_num(((x.reshape(6,12) - model["center"]) / model["width"]).clip(-5,5))
        with torch.no_grad():
            _, hidden = net(torch.from_numpy(z[None]))
            out = linear(hidden[-1])[0]
            value = float(torch.sigmoid(out[0]) if direction else nn.functional.softplus(out[1]))
    else:
        value = float(model["tree"].predict(x.reshape(1,-1), num_threads=1)[0])
        if direction and "platt" in model:
            a, b = model["platt"]
            p = np.clip(value, 1e-6, 1-1e-6)
            value = float(1 / (1 + np.exp(-(a*np.log(p/(1-p)) + b))))
        if not direction:
            value = max(value, model["floor"])
    if not direction:
        scale = float(payload["scale_bps"])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Expected positive finite scale_bps")
        value *= scale
    return dict(model_version=manifest["model_version"], symbol=payload["symbol"], horizon=payload["horizon"],
                forecast_time_utc=payload["forecast_time_utc"], **{manifest["output_key"]: value})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("input", "artifact-dir", "output"):
        cli.add_argument("--"+key, type=Path, required=True)
    args = cli.parse_args()
    manifest = json.loads(Path(__file__).with_name("manifest.json").read_text())
    result = predict(json.loads(args.input.read_text()), manifest, args.artifact_dir)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
