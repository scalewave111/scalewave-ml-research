"""Frozen BTC 15-minute absolute execution return in simple basis points."""
import numpy as np

from replay_io import run


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


if __name__ == "__main__":
    run(predict_frame, "net_magnitude_bps")
