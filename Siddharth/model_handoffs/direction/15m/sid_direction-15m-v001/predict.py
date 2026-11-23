"""Frozen BTC 15-minute direction blend."""
import numpy as np

from replay_io import run


def predict_frame(models, x):
    probabilities = {}
    for name, state in models.items():
        frame = x[state["columns"]].clip(state["low"], state["high"], axis=1)
        p = state["estimator"].predict_proba(frame)[:, 1]
        p = np.where(frame.notna().any(axis=1), p, state["prior"])
        probabilities[name] = np.clip(p * state["probability_scale"] + state["probability_offset"],
                                      1e-6, 1 - 1e-6)
    return .75 * probabilities["boost"] + .25 * probabilities["ridge"]


if __name__ == "__main__":
    run(predict_frame, "p_up")
