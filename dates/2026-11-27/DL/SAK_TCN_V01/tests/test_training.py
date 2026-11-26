import json
import numpy as np
import pandas as pd
import pytest
import torch

from sak_tcn.model import ModelConfig
from sak_tcn.train import digest, fit_fold, replay_fold, score


def test_synthetic_fit_serialization_and_replay(tmp_path):
    torch.set_num_threads(1)
    index = pd.date_range("2023-05-15", "2023-07-01T01:00Z", freq="min", tz="UTC")
    available = index+pd.Timedelta(minutes=16)
    rng = np.random.default_rng(42)
    x = rng.normal(size=(len(index), 3)).astype("float32")
    x[::31, 1] = np.nan
    y = np.where(x[:, 0]+rng.normal(size=len(index)) > 0, 12., -12.)
    issue = pd.Timestamp("2023-07-01", tz="UTC")
    tr = np.flatnonzero(available < issue)[::240]
    te = np.flatnonzero(index >= issue)
    c = ModelConfig(features=3, channels=4, kernel=2, dilations=(1, 2), rates=(1, 4))
    p, record = fit_fold(x, y, index, available, tr, te, issue, c, "tcn",
                         tmp_path/"checkpoint", 42, "cpu", epochs=2, batch=32)
    assert len(p) == len(te) and np.isfinite(p).all()
    assert ((p > 0) & (p < 1)).all()
    assert record["best_epoch"] in (1, 2)
    assert record["weights_sha256"] == digest(tmp_path/"checkpoint/weights.pt")
    state = torch.load(tmp_path/"checkpoint/weights.pt", weights_only=True)
    assert torch.count_nonzero(state["head.weight"]) > 0  # A real optimizer step ran.
    cache = tmp_path/"cache"
    (cache/"folds").mkdir(parents=True)
    np.save(cache/"features.npy", x)
    np.save(cache/"target.npy", y)
    np.save(cache/"folds/2023Q3_test.npy", te)
    recovered = replay_fold(cache, tmp_path/"checkpoint", "2023Q3", batch=32)
    np.testing.assert_allclose(p, recovered, atol=1e-7, rtol=0)
    with pytest.raises(ValueError, match="quarter"):
        replay_fold(cache, tmp_path/"checkpoint", "2023Q4")
    # End-to-end ungated output from an actual synthetic fit.
    frame = pd.DataFrame({
        "p_up": p, "reference_p_up": .55, "predicted_magnitude_bps": 20.,
        "actual_bps": y[te], "target_available_at": available[te], "model_fold": "2023Q3",
    }, index=index[te])
    run = tmp_path/"run"
    run.mkdir()
    frame.to_parquet(run/"predictions.parquet")
    score(run, "ungated")
    comparison = pd.read_csv(run/"evaluation_ungated/comparison.csv")
    assert comparison.forecasts.tolist() == [len(te), len(te)]
    assert (run/"evaluation_ungated/score_diagnostics.csv").is_file()
    assert not (run/"evaluation_pm10").exists()
    # Checkpoint edits cannot silently change inference.
    with (tmp_path/"checkpoint/weights.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        replay_fold(cache, tmp_path/"checkpoint", "2023Q3")
