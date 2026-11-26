import numpy as np
import pandas as pd
import pytest
import torch

from sak_tcn.model import ModelConfig, DirectionTCN, ResidualBlock, build_model
from sak_tcn.protocol import (RobustScaler, Windows, decisions, earlier_thresholds,
                              evaluate, maturity_split, regular_positions, signed_score)


def forecasts():
    a = pd.date_range("2023-04-01", periods=100, freq="min", tz="UTC")
    b = pd.date_range("2023-07-01", periods=4, freq="min", tz="UTC")
    index = a.append(b)
    return pd.DataFrame({
        "p_up": np.r_[np.linspace(.50, .99, 100), [.75, .25, .9, .51]],
        "reference_p_up": np.r_[np.linspace(.50, .99, 100), [.75, .25, .9, .51]],
        "predicted_magnitude_bps": 40.,
        "actual_bps": np.r_[np.ones(100), [-20., 8., 5., -2.]],
        "target_available_at": index + pd.Timedelta(minutes=16),
        "model_fold": ["2023Q2"]*100+["2023Q3"]*4,
    }, index=index)


def test_architecture_size_shape_and_trainable_gradients():
    c = ModelConfig()
    m = DirectionTCN(c)
    assert c.steps == 61
    assert sum(p.numel() for p in m.parameters()) == 5854
    assert (c.steps-1)*c.rates[-1] == 240
    x = torch.randn(2, 2, 292, 61)
    y = m(x)
    assert y.shape == (2,)
    torch.nn.functional.binary_cross_entropy_with_logits(y, torch.tensor([0., 1.])).backward()
    assert m.head.weight.grad.abs().sum() > 0
    assert m.skip.weight.grad.abs().sum() > 0


def test_nontrivial_convolutions_do_not_access_the_future():
    torch.manual_seed(7)
    c = ModelConfig(features=3, channels=4, dilations=(1, 2), dropout=0)
    block = ResidualBlock(c, dilation=2).eval()
    x = torch.randn(2, 4, 40)
    changed = x.clone()
    changed[:, :, 21:] += 1000
    with torch.no_grad():
        a, b = block(x), block(changed)
    torch.testing.assert_close(a[:, :, :21], b[:, :, :21], rtol=0, atol=0)
    assert not torch.allclose(a[:, :, 21:], b[:, :, 21:])


def test_grid_missingness_and_prehistory_never_wrap():
    index = pd.DatetimeIndex(["2023-01-01T00:00Z", "2023-01-01T00:02Z"])
    grid, positions = regular_positions(index)
    assert positions.tolist() == [0, 2]
    c = ModelConfig(features=1, channels=2, kernel=2, dilations=(1,), rates=(1,), dropout=0)
    x = np.array([[1.], [np.nan], [3.]], dtype="float32")
    scale = RobustScaler(np.array([0.]), np.array([1.]))
    windows = Windows(x, np.ones(3), [0, 2], scale, c)
    first, _ = windows[0]
    last, _ = windows[1]
    np.testing.assert_array_equal(first.numpy()[0, 1], [0, 0, 1])
    np.testing.assert_array_equal(last.numpy()[0, 1], [1, 0, 1])
    np.testing.assert_array_equal(last.numpy()[0, 0], [1, 0, 3])
    assert len(grid) == 3


def test_scaler_uses_training_rows_only():
    x = np.array([[1., np.nan], [3., np.nan], [1e9, 999.]], dtype="float32")
    a = RobustScaler.fit(x, [0, 1])
    x[2] = [-1e9, -999.]
    b = RobustScaler.fit(x, [0, 1])
    np.testing.assert_array_equal(a.center, b.center)
    np.testing.assert_array_equal(a.scale, b.scale)
    np.testing.assert_array_equal(a.center, [2, 0])
    transformed = a.transform(np.array([[np.nan, np.nan], [1e8, 3.]]))
    np.testing.assert_array_equal(transformed[0], [0, 0, 0, 0])
    assert transformed[1, 0] == 10
    with pytest.raises(ValueError, match="Infinite"):
        a.transform([[np.inf, 0]])


def test_purge_uses_label_availability_and_rejects_h4a():
    index = pd.date_range("2023-04-01", "2023-07-01", freq="min", tz="UTC")
    available = index+pd.Timedelta(minutes=16)
    issue = pd.Timestamp("2023-07-01", tz="UTC")
    training = np.flatnonzero(available < issue)
    fit, valid = maturity_split(index, available, training, issue, validation_days=30)
    start = issue-pd.Timedelta(days=30)
    assert (available[fit] < start).all()
    assert (index[valid] >= start).all()
    assert not np.intersect1d(fit, valid).size
    with pytest.raises(ValueError, match="mature"):
        maturity_split(index, available, np.arange(len(index)), issue)
    with pytest.raises(ValueError, match="pre-2025"):
        maturity_split(index, available, training, pd.Timestamp("2025-01-01", tz="UTC"))


def test_rules_use_predicted_signed_bps_and_exact_ten_boundary():
    p = np.array([.75, .25, .749, .251, .5, .9])
    mag = np.array([20, 20, 20, 20, 20, 0])
    np.testing.assert_allclose(signed_score(p, mag), [10, -10, 9.96, -9.96, 0, 0])
    side, selected = decisions(p, mag, "pm10")
    assert selected.tolist() == [True, True, False, False, False, False]
    _, raw = decisions(p, mag, "ungated")
    assert raw.tolist() == [True, True, True, True, False, True]
    _, top = decisions(p, mag, "top1", threshold=np.full(6, 9.9))
    _, both = decisions(p, mag, "both", threshold=np.full(6, 9.9))
    np.testing.assert_array_equal(both, selected & top)
    with pytest.raises(ValueError, match="Invalid"):
        signed_score([1.01], [20])


def test_top1_never_fits_on_current_quarter_or_outcomes():
    a = forecasts()
    threshold = earlier_thresholds(a, "p_up")
    assert np.isnan(threshold[:100]).all()
    np.testing.assert_allclose(threshold[100:], [39.2]*4)
    b = a.copy()
    b.iloc[100:, b.columns.get_loc("p_up")] = 1.
    b.actual_bps = -100000.
    np.testing.assert_array_equal(threshold, earlier_thresholds(b, "p_up"))


def test_scoring_keeps_losses_and_zero_months_and_counts_overlap():
    frame = forecasts()
    table, monthly, ledger = evaluate(frame, "pm10")
    q3 = ledger.loc[(ledger.model == "challenger") & (ledger.model_fold == "2023Q3")]
    assert len(q3) == 3
    assert q3.gross_bps.tolist() == [-20, -8, 5]  # Losing trades are retained.
    assert (q3.exit_timestamp_utc.iloc[:-1].to_numpy() >
            q3.entry_timestamp_utc.iloc[1:].to_numpy()).all()  # No K80 merging.
    june = monthly[(monthly.month == "2023-06") & (monthly.model == "challenger")].iloc[0]
    assert june.trades == 0 and june.forecasts == 0 and np.isnan(june.gross_bps)
    row = table[table.model == "challenger"].iloc[0]
    assert row["net_bps_assuming_4bp_round_trip"] == pytest.approx(row.gross_bps-4)
    assert ((ledger.exit_timestamp_utc-ledger.entry_timestamp_utc) == pd.Timedelta(minutes=15)).all()


def test_top1_requires_a_past_oof_fold():
    frame = forecasts().iloc[:100]
    with pytest.raises(ValueError, match="prior completed"):
        evaluate(frame, "top1")


def test_incorrect_calendar_and_target_timing_are_rejected():
    frame = forecasts()
    frame.iloc[-1, frame.columns.get_loc("model_fold")] = "2024Q1"
    with pytest.raises(ValueError, match="quarter"):
        evaluate(frame, "ungated")
    frame = forecasts()
    frame.target_available_at = frame.index
    with pytest.raises(ValueError, match="maturity"):
        evaluate(frame, "ungated")


def test_summary_control_is_invariant_to_historical_order():
    torch.manual_seed(1)
    c = ModelConfig(features=2, channels=2, kernel=2, dilations=(1, 2))
    m = build_model(c, "summary")
    torch.nn.init.normal_(m.linear.weight)
    x = torch.randn(3, 2, 4, c.steps)
    x[:, :, 2:] = 1
    b = x.clone()
    b[:, :, :, :-1] = x[:, :, :, :-1].flip(-1)
    with torch.no_grad():
        torch.testing.assert_close(m(x), m(b), rtol=1e-5, atol=1e-5)
