"""Time, feature, selection and accounting contracts. No model fitting."""
from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

MINUTE_NS = 60_000_000_000
H4_END = pd.Timestamp("2025-01-01", tz="UTC")
RULES = ("ungated", "top1", "pm10", "both")


def utc_index(index):
    index = pd.DatetimeIndex(index)
    if index.tz is None:
        raise ValueError("Timezone-aware UTC timestamps required.")
    index = index.tz_convert("UTC").as_unit("ns")
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Timestamps must be unique and sorted.")
    if not index.equals(index.floor("min")):
        raise ValueError("Exact minute boundaries required.")
    return index


def regular_positions(index):
    index = utc_index(index)
    if len(index) == 0:
        raise ValueError("Empty decision calendar.")
    grid = pd.date_range(index[0], index[-1], freq="min").as_unit("ns")
    return grid, grid.get_indexer(index)


@dataclass
class RobustScaler:
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x, train_rows):
        a = np.asarray(x[np.asarray(train_rows, dtype=np.int64)], dtype=np.float32)
        if not len(a) or np.isinf(a).any():
            raise ValueError("Empty training rows or infinite features.")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            lo, center, hi = np.nanquantile(a, [0.25, 0.5, 0.75], axis=0)
        center = np.where(np.isfinite(center), center, 0)
        scale = np.where(np.isfinite(hi-lo) & ((hi-lo) > 1e-8), hi-lo, 1)
        return cls(center.astype("float32"), scale.astype("float32"))

    def transform(self, a):
        a = np.asarray(a, dtype=np.float32)
        if np.isinf(a).any():
            raise ValueError("Infinite features rejected.")
        mask = np.isfinite(a)
        values = np.where(mask, np.clip((a-self.center)/self.scale, -10, 10), 0)
        return np.concatenate([values, mask.astype("float32")], axis=-1).astype("float32")


class Windows(Dataset):
    """Windows use a complete minute grid: missing minutes never compress time."""
    def __init__(self, x, y, rows, scaler, config):
        self.x, self.y = x, y
        self.rows = np.asarray(rows, dtype=np.int64)
        if len(self.rows) and ((self.rows < 0).any() or (self.rows >= len(x)).any()):
            raise ValueError("Origin outside grid.")
        self.scaler, self.config = scaler, config
        self.offsets = np.array(config.rates)[:, None] * (
            np.arange(config.steps)[None, :] - config.steps + 1
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        positions = row + self.offsets
        values = np.array(self.x[np.maximum(positions, 0)], copy=True)
        values[positions < 0] = np.nan
        transformed = self.scaler.transform(values).transpose(0, 2, 1).copy()
        return torch.from_numpy(transformed), torch.tensor(float(self.y[row] > 0), dtype=torch.float32)


def maturity_split(index, available, training, issue, validation_days=30):
    index = utc_index(index)
    available = pd.DatetimeIndex(pd.to_datetime(available, utc=True))
    issue = pd.Timestamp(issue)
    if issue.tz is None or issue >= H4_END:
        raise ValueError("Training is restricted to pre-2025 H4 issues.")
    rows = np.asarray(training, dtype=np.int64)
    if len(available) != len(index) or not len(rows) or (rows < 0).any() or (rows >= len(index)).any():
        raise ValueError("Invalid training rows or maturity calendar.")
    if not (available[rows] < issue).all() or not (index[rows] < issue).all():
        raise ValueError("Training outcome is not mature before issue.")
    start = issue-pd.Timedelta(days=validation_days)
    inner = rows[(available[rows] < start) & (index[rows] < start)]
    valid = rows[index[rows] >= start]
    if not len(inner) or not len(valid):
        raise ValueError("Empty chronological fit/validation split.")
    return inner, valid


def signed_score(p, magnitude):
    p, magnitude = np.asarray(p, float), np.asarray(magnitude, float)
    if (not np.isfinite(p).all() or not np.isfinite(magnitude).all()
            or ((p < 0) | (p > 1)).any() or (magnitude < 0).any()):
        raise ValueError("Invalid probabilities or magnitudes.")
    return (2*p-1)*magnitude


def earlier_thresholds(frame, probability_column):
    """Quarterly threshold from strictly earlier OOF forecast rows only."""
    score = np.abs(signed_score(frame[probability_column], frame.predicted_magnitude_bps))
    times = utc_index(frame.index)
    labels = pd.DatetimeIndex(pd.to_datetime(frame.target_available_at, utc=True))
    result = np.full(len(frame), np.nan)
    for fold in frame.model_fold.unique():
        selected = frame.model_fold.eq(fold).to_numpy()
        start = pd.Period(str(fold), freq="Q").start_time.tz_localize("UTC")
        earlier = (times < start) & (labels < start)
        if earlier.any():
            result[selected] = np.quantile(score[earlier], .99, method="higher")
    return result


def decisions(p, magnitude, rule, threshold=None):
    if rule not in RULES:
        raise ValueError(rule)
    score = signed_score(p, magnitude)
    # Ungated direction is independent of magnitude, including zero magnitude.
    side = np.sign(2*np.asarray(p)-1) if rule == "ungated" else np.sign(score)
    selected = side != 0
    if rule in ("top1", "both"):
        if threshold is None:
            raise ValueError("Historical strength threshold required.")
        threshold = np.asarray(threshold, float)
        selected &= np.isfinite(threshold) & (np.abs(score) >= threshold)
    if rule in ("pm10", "both"):
        selected &= (score >= 10) | (score <= -10)
    return side, selected


def rank_ic(predicted, actual):
    a, b = pd.Series(np.asarray(predicted)).rank(), pd.Series(np.asarray(actual)).rank()
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return None
    return float(a.corr(b))


def classification(p, actual):
    p = np.clip(np.asarray(p, float), 1e-12, 1-1e-12)
    y = np.asarray(actual) > 0
    return dict(forecasts=len(p),
                log_loss=float(-np.where(y, np.log(p), np.log1p(-p)).mean()),
                brier=float(np.mean((p-y)**2)),
                direction_accuracy=float(np.mean((p > .5) == y)),
                neutral_probability_count=int((p == .5).sum()),
                probability_rank_ic=rank_ic(p, actual))


def trade_stats(actual, side, selected):
    gross = np.asarray(actual)[selected]*side[selected]
    n = len(gross)
    out = dict(trades=n, longs=int((side[selected]>0).sum()),
               shorts=int((side[selected]<0).sum()),
               gross_bps=float(gross.mean()) if n else None,
               sum_entry_bps=float(gross.sum()),
               trade_win_rate=float((gross>0).mean()) if n else None)
    for cost in (0, 2, 4, 8):
        out[f"net_bps_assuming_{cost}bp_round_trip"] = float(gross.mean()-cost) if n else None
    return out


def evaluate(frame, rule):
    """Both models see identical rows. Returns tables; never selects by outcomes."""
    frame = frame.copy()
    frame.index = utc_index(frame.index)
    required = ["p_up", "reference_p_up", "predicted_magnitude_bps", "actual_bps"]
    if not np.isfinite(frame[required].to_numpy(float)).all():
        raise ValueError("Nonfinite forecasts/targets; resolve coverage before scoring.")
    actual_fold = frame.index.tz_localize(None).to_period("Q").astype(str)
    if not np.array_equal(frame.model_fold.astype(str).to_numpy(), actual_fold):
        raise ValueError("Forecast timestamp does not match its model quarter.")
    if pd.to_datetime(frame.target_available_at, utc=True).isna().any():
        raise ValueError("Missing target maturity.")
    if (pd.to_datetime(frame.target_available_at, utc=True) <
            frame.index + pd.Timedelta(minutes=16)).any():
        raise ValueError("Target maturity precedes the declared exit.")
    tables, month_rows, records = [], [], []
    for name, column in (("SAK", "reference_p_up"), ("challenger", "p_up")):
        threshold = earlier_thresholds(frame, column) if rule in ("top1", "both") else None
        side, selected = decisions(frame[column], frame.predicted_magnitude_bps, rule, threshold)
        scored = np.isfinite(threshold) if threshold is not None else np.ones(len(frame), bool)
        if not scored.any():
            raise ValueError("Top-1% needs at least one prior completed OOF fold.")
        record = {"model": name, "rule": rule, "warmup_forecasts": int((~scored).sum()),
                  **classification(frame.loc[scored, column], frame.loc[scored, "actual_bps"]),
                  **trade_stats(frame.actual_bps.to_numpy(), side, selected)}
        record["signed_score_rank_ic"] = rank_ic(
            signed_score(frame.loc[scored, column], frame.loc[scored, "predicted_magnitude_bps"]),
            frame.loc[scored, "actual_bps"])
        record["selection_rate"] = record["trades"]/int(scored.sum())
        record["gross_bps_per_scored_forecast"] = record["sum_entry_bps"]/int(scored.sum())
        tables.append(record)
        months = frame.index.strftime("%Y-%m")
        for month in pd.period_range(frame.index.min().tz_localize(None).to_period("M"),
                                     frame.index.max().tz_localize(None).to_period("M"), freq="M"):
            mask = months == str(month)
            mselected = selected & mask
            month_rows.append({"model": name, "rule": rule, "month": str(month),
                               "forecasts": int((scored & mask).sum()),
                               "warmup_forecasts": int((~scored & mask).sum()),
                               **trade_stats(frame.actual_bps.to_numpy(), side, mselected)})
        ledger = frame.loc[selected, ["actual_bps", "model_fold"]].copy()
        ledger["model"] = name
        ledger["side"] = side[selected]
        ledger["score_bps"] = signed_score(frame[column], frame.predicted_magnitude_bps)[selected]
        ledger["entry_timestamp_utc"] = ledger.index + pd.Timedelta(minutes=1)
        ledger["exit_timestamp_utc"] = ledger.index + pd.Timedelta(minutes=16)
        ledger["gross_bps"] = ledger.actual_bps*ledger.side
        if threshold is not None:
            ledger["strength_threshold_bps"] = threshold[selected]
        records.append(ledger)
    return pd.DataFrame(tables), pd.DataFrame(month_rows), pd.concat(records)


def paired_daily_uncertainty(frame, rule, block_days=14, replicates=1000, seed=20261127):
    """Calendar P&L difference at equal entry notional, retaining zero-trade days."""
    daily = []
    thresholds = {}
    valid = np.ones(len(frame), bool)
    for col in ("reference_p_up", "p_up"):
        thresholds[col] = earlier_thresholds(frame, col) if rule in ("top1", "both") else None
        if thresholds[col] is not None:
            valid &= np.isfinite(thresholds[col])
    if not valid.any():
        raise ValueError("No common scored dates.")
    times = frame.index[valid]
    days = pd.date_range(times.min().floor("D"), times.max().floor("D"), freq="D")
    for col in ("reference_p_up", "p_up"):
        side, select = decisions(frame[col], frame.predicted_magnitude_bps, rule, thresholds[col])
        select &= valid
        pnl = np.where(select, side*frame.actual_bps.to_numpy()-4, 0)
        s = pd.Series(pnl[valid], index=times).groupby(times.floor("D")).sum().reindex(days, fill_value=0)
        daily.append(s.to_numpy())
    delta = daily[1]-daily[0]
    if len(delta) < block_days:
        return {"status": "INSUFFICIENT_DAYS", "days": len(delta)}
    rng = np.random.default_rng(seed)
    sampled = []
    for _ in range(replicates):
        starts = rng.integers(0, len(delta)-block_days+1, size=int(np.ceil(len(delta)/block_days)))
        positions = (starts[:, None]+np.arange(block_days)).ravel()[:len(delta)]
        sampled.append(float(delta[positions].mean()))
    return {"status": "DESCRIPTIVE_EXPOSED_HISTORY", "block_days": block_days,
            "assumed_round_trip_cost_bps": 4, "replicates": replicates,
            "delta_mean_daily_entry_notional_bps": float(delta.mean()),
            "ci95": np.quantile(sampled, [.025, .975]).tolist(),
            "not_account_return": True, "not_corrected_for_research_selection": True}
