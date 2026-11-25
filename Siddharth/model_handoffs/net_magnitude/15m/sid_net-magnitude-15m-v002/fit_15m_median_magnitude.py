"""Walk-forward median BTC magnitude, scored on the Week 2 minute calendar."""
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import duckdb
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


ROOT = Path(__file__).resolve().parents[1]
H4 = ROOT.parent / "Week 3/Datasets/HANDOFF_4_ML_RESEARCH_V1_MINUTE.zip"
ZIP = ROOT / "Code/BTC_15M_WEEK2.zip"
FIRST = pd.Timestamp("2023-04-01", tz="UTC")
END = pd.Timestamp("2025-01-01", tz="UTC")
MINUTE = pd.Timedelta(minutes=1)


def load_data():
    prefix = "HANDOFF_4_ML_RESEARCH_V1"
    with duckdb.connect() as db, ZipFile(H4) as archive, ZipFile(ZIP) as pack:
        db.execute("SET TimeZone='UTC'")
        keys = db.read_parquet(BytesIO(pack.read("BTC_15M_WEEK2/inputs/DECISION_ROWS.parquet")))
        eligible = pd.DatetimeIndex(keys.select("decision_time").df().decision_time)
        ref = json.loads(pack.read("BTC_15M_WEEK2/results_reference/BASELINE_EXEC.json"))
        zoo = pd.read_csv(pack.open("BTC_15M_WEEK2/results_reference/ZOO_EXEC.csv"))
        target = float(zoo.set_index("model").loc["CAT_MAE", "primary"])
        # Dataset.y marks the first minute-end of a new binding as the transition.
        transitions = json.loads(archive.read(f"{prefix}/catalog/BINDING_TRANSITIONS.json"))
        cuts = [pd.Timestamp(t["date"], tz="UTC") + MINUTE for t in transitions[1:]]
        prices = {}
        for asset in ["BTC", "ETH"]:
            path = f"{prefix}/data/prices/PRICE_1m_{asset}USDT.parquet"
            rows = db.read_parquet(BytesIO(archive.read(path))).select("""
                window_end, CASE WHEN source_structurally_valid AND f066__is_valid
                AND f066__value > 0 AND available_after <= window_end
                AND decision_time <= window_end THEN f066__value END AS price
            """).df().set_index("window_end", verify_integrity=True)
            prices[asset] = rows.price
        blocks = []
        for path in archive.namelist():
            if "/data/y_actual/horizon_minutes=15/symbol=BTCUSDT/" in path and path.endswith(".parquet"):
                rows = db.read_parquet(BytesIO(archive.read(path))).filter("""
                    label_status='VALID' AND target_end=decision_time+interval '15 minutes'
                    AND target_available_at >= target_end
                """).select("decision_time,target_end,target_available_at,signed_return_simple_bps").df()
                blocks.append(rows)
    labels = pd.concat(blocks).set_index("decision_time", verify_integrity=True).sort_index()
    for cut in cuts:
        labels = labels.loc[~((labels.index < cut) & (labels.target_end >= cut))]
    labels.index -= MINUTE  # Execution return starts one minute after the forecast.
    idx = eligible.intersection(labels.index).sort_values()
    labels = labels.loc[idx].dropna()
    assert labels.index.is_unique and np.isfinite(labels.signed_return_simple_bps).all()
    for quarter in ref["per_quarter"]:
        issue = pd.Timestamp(quarter["issue"], tz="UTC")
        part = labels.loc[(labels.index >= issue) & (labels.index < issue + pd.DateOffset(months=3))]
        assert len(part) == quarter["test_rows"], (issue, len(part), quarter["test_rows"])
        np.testing.assert_allclose(part.signed_return_simple_bps.mean(), quarter["test_mean_bps"], atol=1e-5)
    return pd.DataFrame(prices).sort_index().asfreq("min"), labels, ref["per_quarter"], target


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


def fit_model(x, y, scale):
    # Weighting the normalized target by its scale gives absolute error in bps.
    model = LGBMRegressor(objective="regression_l1", n_estimators=300, learning_rate=.05,
                          max_depth=4, num_leaves=16, min_child_samples=180, reg_lambda=4,
                          n_jobs=3, verbosity=-1, random_state=72, deterministic=True,
                          force_col_wise=True)
    model.fit(x, y/scale, sample_weight=scale/scale.mean())
    return model


def predict(model, x, scale):
    return np.maximum(0, model.predict(x[model.feature_name_]) * scale)


def prune(x):
    """Remove near-duplicates and constants using training features only."""
    rank_correlation = x.iloc[::31].corr(method="spearman").abs()
    kept, removed = [], {}
    for name in x.columns:
        if x[name].nunique(dropna=True) <= 1:
            removed[name] = "constant"
            continue
        duplicates = rank_correlation.loc[name, kept].dropna()
        if len(duplicates) and duplicates.max() > .975:
            removed[name] = "correlated with " + duplicates.idxmax()
        else:
            kept.append(name)
    return kept, removed


def main():
    prices, labels, reference, zip_mae = load_data()
    print("ZIP quarterly counts and return means verified", flush=True)
    x, scale, groups = build_features(prices)
    x, scale = x.reindex(labels.index), scale.reindex(labels.index)
    y = labels.signed_return_simple_bps.abs()
    available = labels.target_available_at
    idx = y.index
    incumbent = pd.read_csv(ROOT/"Report/15m_signed_scores_price_check.csv",
                            index_col="forecast_time_utc", parse_dates=True)
    out = ROOT/"Report"/datetime.now(timezone.utc).strftime("median_magnitude_15m_%Y%m%dT%H%M%SZ")
    out.mkdir()
    folds, selections, issued = [], {}, []
    for quarter in reference:
        issue = pd.Timestamp(quarter["issue"], tz="UTC")
        tr = (idx >= issue-pd.DateOffset(months=12)) & (idx < issue) & (available < issue)
        te = (idx >= issue) & (idx < issue+pd.DateOffset(months=3))
        assert tr.sum() == quarter["train_rows"] and te.sum() == quarter["test_rows"]
        assert available[tr].max() < idx[te].min()
        baseline = (y[te] - y[tr].median()).abs().mean()
        np.testing.assert_allclose(baseline, quarter["mag_mae"], atol=1e-5)
        cols, removed = prune(x.loc[tr])
        model = fit_model(x.loc[tr, cols], y[tr], scale[tr])
        unused = [n for n, splits in zip(cols, model.feature_importances_) if not splits]
        if unused:
            removed.update({n: "zero training splits" for n in unused})
            cols = [n for n in cols if n not in unused]
            model = fit_model(x.loc[tr, cols], y[tr], scale[tr])
        fitted = predict(model, x.loc[te], scale[te])
        assert np.isfinite(fitted).all()
        issued.append(pd.DataFrame(dict(actual_bps=y[te], forecast_bps=fitted), index=idx[te]))
        folds.append(dict(issue=issue, features=len(cols), validation_rows=0,
                          training_first=idx[tr].min(), training_last=idx[tr].max(),
                          latest_training_label=available[tr].max(), training_rows=tr.sum(),
                          test_first=idx[te].min(), test_last=idx[te].max(), test_rows=te.sum(),
                          historical_baseline_mae=baseline,
                          mae=np.abs(fitted-y[te]).mean()))
        key = str(issue.date())
        selections[key] = {"retained": cols, "removed": removed}
        checkpoint = out/f"{key}.joblib"
        joblib.dump(model, checkpoint)
        np.testing.assert_allclose(fitted, predict(joblib.load(checkpoint), x.loc[te], scale[te]))
        print(key, len(cols), "features;", int(te.sum()), "test rows", flush=True)
    df = pd.concat(issued, verify_integrity=True)
    df.index.name = "forecast_time_utc"
    df["previous_bps"] = incumbent.net_magnitude_bps.reindex(df.index)
    shared = df.previous_bps.notna()
    np.testing.assert_allclose(df.loc[shared, "actual_bps"], incumbent.loc[df.index[shared],
                               "price_return_bps"].abs(), atol=1e-5)
    primary = df.index < END
    assert primary.all() and len(df) == 923007 and shared.sum() == 30767
    scores = []
    for name, mask in [("ZIP seven quarters, every minute",primary),
                       ("ZIP seven quarters, shared half-hours",primary & shared)]:
        part = df.loc[mask]
        delta = part.forecast_bps - part.actual_bps
        old = part.previous_bps - part.actual_bps
        scores.append(dict(cohort=name, n=len(part), mae=delta.abs().mean(), rmse=np.sqrt((delta**2).mean()),
                           previous_mae=old.abs().mean() if old.notna().all() else np.nan,
                           previous_rmse=np.sqrt((old**2).mean()) if old.notna().all() else np.nan,
                           zip_catboost_mae=zip_mae if name.endswith("every minute") else np.nan))
    df.to_parquet(out/"predictions.parquet")
    pd.DataFrame(folds).to_csv(out/"folds.csv", index=False)
    pd.DataFrame(scores).to_csv(out/"scores.csv", index=False)
    (out/"feature_sets.json").write_text(json.dumps({"groups":groups, "by_quarter":selections}, indent=2))
    print(out, pd.DataFrame(scores).to_string(index=False), sep="\n")


if __name__ == "__main__":
    main()
