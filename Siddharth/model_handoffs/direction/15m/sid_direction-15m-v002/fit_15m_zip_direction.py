"""Minute-issued BTC direction on the ZIP's seven quarterly execution-return folds."""
from datetime import datetime, timezone
from io import BytesIO
import json
import sys
from zipfile import ZipFile

import duckdb
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from fit_15m_median_magnitude import H4, ROOT, ZIP, load_data

sys.path.insert(0, str(ROOT.parent / "Week 3/Code"))
from test_16h_absorption_signal import StoredZipView


FAMILIES = {"price": ["price_displacement", "price_move_retention"],
            "flow": ["flow_pressure", "flow_cvd_slope"],
            "liquidation": ["liq_side_pressure", "liq_activity_severity"]}
WINDOWS = ["5m", "15m", "1h"]


def market_fields():
    frames = []
    with ZipFile(H4) as archive, H4.open("rb") as raw, duckdb.connect() as db:
        db.execute("SET TimeZone='UTC'")
        shards = sorted((n for n in archive.namelist() if "/data/x_shards/" in n and
                         n.endswith(".zip") and n[-14:-4] <= "2024-12-31"))
        for i, shard in enumerate(shards):
            parts = []
            with ZipFile(StoredZipView(raw, archive.getinfo(shard))) as day:
                for family, signals in FAMILIES.items():
                    name, = [n for n in day.namelist() if n.startswith("FORGE/F4/") and
                             f"/family={family}/" in n and "/symbol=BTCUSDT/" in n and n.endswith(".parquet")]
                    data = db.read_parquet(BytesIO(day.read(name)))
                    fields = [f"{s}__{w}" for s in signals for w in WINDOWS]
                    select = [f'CASE WHEN available_after <= decision_time AND {f}__valid '
                              f'THEN {f}__value::FLOAT END AS {f}' for f in fields]
                    part = data.select("decision_time," + ",".join(select)).df()
                    parts.append(part.set_index("decision_time", verify_integrity=True))
            frames.append(pd.concat(parts, axis=1))
            if (i+1) % 250 == 0:
                print("Read market features from", i+1, "daily shards", flush=True)
    return pd.concat(frames, verify_integrity=True).sort_index()


def build_features(prices, market, idx):
    # load_data enforces source validity and publication by each price minute-end.
    market = market.reindex(idx).copy()
    px = np.log(prices)*10000
    for asset in ["BTC", "ETH"]:
        path = px[asset]
        vol = path.diff().abs().ewm(halflife=60, min_periods=15).mean().clip(lower=.05)
        for n in (1, 5, 15, 60):
            market[f"{asset}_return_{n}m_scaled"] = (path.diff(n)/(vol*np.sqrt(n))).reindex(idx).clip(-8,8)
    market["btc_eth_relative_15m"] = market.BTC_return_15m_scaled - market.ETH_return_15m_scaled
    market["flow_change"] = market.flow_pressure__5m - market.flow_pressure__1h
    market["liquidation_change"] = market.liq_side_pressure__5m - market.liq_side_pressure__1h
    market["weekend"] = (idx.dayofweek >= 5).astype(float)
    return market.replace([np.inf,-np.inf],np.nan).astype("float32")


def linear_inputs(x, center, scale):
    return np.tanh(((x-center)/scale).fillna(0).to_numpy(dtype="float64"))


def fit(x, y):
    sample = x.iloc[::31]
    corr = sample.corr(method="spearman").abs()
    cols, removed = [], {}
    for name in x:
        paired = corr.loc[name, cols].dropna()
        if sample[name].nunique() < 2:
            removed[name] = "constant or missing on the training screen"
        elif len(paired) and paired.max() > .985:
            removed[name] = "redundant with " + paired.idxmax()
        else:
            cols.append(name)
    data = x[cols]
    center = data.median()
    scale = (sample[cols].quantile(.85)-sample[cols].quantile(.15))/2
    scale = scale.where(scale > 1.e-9, sample[cols].std()).fillna(1).clip(lower=1.e-9)
    linear = LogisticRegression(C=.05, max_iter=120)
    linear.fit(linear_inputs(data,center,scale), y)
    tree = LGBMClassifier(n_estimators=120, learning_rate=.035, max_depth=3, num_leaves=8,
                          min_child_samples=500, reg_lambda=8, n_jobs=3, verbosity=-1,
                          random_state=72, deterministic=True, force_col_wise=True)
    tree.fit(data, y)
    return dict(columns=cols, removed=removed, center=center, scale=scale, linear=linear, tree=tree)


def raw_probability(model, x):
    data = x[model["columns"]]
    linear = model["linear"].predict_proba(linear_inputs(data,model["center"],model["scale"]))[:,1]
    tree = model["tree"].predict_proba(data)[:,1]
    return .65*linear + .35*tree


def loss(y, probability):
    return -np.log(np.where(y,probability,1-probability).clip(1e-12,1)).mean()


def shrinkage(history):
    if history.empty:
        return 1.
    base = logit(history.training_up_rate.to_numpy())
    difference = logit(history.raw_p_up.to_numpy()) - base
    objective = lambda weight: loss(history.actual_up.to_numpy(), expit(base+weight*difference))
    estimate = minimize_scalar(objective, bounds=(-2.,8.), method="bounded").x
    return float(np.clip((len(history)*estimate+4400*.5)/(len(history)+4400),0,1))


def main():
    prices, labels, reference, _ = load_data()
    ref = pd.DataFrame(reference)
    first = pd.Timestamp(ref.issue.iloc[0],tz="UTC")
    end = pd.Timestamp(ref.issue.iloc[-1],tz="UTC")+pd.DateOffset(months=3)
    labels = labels.loc[labels.index < end]
    idx = labels.index
    score_idx = idx[idx>=first]
    price_ret = 10000*(prices.BTC.reindex(score_idx+pd.Timedelta(minutes=16)).to_numpy() /
                       prices.BTC.reindex(score_idx+pd.Timedelta(minutes=1)).to_numpy()-1)
    np.testing.assert_allclose(price_ret, labels.signed_return_simple_bps.loc[score_idx],atol=1e-8,rtol=0)
    y = labels.signed_return_simple_bps > 0
    market = market_fields()
    x = build_features(prices,market,idx)
    assert len(score_idx)==923007 and x.index.is_unique
    out = ROOT/"Report"/datetime.now(timezone.utc).strftime("zip_direction_15m_%Y%m%dT%H%M%SZ")
    out.mkdir()
    data = pd.DataFrame(index=score_idx)
    data["actual_up"] = y.loc[score_idx]
    data["available"] = labels.target_available_at.loc[score_idx]
    data["actual_bps"] = labels.signed_return_simple_bps.loc[score_idx]
    fold_rows = []
    for quarter in reference:
        issue = pd.Timestamp(quarter["issue"],tz="UTC")
        train = (idx >= issue-pd.DateOffset(months=12)) & (idx<issue) & (labels.target_available_at < issue)
        test = idx[(idx>=issue)&(idx<issue+pd.DateOffset(months=3))]
        assert train.sum()==quarter["train_rows"] and len(test)==quarter["test_rows"]
        rate = y[train].mean()
        np.testing.assert_allclose(rate,quarter["dir_p_up_fitted"],atol=1e-6)
        history = data.loc[data.index < issue]
        weight = shrinkage(history)
        mature_weight = shrinkage(history.loc[history.available < issue])
        model = fit(x.loc[train],y[train])
        raw = raw_probability(model,x.loc[test])
        def calibrate(w):
            return expit(logit(rate)+w*(logit(raw)-logit(rate)))
        p = calibrate(weight)
        data.loc[test,"training_up_rate"] = rate
        data.loc[test,"raw_p_up"] = raw
        data.loc[test,"p_up"] = p
        data.loc[test,"mature_calibration_p_up"] = calibrate(mature_weight)
        model.update(training_up_rate=rate, calibration=weight, mature_calibration=mature_weight)
        path = out/f"{issue.date()}.joblib"
        joblib.dump(model,path)
        np.testing.assert_allclose(raw[:30],raw_probability(joblib.load(path),x.loc[test[:30]]))
        fold_rows.append(dict(issue=issue, train_rows=int(train.sum()), test_rows=len(test),
                              first_train=idx[train].min(), last_train=idx[train].max(),
                              latest_train_label=labels.target_available_at[train].max(),
                              test_last=test[-1],validation_rows=0,features=len(model["columns"]),
                              removed=json.dumps(model["removed"]),calibration=weight,
                              mature_calibration=mature_weight,accuracy=((p>.5)==y[test]).mean(),
                              correct=int(((p>.5)==y[test]).sum()),log_loss=loss(y[test],p)))
        print(str(issue.date()),"accuracy", round(fold_rows[-1]["accuracy"]*100,3),flush=True)
    old = pd.read_csv(ROOT/"Report/15m_signed_scores_price_check.csv",index_col=0,parse_dates=True)
    data["previous_p_up"] = old.p_up.reindex(data.index)
    half = data.previous_p_up.notna()
    assert half.sum()==30767 and data.drop(columns="previous_p_up").notna().all().all()
    np.testing.assert_array_equal(data.actual_up[half],old.loc[data.index[half],"price_return_bps"]>0)
    scores = []
    for cohort,part in [("ZIP minutes",data),("Shared half-hours",data.loc[half])]:
        for model,col in [("challenger","p_up"),("mature calibration","mature_calibration_p_up"),
                          ("raw","raw_p_up"),("prior UP frequency","training_up_rate"),("previous","previous_p_up")]:
            if part[col].notna().all():
                hit = (part[col]>.5)==part.actual_up
                scores.append(dict(cohort=cohort,model=model,n=len(part),correct=int(hit.sum()),
                                   accuracy_pct=100*hit.mean(),log_loss=loss(part.actual_up,part[col])))
    with ZipFile(ZIP) as z:
        zoo = pd.read_csv(z.open("BTC_15M_WEEK2/results_reference/ZOO_EXEC.csv"))
    for row in zoo.query("target=='dir'").itertuples():
        scores.append(dict(cohort="ZIP minutes",model=row.model,n=row.rows,correct=round(row.accuracy*row.rows),
                           accuracy_pct=100*row.accuracy,log_loss=row.primary))
    data.rename_axis("forecast_time_utc").to_parquet(out/"predictions.parquet")
    pd.DataFrame(fold_rows).to_csv(out/"folds.csv",index=False)
    pd.DataFrame(scores).to_csv(out/"scores.csv",index=False)
    print(out, pd.DataFrame(scores).to_string(index=False), sep="\n", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=3):
        main()
