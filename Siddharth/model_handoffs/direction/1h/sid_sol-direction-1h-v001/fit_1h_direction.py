"""SOL60AF V01: fit and score tree and neural direction / magnitude heads for the SOLUSDT 1-hour horizon.

Per quarterly issue c: training rows are every 5th UTC minute in [c-12m, c-30d) with 60-minute labels available
before c-30d; [c-30d, c) (labels available before c) is used for early stopping and Platt calibration.
Heads (direction = P(UP) of the t+1 -> t+61 SOL VWAP return; magnitude = |return| / scale, scale = SOL 1-min |r|
EWM(180) x sqrt(60), restored to bps at inference):
  T_lgb  LightGBM on all 3,304 inputs (binary; L2 magnitude)
  T_xgb  XGBoost hist on the top-400 T_lgb inputs (binary:logistic; reg:squarederror magnitude)
  N_mlp  torch MLP 256-128-64 shared trunk, two outputs (BCE + Huber), top-256 inputs, robust z-scores
  N_gru  torch GRU(64) over 32 steps spaced 5 minutes (t-155 ... t) of the top-48 inputs, two outputs
usage: python sol60_train.py fit | score_hist | score_4a
"""
import json
import os
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sol60_common import (DATA, FOLDS, FROZEN, H, INPUTS, OUT, WARM, approved_columns, day_slices, extra_names,
                          feature_names, price_block, prices)

TH = int(os.environ.get("SOL60_THREADS", "4"))
torch.set_num_threads(TH)
SEED = 20261127
LGBP = dict(learning_rate=.03, num_leaves=31, max_depth=7, min_data_in_leaf=400, lambda_l2=30.0, feature_fraction=.15,
            bagging_fraction=.7, bagging_freq=1, max_bin=63, seed=SEED, deterministic=True, force_col_wise=True,
            num_threads=TH, verbosity=-1)
XGBP = dict(tree_method="hist", max_depth=5, eta=.03, subsample=.7, colsample_bytree=.5, min_child_weight=50,
            reg_lambda=30, max_bin=64, nthread=TH, seed=SEED)
TOP_XGB, TOP_MLP, TOP_GRU, SEQ, STEP = 400, 256, 48, 32, 5
HEADS = ["T_lgb", "T_xgb", "N_mlp", "N_gru"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Robust:
    def __init__(self, X):
        self.med = np.nanmedian(X, axis=0)
        q = np.nanpercentile(X, [25, 75], axis=0)
        self.iqr = np.where((q[1] - q[0]) > 1e-9, q[1] - q[0], np.nanstd(X, axis=0) + 1e-9)
        self.med, self.iqr = np.nan_to_num(self.med), np.nan_to_num(self.iqr, nan=1.0)

    def __call__(self, X):
        return np.nan_to_num(np.clip((X - self.med) / self.iqr, -5, 5), nan=0.0).astype("float32")


class MLP(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(k, 256), nn.ReLU(), nn.Dropout(.2), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(.2),
                               nn.Linear(128, 64), nn.ReLU())
        self.o = nn.Linear(64, 2)

    def forward(self, x):
        return self.o(self.f(x))


class GRU(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.g = nn.GRU(k, 64, batch_first=True)
        self.o = nn.Sequential(nn.Dropout(.2), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        return self.o(self.g(x)[0][:, -1])


def loss_fn(out, up, mag):
    return nn.functional.binary_cross_entropy_with_logits(out[:, 0], up) + nn.functional.huber_loss(nn.functional.softplus(out[:, 1]), mag, delta=1.5)


def train_net(net, make, tr, ca, up, mag, epochs, patience=2, bs=512):
    torch.manual_seed(SEED)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, state, bad, hist = np.inf, None, 0, []
    upt, magt = torch.tensor(up, dtype=torch.float32), torch.tensor(mag, dtype=torch.float32)
    rng = np.random.default_rng(SEED)
    for ep in range(epochs):
        net.train()
        order = rng.permutation(tr)
        for a in range(0, len(order), bs):
            b = order[a:a + bs]
            opt.zero_grad()
            loss_fn(net(make(b)), upt[b], magt[b]).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(np.mean([loss_fn(net(make(ca[a:a + 4096])), upt[ca[a:a + 4096]], magt[ca[a:a + 4096]]).item() for a in range(0, len(ca), 4096)]))
        hist.append(vl)
        if vl < best - 1e-5:
            best, state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad > patience:
                break
    net.load_state_dict(state)
    return hist


def net_predict(net, make, idx, bs=4096):
    net.eval()
    out = []
    with torch.no_grad():
        for a in range(0, len(idx), bs):
            o = net(make(idx[a:a + bs])).numpy()
            out.append(np.column_stack([1 / (1 + np.exp(-o[:, 0])), np.log1p(np.exp(o[:, 1]))]))
    return np.vstack(out)


def fit_fold(issue):
    d = OUT / "models" / str(issue.date())
    if (d / "fold.json").exists():
        print("exists", issue.date(), flush=True)
        return
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    names = feature_names()
    X = np.load(DATA / "hist_s5_X.npy", mmap_mode="r")
    meta = pd.read_feather(DATA / "hist_s5_meta.feather").set_index("timestamp_utc")
    idx = meta.index
    cal0 = issue - pd.Timedelta(days=30)
    win = np.flatnonzero((idx >= issue - pd.DateOffset(months=12)) & (idx < issue))
    lo, hi = win[0], win[-1] + 1
    sp = lo + int(np.searchsorted(idx[lo:hi], cal0))
    r, sc, av = meta.r.to_numpy(), meta.scale.to_numpy(), pd.DatetimeIndex(meta.avail)
    ok = np.isfinite(r) & meta.valid.fillna(False).to_numpy().astype(bool) & np.isfinite(sc)
    tr = np.arange(lo, sp)[ok[lo:sp] & (av[lo:sp] < cal0)]
    ca = np.arange(sp, hi)[ok[sp:hi] & (av[sp:hi] < issue)]
    up = (np.nan_to_num(r) > 0).astype("float32")
    mag = (np.abs(np.nan_to_num(r)) / sc).astype("float32")
    log = dict(issue=issue, train_rows=len(tr), cal_rows=len(ca), train_first=idx[tr[0]], train_last=idx[tr[-1]],
               cal_first=idx[ca[0]], cal_last=idx[ca[-1]], train_latest_label=av[tr].max(), cal_latest_label=av[ca].max(), heads={})
    assert log["train_latest_label"] < cal0 and log["cal_latest_label"] < issue
    view = X[lo:hi]
    Xtr, Xca = view[tr - lo], view[ca - lo]
    raw, pm = {}, {}
    # T_lgb
    ds = lgb.Dataset(Xtr, up[tr], feature_name=names, free_raw_data=False)
    b = lgb.train(dict(LGBP, objective="binary"), ds, 1500, valid_sets=[lgb.Dataset(Xca, up[ca], reference=ds)], callbacks=[lgb.early_stopping(120, verbose=False)])
    b.save_model(str(d / "T_lgb_dir.txt"), num_iteration=b.best_iteration)
    raw["T_lgb"] = b.predict(Xca, num_iteration=b.best_iteration)
    gain = pd.Series(b.feature_importance("gain", iteration=b.best_iteration), index=names).sort_values(ascending=False)
    gain.to_csv(d / "T_lgb_dir_gain.csv", header=["gain"])
    bm = lgb.train(dict(LGBP, objective="l2"), lgb.Dataset(Xtr, mag[tr], feature_name=names, free_raw_data=False), 1500,
                   valid_sets=[lgb.Dataset(Xca, mag[ca])], callbacks=[lgb.early_stopping(120, verbose=False)])
    bm.save_model(str(d / "T_lgb_mag.txt"), num_iteration=bm.best_iteration)
    pm["T_lgb"] = bm.predict(Xca, num_iteration=bm.best_iteration)
    mgain = pd.Series(bm.feature_importance("gain", iteration=bm.best_iteration), index=names).sort_values(ascending=False)
    log["heads"]["T_lgb"] = dict(dir_iter=b.best_iteration, mag_iter=bm.best_iteration)
    del ds
    # rank inputs for the smaller heads: interleave direction and magnitude gain ranks
    order = list(dict.fromkeys([n for pair in zip(gain.index, mgain.index) for n in pair]))
    pos = {n: j for j, n in enumerate(names)}
    sel = {k: [pos[n] for n in order[:n]] for k, n in (("xgb", TOP_XGB), ("mlp", TOP_MLP), ("gru", TOP_GRU))}
    # T_xgb
    c = sel["xgb"]
    dtr, dca = xgb.DMatrix(Xtr[:, c], up[tr]), xgb.DMatrix(Xca[:, c], up[ca])
    bx = xgb.train(dict(XGBP, objective="binary:logistic", eval_metric="logloss"), dtr, 1500, [(dca, "cal")], early_stopping_rounds=120, verbose_eval=False)
    raw["T_xgb"] = bx.predict(dca, iteration_range=(0, bx.best_iteration + 1))
    bx.save_model(str(d / "T_xgb_dir.json"))
    dtr.set_label(mag[tr]); dca.set_label(mag[ca])
    bxm = xgb.train(dict(XGBP, objective="reg:squarederror", eval_metric="rmse"), dtr, 1500, [(dca, "cal")], early_stopping_rounds=120, verbose_eval=False)
    pm["T_xgb"] = bxm.predict(dca, iteration_range=(0, bxm.best_iteration + 1))
    bxm.save_model(str(d / "T_xgb_mag.json"))
    log["heads"]["T_xgb"] = dict(dir_iter=bx.best_iteration, mag_iter=bxm.best_iteration)
    del dtr, dca
    # N_mlp
    c = sel["mlp"]
    rs = Robust(Xtr[:, c])
    Z = rs(view[:, c])
    zt = torch.tensor(Z)
    net = MLP(len(c))
    h = train_net(net, lambda b: zt[b - lo], tr, ca, up, mag, epochs=20, patience=3)
    o = net_predict(net, lambda b: zt[b - lo], ca)
    raw["N_mlp"], pm["N_mlp"] = o[:, 0], o[:, 1]
    torch.save(net.state_dict(), d / "N_mlp.pt")
    log["heads"]["N_mlp"] = dict(epochs=len(h), cal_loss=min(h))
    # N_gru: sequences of rows j-31 ... j (5-minute spacing in the strided matrix)
    c = sel["gru"]
    g0 = max(0, lo - SEQ)
    rg = Robust(Xtr[:, c])
    G = torch.tensor(rg(X[g0:hi][:, c]))
    offs = torch.arange(-(SEQ - 1), 1)

    def make(b):
        return G[torch.as_tensor(b - g0)[:, None] + offs[None, :]]
    gnet = GRU(len(c))
    trg = tr[tr - g0 >= SEQ - 1]
    h = train_net(gnet, make, trg, ca, up, mag, epochs=12, patience=2)
    o = net_predict(gnet, make, ca)
    raw["N_gru"], pm["N_gru"] = o[:, 0], o[:, 1]
    torch.save(gnet.state_dict(), d / "N_gru.pt")
    log["heads"]["N_gru"] = dict(epochs=len(h), cal_loss=min(h))
    joblib.dump(dict(sel={k: [names[j] for j in v] for k, v in sel.items()}, mlp_scaler=rs, gru_scaler=rg), d / "preprocess.joblib")
    ar = np.abs(r[ca])
    for k in HEADS:
        pl = LogisticRegression(C=1e6).fit(logit(raw[k]).reshape(-1, 1), up[ca])
        p = pl.predict_proba(logit(raw[k]).reshape(-1, 1))[:, 1]
        mb = pm[k] * sc[ca]
        log["heads"][k].update(platt_a=float(pl.coef_[0, 0]), platt_b=float(pl.intercept_[0]), cal_auc=float(roc_auc_score(up[ca], raw[k])),
                               cal_acc=float(((p > .5) == (up[ca] > 0)).mean()), cal_mag_mae=float(np.abs(mb - ar).mean()), cal_mag_bias=float((mb - ar).mean()))
    log["cal_const_mae"] = float(np.abs(np.median(np.abs(r[tr])) - ar).mean())
    log["seconds"] = round(time.time() - t0, 1)
    (d / "fold.json").write_text(json.dumps(log, indent=2, default=str))
    print("fitted", issue.date(), log["seconds"], {k: (round(v["cal_auc"], 4), round(v["cal_mag_mae"], 2)) for k, v in log["heads"].items()}, flush=True)


class Fold:
    def __init__(self, issue):
        d = OUT / "models" / str(issue.date())
        self.log = json.loads((d / "fold.json").read_text())
        self.pp = joblib.load(d / "preprocess.joblib")
        self.names = feature_names()
        pos = {n: j for j, n in enumerate(self.names)}
        self.c = {k: np.asarray([pos[n] for n in v]) for k, v in self.pp["sel"].items()}
        self.ld, self.lm = lgb.Booster(model_file=str(d / "T_lgb_dir.txt")), lgb.Booster(model_file=str(d / "T_lgb_mag.txt"))
        self.xd, self.xm = xgb.Booster(), xgb.Booster()
        self.xd.load_model(str(d / "T_xgb_dir.json")); self.xm.load_model(str(d / "T_xgb_mag.json"))
        self.mlp, self.gru = MLP(len(self.c["mlp"])), GRU(len(self.c["gru"]))
        self.mlp.load_state_dict(torch.load(d / "N_mlp.pt")); self.gru.load_state_dict(torch.load(d / "N_gru.pt"))

    def predict(self, F, seqsrc, seqpos, scale):
        """F: full-minute feature matrix (rows to score); seqsrc: scaled GRU inputs for a contiguous minute range;
        seqpos: row of each scored minute inside seqsrc."""
        raw, mag = {}, {}
        raw["T_lgb"], mag["T_lgb"] = self.ld.predict(F, num_threads=TH), self.lm.predict(F, num_threads=TH)
        dm = xgb.DMatrix(F[:, self.c["xgb"]])
        raw["T_xgb"], mag["T_xgb"] = self.xd.predict(dm), self.xm.predict(dm)
        zt = torch.tensor(self.pp["mlp_scaler"](F[:, self.c["mlp"]]))
        o = net_predict(self.mlp, lambda b: zt[torch.as_tensor(b)], np.arange(len(F)))
        raw["N_mlp"], mag["N_mlp"] = o[:, 0], o[:, 1]
        offs = torch.arange(-(SEQ - 1), 1) * STEP
        G = torch.tensor(seqsrc)
        o = net_predict(self.gru, lambda b: G[torch.as_tensor(seqpos[b])[:, None] + offs[None, :]], np.arange(len(F)))
        raw["N_gru"], mag["N_gru"] = o[:, 0], o[:, 1]
        out = {}
        for k in HEADS:
            a, b = self.log["heads"][k]["platt_a"], self.log["heads"][k]["platt_b"]
            out["raw_" + k] = raw[k]
            out["p_" + k] = 1 / (1 + np.exp(-(a * logit(raw[k]) + b)))
            out["m_" + k] = np.clip(mag[k], 0, None) * scale
        return out


def full_minutes(split, index, extra):
    """Full-minute feature matrix for `index` (3,270 cache fields + SOL extras)."""
    cols, _ = approved_columns()
    out = np.empty((len(index), len(cols) + extra.shape[1]), dtype="float32")
    ex = extra.reindex(index).to_numpy()
    for date, loc, rows in day_slices(index):
        p = INPUTS / split / (date + ".npy")
        if not p.exists():
            out[loc, :len(cols)] = np.nan
        else:
            out[loc, :len(cols)] = np.load(p, mmap_mode="r")[rows][:, cols]
        out[loc, len(cols):] = ex[loc]
    return out


def score_block(fold, split, index, extra, scale, pre_split=None):
    """Score `index`; the GRU warm-up uses the preceding 155 minutes from `pre_split` (or the same split)."""
    pad = (SEQ - 1) * STEP
    warm_idx = pd.date_range(index[0] - pad * pd.Timedelta(minutes=1), periods=pad, freq="min", tz="UTC")
    res = []
    for q0 in range(0, len(index), 1440 * 7):
        part = index[q0:q0 + 1440 * 7]
        lead = warm_idx if q0 == 0 else index[q0 - pad:q0]
        lead_split = (pre_split or split) if q0 == 0 else split
        F = full_minutes(split, part, extra)
        if q0 == 0 and pre_split is None and split != "historical":
            L = np.full((pad, F.shape[1]), np.nan, dtype="float32")  # no earlier release minutes available
        else:
            L = full_minutes(lead_split, lead, extra)
        S = fold.pp["gru_scaler"](np.vstack([L, F])[:, fold.c["gru"]])
        res.append(pd.DataFrame(fold.predict(F, S, np.arange(pad, pad + len(part)), scale.reindex(part).to_numpy()), index=part))
    return pd.concat(res)


def score_hist_fold(issue):
    sol, eth = prices("historical")
    extra, scale = price_block(sol, eth)
    idx = sol.index[(sol.index >= issue) & (sol.index < issue + pd.DateOffset(months=3))]
    f = score_block(Fold(issue), "historical", idx, extra, scale).assign(issue=issue)
    f.to_parquet(OUT / "models" / str(issue.date()) / "oof_forecasts.parquet")
    print("scored", issue.date(), len(f), flush=True)


def main():
    stage = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    if stage == "fit":
        (OUT / "training_config.json").write_text(json.dumps(dict(lgb=LGBP, xgb=XGBP, top=dict(xgb=TOP_XGB, mlp=TOP_MLP, gru=TOP_GRU),
            gru_seq=SEQ, gru_step_minutes=STEP, folds=[str(f) for f in FOLDS], frozen=str(FROZEN), horizon=H, entry_bps=12,
            features=len(feature_names()), doc=__doc__), indent=2))
        for issue in FOLDS:
            fit_fold(issue)
    elif stage == "score_hist":
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(int(os.environ.get("SOL60_WORKERS", "2"))) as pool:
            list(pool.map(score_hist_fold, FOLDS))
    elif stage == "score_4a":
        fold = Fold(FROZEN)
        hs, he = prices("historical")
        vs, ve = prices("validation")
        ts, te = prices("test")
        parts = []
        ev, sv = price_block(pd.concat([hs.iloc[-WARM:], vs]), pd.concat([he.iloc[-WARM:], ve]))
        parts.append(score_block(fold, "validation", vs.index, ev, sv).assign(block="validation"))
        et, st = price_block(pd.concat([vs.iloc[-WARM:], ts]), pd.concat([ve.iloc[-WARM:], te]))
        parts.append(score_block(fold, "test", ts.index, pd.concat([ev, et[~et.index.isin(ev.index)]]), st, pre_split="validation").assign(block="test"))
        pd.concat(parts).assign(issue=FROZEN).to_parquet(OUT / "handoff4a_forecasts.parquet")
        print("scored 4a", flush=True)


if __name__ == "__main__":
    main()
