r"""Train UP GRU / DOWN three-window hybrid for conditional means, then combine.

Run from Week 2: python .\Code\fit_conditional_means_signed_48h.py
NEW training on execution. No fitting on import. Existing artifacts stay unchanged.
Dependencies: numpy, pandas, scipy, scikit-learn, torch, joblib, threadpoolctl.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.metadata
import json

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
WEEK1 = HERE.parent.parent / 'Week 1'
EXP = HERE / 'experiments'
COLS = [
    'F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__15m__value',
    'F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__5m__value',
    'F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__1h__value',
    'PRICE__vol_ratio_4h_24h', 'PRICE__realized_vol_168h',
    'F4Z__liquidation__venue_breadth_percentile__1h__value',
    'F4Z__funding__funding_persistence_percentile__5m__value',
    'F4Z__basis__basis_level_rz__1h__value',
]
STARTS = pd.to_datetime(['2024-01-01', '2024-04-01', '2024-07-01', '2024-10-01'], utc=True)
WEIGHTS = [0., .1, .25, .5]
STEPS, SEED = 48, 48


def check(ok, message):
    if not ok:
        raise ValueError(message)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    frame = pd.read_csv(path)
    frame['decision_time'] = pd.to_datetime(frame.decision_time, utc=True, errors='raise')
    check(len(frame) > 0 and not frame.decision_time.isna().any()
          and not frame.decision_time.duplicated().any(), f'Invalid timestamps: {path}')
    return frame.set_index('decision_time').sort_index()


def score(actual, prediction):
    a, p = np.asarray(actual, float), np.asarray(prediction, float)
    check(a.shape == p.shape and a.size > 0 and np.isfinite(a).all()
          and np.isfinite(p).all(), 'Invalid scoring arrays')
    err = p - a
    ic = pd.Series(p).corr(pd.Series(a), method='spearman') if np.ptp(p) > 0 and np.ptp(a) > 0 else None
    return dict(rows=len(a), mae_bps=float(np.abs(err).mean()),
                rmse_bps=float(np.sqrt(np.mean(err**2))), mse_bps2=float(np.mean(err**2)),
                bias_bps=float(err.mean()), rank_ic=float(ic) if ic is not None and np.isfinite(ic) else None)


def prep(x, keys, robust):
    a = x.loc[keys, COLS]
    median = a.median().fillna(0.)
    filled = a.fillna(median)
    if robust:
        center = median
        scale = filled.quantile(.75) - filled.quantile(.25)
        scale = scale.where(scale > 1e-6, filled.std(ddof=0)).fillna(1.).clip(lower=1e-6)
    else:
        center = filled.mean()
        scale = filled.std(ddof=0).replace(0, 1.).fillna(1.)
    return dict(columns=COLS, median=median, center=center, scale=scale)


def design(x, p, clip=False):
    a = x[p['columns']]
    z = (a.fillna(p['median']) - p['center']) / p['scale']
    if clip:
        z = z.clip(-8, 8)
    values = np.column_stack([z.to_numpy(float), a.isna().to_numpy(float)])
    check(np.isfinite(values).all(), 'Nonfinite transformed features')
    return values


class MeanGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(16, 16, batch_first=True)
        self.head = nn.Linear(16, 1)

    def forward(self, a):
        _, hidden = self.gru(a)
        return nn.functional.softplus(self.head(hidden[-1])).squeeze(1)


def sequence_loader(x, keys, p, target=None):
    # Only snapshots at/before each origin. Missing timestamps keep their slots.
    values = design(x, p, clip=True).astype(np.float32)
    values = np.vstack([values, np.r_[np.zeros(8), np.ones(8)].astype(np.float32)])
    offsets = np.arange(STEPS - 1, -1, -1) * pd.Timedelta(minutes=30).value
    times = keys.asi8[:, None] - offsets[None, :]
    ids = x.index.get_indexer(pd.to_datetime(times.ravel(), utc=True)).reshape(len(keys), STEPS)
    missing = int((ids < 0).sum())
    ids[ids < 0] = len(values) - 1
    targets = None if target is None else target.loc[keys].to_numpy(np.float32) / 100.

    class Sequences(torch.utils.data.Dataset):
        def __len__(self):
            return len(keys)

        def __getitem__(self, i):
            a = torch.from_numpy(values[ids[i]])
            return a if targets is None else (a, torch.tensor(targets[i]))

    loader = torch.utils.data.DataLoader(Sequences(), batch_size=256,
                                       shuffle=target is not None, num_workers=0)
    return loader, missing


def infer_gru(model, loader):
    model.eval()
    with torch.inference_mode():
        return np.concatenate([model(a).numpy().astype(float) * 100. for a in loader])


def fit_gru(x, magnitude, train, valid=None, epochs=None):
    torch.manual_seed(SEED)
    p = prep(x, train, robust=True)
    loader, missing = sequence_loader(x, train, p, magnitude)
    model = MeanGRU()
    # Initialize the positive output at the TRAINING SIDE MEAN, not median.
    mean_pct = max(float(magnitude.loc[train].mean()) / 100., 1e-6)
    nn.init.zeros_(model.head.weight)
    nn.init.constant_(model.head.bias, mean_pct + np.log(-np.expm1(-mean_pct)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    history = []
    best_state, best_epoch, stale = copy.deepcopy(model.state_dict()), 0, 0
    if epochs is None:
        check(valid is not None and len(valid) > 0, 'GRU epoch selection needs validation rows')
        val_loader, _ = sequence_loader(x, valid, p)
        best = score(magnitude.loc[valid], infer_gru(model, val_loader))['mse_bps2']
        history.append(dict(epoch=0, validation_mse_bps2=best))
    for epoch in range(1, (24 if epochs is None else epochs) + 1):
        model.train()
        for a, b in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(a) - b)**2).mean()
            check(bool(torch.isfinite(loss)), 'Nonfinite GRU squared-error loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        if epochs is None:
            mse = score(magnitude.loc[valid], infer_gru(model, val_loader))['mse_bps2']
            history.append(dict(epoch=epoch, validation_mse_bps2=mse))
            print(f'  UP GRU epoch {epoch}: validation MSE {mse:.3f}', flush=True)
            if mse < best - 1e-4:
                best, best_epoch, best_state, stale = mse, epoch, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
            if stale >= 5:
                break
        else:
            best_epoch, best_state = epoch, copy.deepcopy(model.state_dict())
    return dict(prep=p, state={k: v.cpu().numpy() for k, v in best_state.items()},
                epochs=best_epoch, history=history, training_rows=len(train),
                train_last=train.max(), missing_sequence_slots=missing)


def predict_gru(artifact, x, keys):
    model = MeanGRU()
    model.load_state_dict({k: torch.as_tensor(v) for k, v in artifact['state'].items()})
    loader, missing = sequence_loader(x, keys, artifact['prep'])
    return infer_gru(model, loader), missing


def fit_hybrid(x, magnitude, train):
    check(len(train) >= 4096, 'DOWN RF needs at least 4096 eligible training rows')
    p = prep(x, train, robust=False)
    z, target = design(x.loc[train], p), magnitude.loc[train].to_numpy(float)
    linear = Ridge(alpha=1.).fit(z, target)
    forest = RandomForestRegressor(n_estimators=128, criterion='squared_error',
        max_depth=3, min_samples_leaf=128, max_features=1., bootstrap=True,
        max_samples=4096, n_jobs=1, random_state=SEED).fit(z, target)
    return dict(prep=p, linear=linear, forest=forest)


def hybrid_parts(artifact, x, keys):
    z = design(x.loc[keys], artifact['prep'])
    return artifact['linear'].predict(z), artifact['forest'].predict(z)


def blend(parts, weight):
    # Project onto the valid nonnegative magnitude range; no log conversion.
    return np.maximum(0., (1. - weight) * parts[0] + weight * parts[1])


def predict_means(bundle, x, keys):
    """Label-free inference: x is timestamped feature history, keys are origins."""
    check(len(keys) > 0 and x.index.is_unique and x.index.is_monotonic_increasing and keys.is_unique,
          'Inference timestamps must be unique and feature history sorted')
    check(keys.isin(x.index).all() and (keys >= pd.Timestamp(bundle['issue'])).all(),
          'Missing origin features or model issued after forecast origin')
    x = x[COLS].replace([np.inf, -np.inf], np.nan)
    up, missing = predict_gru(bundle['up'], x, keys)
    down = blend(hybrid_parts(bundle['down'], x, keys), bundle['down_weight'])
    check(np.isfinite(up).all() and np.isfinite(down).all()
          and (up >= 0).all() and (down >= 0).all(), 'Invalid conditional mean estimates')
    return up, down, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=WEEK1 / 'Code/experiments/eight_feature_exhaustive_v1/prepared.joblib')
    parser.add_argument('--direction-run', type=Path, default=EXP / 'direction_v45_style_v1/20261112T045009318177Z')
    parser.add_argument('--output-dir', type=Path, default=EXP / 'conditional_means_signed_48h_v1')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    direction_path = args.direction_run / 'development_predictions.csv'
    done = json.loads((args.direction_run / 'COMPLETE.json').read_text())
    check(done.get('status') == 'COMPLETE', 'Direction run is incomplete')
    check(done['artifacts']['development_predictions.csv'] == sha(direction_path), 'Direction prediction hash mismatch')
    direction = read_csv(direction_path)
    pipeline_path = args.direction_run / 'final/pipeline.joblib'
    artifact_hashes = {k.replace('\\', '/'): v for k, v in done['artifacts'].items()}
    check(artifact_hashes.get('final/pipeline.joblib') == sha(pipeline_path), 'Direction pipeline hash mismatch')
    direction_cutoff = pd.Timestamp(joblib.load(pipeline_path)['max_label_end'])
    check(direction_cutoff < direction.index.min(), 'Direction training outcomes overlap forecast origins')
    x, y, dev, _ = joblib.load(args.prepared)
    check(x.index.equals(y.index) and x.index.is_unique and x.index.is_monotonic_increasing,
          'Prepared feature/label indexes differ or are not unique/sorted')
    check(x.index.tz is not None and y.index.tz is not None, 'Prepared times must be timezone-aware')
    y = y.copy()
    y['label_end'] = pd.to_datetime(y.label_end, utc=True)
    check((y.label_end == y.index + pd.Timedelta(hours=48)).all(), 'Targets are not 48-hour endpoints')
    dev = pd.DatetimeIndex(dev).sort_values()
    check(len(dev) == 18481 and dev.equals(direction.index) and dev.isin(y.index).all(),
          'Expected exact 18,481-row direction/development alignment; no row dropping')
    x = x[COLS].replace([np.inf, -np.inf], np.nan)
    signed = pd.Series(10000. * np.expm1(y.signed_return_log_bps.to_numpy(float) / 10000.), index=y.index)
    check(np.isfinite(signed).all(), 'Invalid ordinary-return targets')
    check((signed.loc[dev] != 0).all(), 'Flat development outcomes need an explicit three-way policy')
    magnitude = signed.abs()
    check(np.array_equal(direction.y.to_numpy(), (signed.loc[dev] > 0).astype(int).to_numpy()),
          'Direction outcomes disagree with magnitude outcomes')
    check(np.isfinite(direction[['p_up', 'p_down']]).all().all()
          and direction.p_up.between(0, 1).all() and direction.p_down.between(0, 1).all(),
          'Invalid direction probabilities')
    sum_error = float((direction.p_up + direction.p_down - 1).abs().max())
    check(sum_error <= 2 * np.finfo(np.float32).eps, 'Direction probabilities do not sum to one')
    direction['p_up'] = direction.p_up.astype(float)
    direction['p_down'] = 1. - direction.p_up
    out = args.output_dir / pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    contract = dict(status='STARTED', symbol='BTCUSDT', horizon_hours=48,
        units='ordinary bps; percent = bps/100; DOWN mean magnitude stored positive',
        target='Conditional average absolute endpoint return separately on UP and DOWN rows',
        up_model='8 features + missing flags; 48 half-hour snapshots; GRU16 + softplus; MSE loss',
        up_selection='MSE on preceding first two validation months; max24 epochs, patience5; training-mean initialization',
        down_model='Ridge alpha1 + squared-error RF128, depth3, leaf128, bootstrap4096; nonnegative blend',
        down_selection='Row-weighted validation MSE across cutoffs 9/6/3 months before issuance; RF weights 0/.1/.25/.5',
        features=COLS, seed=SEED, starts=list(STARTS),
        training_rule='label_end strictly earlier than each fit cutoff; side filter uses matured labels',
        formula='p_up * up_mean_bps - p_down * down_mean_magnitude_bps',
        direction_run=str(args.direction_run), direction_retrained=False,
        direction_max_training_label_end=direction_cutoff, direction_pipeline_sha256=sha(pipeline_path),
        direction_exact_v45=done.get('exact_v45_reproduction'),
        probability_sum_error_before_correction=sum_error,
        prepared_sha256=sha(args.prepared), direction_sha256=sha(direction_path),
        source_sha256=sha(Path(__file__)),
        limitation='Mean-targeting loss does not guarantee calibrated conditional means. Historical exposed evaluation; no automatic winner selection.',
        versions={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'torch', 'joblib']})
    dump(out / 'CONTRACT.json', contract)
    frames, audits, validation = [], [], []
    for issue in STARTS:
        end = issue + pd.DateOffset(months=3) if issue != STARTS[-1] else dev.max() + pd.Timedelta(minutes=30)
        te = dev[(dev >= issue) & (dev < end)]
        ot = y.index[y.label_end < issue]
        up_train, down_train = ot[signed.loc[ot] > 0], ot[signed.loc[ot] < 0]
        inner = issue - pd.DateOffset(months=3)
        ti = y.index[(y.label_end < inner) & (signed > 0)]
        va = y.index[(y.index >= inner) & (y.index < issue - pd.DateOffset(months=1))
                     & (y.label_end < issue) & (signed > 0)]
        check(min(len(ti), len(va), len(te), len(up_train)) >= 100, 'Insufficient UP training/validation/test data')
        print(f'START {issue.date()}: UP mean GRU selection', flush=True)
        chosen = fit_gru(x, magnitude, ti, va)
        validation.extend(dict(issue=str(issue), side='UP', **h) for h in chosen['history'])
        up = fit_gru(x, magnitude, up_train, epochs=chosen['epochs'])
        trials = []
        for months in [9, 6, 3]:
            cutoff = issue - pd.DateOffset(months=months)
            tr = y.index[(y.label_end < cutoff) & (signed < 0)]
            vk = y.index[(y.index >= cutoff) & (y.index < cutoff + pd.DateOffset(months=2))
                         & (y.label_end < cutoff + pd.DateOffset(months=3)) & (signed < 0)]
            check(len(vk) >= 100, 'Insufficient DOWN validation rows')
            print(f'  DOWN mean hybrid: validation cutoff {cutoff.date()}', flush=True)
            parts = hybrid_parts(fit_hybrid(x, magnitude, tr), x, vk)
            for weight in WEIGHTS:
                trials.append(dict(issue=str(issue), side='DOWN', months=months, weight=weight,
                                   **score(magnitude.loc[vk], blend(parts, weight))))
        validation.extend(trials)
        def objective(weight):
            rows = [r for r in trials if r['weight'] == weight]
            return sum(r['mse_bps2'] * r['rows'] for r in rows) / sum(r['rows'] for r in rows), weight
        weight = min(WEIGHTS, key=objective)
        bundle = dict(issue=issue, horizon_hours=48, units='ordinary bps', up=up,
                      down=fit_hybrid(x, magnitude, down_train), down_weight=weight,
                      up_max_label_end=y.loc[up_train, 'label_end'].max(),
                      down_max_label_end=y.loc[down_train, 'label_end'].max())
        up_pred, down_pred, missing = predict_means(bundle, x, te)
        model_path = out / f'{issue:%Y-%m}_mean_models.joblib'
        joblib.dump(bundle, model_path, compress=3)
        replay_up, replay_down, _ = predict_means(joblib.load(model_path), x, te)
        np.testing.assert_allclose(up_pred, replay_up, rtol=0, atol=1e-6)
        np.testing.assert_allclose(down_pred, replay_down, rtol=0, atol=1e-6)
        f = direction.loc[te, ['p_up', 'p_down']].copy()
        f['symbol'], f['horizon_hours'], f['issue'] = 'BTCUSDT', 48, issue
        f['forecast_end'] = te + pd.Timedelta(hours=48)
        f['up_mean_bps'], f['down_mean_magnitude_bps'] = up_pred, down_pred
        f['up_mean_pct'], f['down_mean_magnitude_pct'] = up_pred / 100., down_pred / 100.
        f['up_contribution_bps'] = f.p_up * f.up_mean_bps
        f['down_contribution_bps'] = -f.p_down * f.down_mean_magnitude_bps
        f['signed_return_forecast_bps'] = f.up_contribution_bps + f.down_contribution_bps
        f['signed_return_forecast_pct'] = f.signed_return_forecast_bps / 100.
        f['up_train_mean_bps'], f['down_train_mean_bps'] = magnitude.loc[up_train].mean(), magnitude.loc[down_train].mean()
        f['side_mean_baseline_bps'] = f.p_up * f.up_train_mean_bps - f.p_down * f.down_train_mean_bps
        f['signed_mean_baseline_bps'] = signed.loc[ot].mean()
        f['zero_baseline_bps'] = 0.
        f['actual_signed_return_bps'] = signed.loc[te]
        f['actual_signed_return_pct'] = signed.loc[te] / 100.
        f.to_csv(out / f'{issue:%Y-%m}_predictions.csv', index_label='decision_time')
        frames.append(f)
        audits.append(dict(issue=issue, up_rows=len(up_train), down_rows=len(down_train),
            up_epochs=chosen['epochs'], down_rf_weight=weight, replay_passed=True,
            up_max_label_end=bundle['up_max_label_end'], down_max_label_end=bundle['down_max_label_end'],
            missing_up_sequence_slots=missing, model_sha256=sha(model_path)))
        dump(out / 'FIT_AUDIT.json', audits)
        pd.DataFrame(validation).to_csv(out / 'validation_scores.csv', index=False)
        print(f'DONE {issue.date()}: UP epochs={chosen["epochs"]}; DOWN RF weight={weight}', flush=True)
    pred = pd.concat(frames).sort_index()
    check(pred.index.equals(dev), 'Final prediction timestamps do not match development timestamps')
    pred.to_csv(out / 'signed_return_predictions.csv', index_label='decision_time')
    rows, sides = [], []
    for period, f in [('ALL', pred)] + [(str(k), g) for k, g in pred.groupby('issue')]:
        for name, column in [('combined_mean_models', 'signed_return_forecast_bps'),
                             ('same_probabilities_side_means', 'side_mean_baseline_bps'),
                             ('training_signed_mean', 'signed_mean_baseline_bps'), ('zero', 'zero_baseline_bps')]:
            rows.append(dict(period=period, model=name, **score(f.actual_signed_return_bps, f[column])))
        for side, mask, col, base in [
            ('UP', f.actual_signed_return_bps > 0, 'up_mean_bps', 'up_train_mean_bps'),
            ('DOWN', f.actual_signed_return_bps < 0, 'down_mean_magnitude_bps', 'down_train_mean_bps')]:
            g = f.loc[mask]
            if not g.empty:
                for model, c in [('mean_model', col), ('training_side_mean', base)]:
                    sides.append(dict(period=period, side=side, model=model,
                        **score(g.actual_signed_return_bps.abs(), g[c])))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / 'signed_metrics.csv', index=False)
    pd.DataFrame(sides).to_csv(out / 'magnitude_metrics.csv', index=False)
    summary = ('48-hour signed forecast from new conditional-mean models. Ordinary bps.\n'
        'UP GRU: squared-error loss. DOWN hybrid: squared-error RF + Ridge; three-window MSE blend selection.\n'
        'No development-outcome branch choice or automatic model promotion.\n'
        'Mean targeting is not proof of calibration or improvement. Historical exposed evaluation.\n'
        'No uncertainty analysis, trading or fresh-data confirmation in this runner.\n\n'
        + metrics.to_string(index=False))
    (out / 'SUMMARY.txt').write_text(summary, encoding='utf-8')
    dump(out / 'COMPLETE.json', dict(status='COMPLETE', rows=len(pred), folds=len(audits),
        training_performed=True, direction_retrained=False, saved_model_replay_passed=True,
        contract_sha256=sha(out / 'CONTRACT.json'), prediction_sha256=sha(out / 'signed_return_predictions.csv')))
    print(summary)
    print(f'Outputs saved: {out}')


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
