r"""Tune UP/DOWN conditional mean MAGNITUDES ONLY. New CPU training.

python .\Code\tune_conditional_mean_magnitudes_48h.py
No direction probabilities, signed-return forecasts or trading calculations.
Requires numpy, pandas, scipy, scikit-learn, joblib and threadpoolctl.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import warnings
import gc

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
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
PACKS = {'four': [COLS[i] for i in [0, 1, 3, 5]],
         'five': [COLS[i] for i in [0, 1, 3, 5, 4]], 'eight': COLS}
ISSUES = pd.to_datetime(['2024-01-01', '2024-04-01', '2024-07-01', '2024-10-01'], utc=True)
FAMILIES = ['ridge', 'rf', 'extra_trees', 'boosting', 'mlp']


def check(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8388608), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding='utf-8')


def metric(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    check(len(y) > 0 and y.shape == p.shape and np.isfinite(y).all() and np.isfinite(p).all(), 'Invalid scoring arrays')
    error = p - y
    ic = pd.Series(p).corr(pd.Series(y), method='spearman') if np.ptp(p) > 0 and np.ptp(y) > 0 else None
    return dict(rows=len(y), mse_bps2=float(np.mean(error**2)), rmse_bps=float(np.sqrt(np.mean(error**2))),
                mae_bps=float(np.abs(error).mean()), bias_bps=float(error.mean()),
                rank_ic=float(ic) if ic is not None and np.isfinite(ic) else None,
                predicted_std_bps=float(p.std()), mean_prediction_bps=float(p.mean()), mean_actual_bps=float(y.mean()))


def candidates(count, seed):
    """Fixed seeded search schedule, generated before reading evaluation outcomes."""
    rng = np.random.default_rng(seed)
    result, seen = [], set()
    def pick(values):
        return values[int(rng.integers(len(values)))]
    for i in range(count):
        family = FAMILIES[i % len(FAMILIES)]
        for _ in range(10000):
            c = dict(family=family, pack=pick(['four', 'five', 'eight']), history_days=pick([None, 365, 540]))
            if family == 'ridge':
                c['alpha_per_row'] = pick([.0001, .001, .01, .1, 1.])
            elif family in ['rf', 'extra_trees']:
                c.update(trees=pick([128, 256]), depth=pick([3, 6, 10]), leaf=pick([64, 128, 256]),
                         max_features=pick([.5, 1.]), sample=pick([4096, None]))
            elif family == 'boosting':
                c.update(leaves=pick([3, 7, 15]), leaf=pick([128, 256, 512]), iterations=pick([60, 120, 200]),
                         learning_rate=pick([.02, .05]), l2=pick([1., 10., 100.]))
            else:
                c.update(hidden=pick([[8, 4], [16, 8], [32, 16], [32, 16, 8]]),
                         alpha=pick([.01, .1, 1., 10.]), learning_rate=pick([.0003, .001]),
                         iterations=pick([80, 160]), activation=pick(['relu', 'tanh']))
            signature = json.dumps(c, sort_keys=True)
            if signature not in seen:
                seen.add(signature)
                c['id'] = f'trial_{i:03d}'
                result.append(c)
                break
        else:
            raise ValueError('Could not generate unique candidates')
    return result


def train_keys(y, side_mask, cutoff, days):
    mask = (y.label_end < cutoff) & side_mask
    if days is not None:
        mask &= y.index >= cutoff - pd.Timedelta(days=days)
    keys = y.index[mask]
    check(len(keys) >= 1024, f'Fewer than1024 side-specific training rows at {cutoff}, history={days}')
    check(y.loc[keys, 'label_end'].max() < cutoff, 'Training-label purge failed')
    return keys


def transform(x, prep):
    a = x[prep['columns']]
    z = ((a.fillna(prep['median']) - prep['center']) / prep['scale']).clip(-8, 8)
    values = np.column_stack([z.to_numpy(float), a.isna().to_numpy(float)])
    check(np.isfinite(values).all(), 'Nonfinite feature design')
    return values


def fit(x, target, keys, config, seed):
    a = x.loc[keys, PACKS[config['pack']]]
    median = a.median().fillna(0.)
    filled = a.fillna(median)
    prep = dict(columns=list(a.columns), median=median, center=filled.mean(),
                scale=filled.std(ddof=0).where(lambda v: v > 1e-8, 1.).fillna(1.))
    z = transform(x.loc[keys], prep)
    yy = target.loc[keys].to_numpy(float)
    # Affine scaling preserves the conditional mean and squared-error optimum.
    # No log transform, target clipping, outlier deletion or sample reweighting.
    center, scale = float(yy.mean()), max(float(yy.std()), 1.)
    normalized = (yy - center) / scale
    family = config['family']
    if family == 'ridge':
        model = Ridge(alpha=config['alpha_per_row'] * len(keys))
    elif family in ['rf', 'extra_trees']:
        cls = RandomForestRegressor if family == 'rf' else ExtraTreesRegressor
        sampled = config['sample']
        if sampled is not None:
            sampled = min(sampled, len(keys))
        model = cls(n_estimators=config['trees'], criterion='squared_error', max_depth=config['depth'],
                    min_samples_leaf=config['leaf'], max_features=config['max_features'],
                    bootstrap=sampled is not None, max_samples=sampled, n_jobs=1, random_state=seed)
    elif family == 'boosting':
        model = HistGradientBoostingRegressor(loss='squared_error', max_iter=config['iterations'],
            max_leaf_nodes=config['leaves'], min_samples_leaf=config['leaf'], learning_rate=config['learning_rate'],
            l2_regularization=config['l2'], early_stopping=False, random_state=seed)
    else:
        model = MLPRegressor(hidden_layer_sizes=tuple(config['hidden']), activation=config['activation'],
            solver='adam', alpha=config['alpha'], learning_rate_init=config['learning_rate'],
            batch_size=256, max_iter=config['iterations'], early_stopping=False,
            n_iter_no_change=config['iterations'] + 1, tol=0., shuffle=True, random_state=seed)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(z, normalized)
    return dict(model=model, prep=prep, target_center=center, target_scale=scale, config=config,
                fit_warnings=[str(w.message) for w in captured], training_rows=len(keys),
                train_first=keys.min(), train_last=keys.max())


def predict(artifact, x):
    p = artifact['model'].predict(transform(x, artifact['prep'])) * artifact['target_scale'] + artifact['target_center']
    check(np.isfinite(p).all(), 'Nonfinite model prediction')
    # Magnitude is nonnegative; projection does not delete large outcomes.
    return np.maximum(0., p)


def predict_bundle(bundle, features):
    """Label-free predictions for all supplied origins: best and top-three average."""
    check(features.index.is_unique and features.index.tz is not None and len(features) > 0,
          'Inference requires unique timezone-aware origin timestamps')
    check((features.index >= pd.Timestamp(bundle['issue'])).all(), 'Model issued after inference origin')
    features = features.replace([np.inf, -np.inf], np.nan)
    predictions = [predict(a, features) for a in bundle['models']]
    return pd.DataFrame({'best_mean_bps': predictions[0], 'ensemble_mean_bps': np.mean(predictions, axis=0)}, index=features.index)


def load_reference(path):
    f = pd.read_csv(path, usecols=['decision_time', 'issue', 'actual_signed_return_bps',
        'UP_mlp_mean_bps', 'DOWN_mlp_mean_bps', 'UP_rf_mean_bps', 'DOWN_rf_mean_bps',
        'UP_reference_mean_bps', 'DOWN_reference_mean_bps'])
    f['decision_time'] = pd.to_datetime(f.decision_time, utc=True)
    check(not f.decision_time.isna().any() and not f.decision_time.duplicated().any(), 'Invalid reference times')
    f['issue'] = pd.to_datetime(f.issue, utc=True)
    return f.set_index('decision_time').sort_index()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trials', type=int, default=25, help='Total configurations, balanced over five families (default25)')
    p.add_argument('--seed', type=int, default=48)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--prepared', type=Path, default=HERE.parent.parent / 'Week 1/Code/experiments/eight_feature_exhaustive_v1/prepared.joblib')
    p.add_argument('--reference-run', type=Path, default=EXP / 'mlp_rf_conditional_means_48h_v1/20261112T064753697944Z')
    p.add_argument('--output-dir', type=Path, default=EXP / 'conditional_mean_magnitude_tuning_v1')
    args = p.parse_args()
    check(5 <= args.trials <= 100 and args.threads > 0, 'Use5-100 trials and positive thread count')
    configs = candidates(args.trials, args.seed)
    reference_path = args.reference_run / 'signed_return_predictions.csv'
    done = json.loads((args.reference_run / 'COMPLETE.json').read_text())
    check(done['status'] == 'COMPLETE' and done['prediction_sha256'] == sha(reference_path), 'Reference integrity check failed')
    ref = load_reference(reference_path)
    x, y, dev, _ = joblib.load(args.prepared)
    check(x.index.equals(y.index) and x.index.is_unique and x.index.is_monotonic_increasing
          and x.columns.is_unique and x.index.tz is not None, 'Invalid prepared indexes')
    y = y.copy()
    y['label_end'] = pd.to_datetime(y.label_end, utc=True)
    check((y.label_end == y.index + pd.Timedelta(hours=48)).all(), 'Unexpected target horizon')
    dev = pd.DatetimeIndex(dev).sort_values()
    check(len(dev) == 18481 and dev.equals(ref.index) and dev.isin(y.index).all(), 'Evaluation timestamps must match exactly')
    x = x[COLS].replace([np.inf, -np.inf], np.nan)
    log_target = y.signed_return_log_bps.astype(float)
    target = pd.Series(np.abs(10000. * np.expm1(log_target.to_numpy() / 10000.)), index=y.index)
    check(np.isfinite(target).all() and (log_target.loc[dev] != 0).all(), 'Nonfinite or flat development targets')
    np.testing.assert_allclose(target.loc[dev], ref.actual_signed_return_bps.abs(), rtol=0, atol=1e-7)
    check(np.array_equal(np.sign(log_target.loc[dev]), np.sign(ref.actual_signed_return_bps)), 'Actual side disagreement')
    out = args.output_dir / pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    dump(out / 'CONTRACT.json', dict(status='STARTED', task='UP/DOWN mean magnitude tuning only',
        horizon_hours=48, symbol='BTCUSDT', target='positive ordinary-bps conditional mean magnitude',
        features=PACKS, configs=configs, seed=args.seed, offsets_months=[9, 6, 3],
        preprocessing='Train-only median imputation, mean/std scaling, +/-8 feature clipping, missing flags; train-only affine target scaling',
        loss='squared error for all learned models; no target clipping or log-target regression',
        selection='Row-weighted MSE over three purged chronological validation windows separately per side/fold',
        validation='First two months following each inner cutoff; validation outcomes mature before end of that three-month block',
        policies=['best earlier-validation model', 'equal-weight mean of top3 earlier-validation models'],
        policies_not_test_selected=True, baseline_not_selection_candidate=True,
        training_cutoff='label_end strictly before fit cutoff; candidate history may be expanding/365/540 days',
        fit_budget_inner=4 * 2 * 3 * args.trials, fit_budget_outer=4 * 2 * 3,
        predictions_on_all_origins=True, signed_return_forecasts=False, direction_model_used=False,
        source_sha256=sha(Path(__file__)), prepared_sha256=sha(args.prepared), reference_sha256=sha(reference_path),
        versions={v: importlib.metadata.version(v) for v in ['numpy', 'pandas', 'scikit-learn', 'joblib']},
        limits='Historically exposed evaluation; search does not guarantee lower error. One optimization seed; MLP fixed epoch budget and convergence warnings recorded. No uncertainty or new feature-source audit.'))
    validations, rankings, audits, diagnostics, predictions = [], [], [], [], []
    with threadpool_limits(limits=args.threads):
        for issue in ISSUES:
            te = ref.index[ref.issue.eq(issue)]
            check(len(te) > 0, 'Empty evaluation fold')
            frame = pd.DataFrame({'issue': str(issue), 'symbol': 'BTCUSDT', 'horizon_hours': 48,
                'actual_direction': np.where(log_target.loc[te] > 0, 'UP', 'DOWN'),
                'actual_magnitude_bps': target.loc[te]}, index=te)
            for side in ['UP', 'DOWN']:
                side_mask = log_target > 0 if side == 'UP' else log_target < 0
                print(f'START {issue.date()} {side}: {args.trials} trials x3 validation windows', flush=True)
                side_trials = []
                for config in configs:
                    squared_sum, count = 0., 0
                    for months in [9, 6, 3]:
                        cutoff = issue - pd.DateOffset(months=months)
                        tr = train_keys(y, side_mask, cutoff, config['history_days'])
                        va = y.index[(y.index >= cutoff) & (y.index < cutoff + pd.DateOffset(months=2))
                                     & (y.label_end < cutoff + pd.DateOffset(months=3)) & side_mask]
                        check(len(va) >= 100 and y.loc[va, 'label_end'].max() < issue, 'Invalid validation window')
                        print(f'  {config["id"]} {config["family"]}/{config["pack"]} history={config["history_days"]} offset={months}', flush=True)
                        artifact = fit(x, target, tr, config, args.seed)
                        m = metric(target.loc[va], predict(artifact, x.loc[va]))
                        squared_sum += m['mse_bps2'] * m['rows']
                        count += m['rows']
                        validations.append(dict(issue=str(issue), side=side, trial=config['id'], offset_months=months,
                            training_rows=len(tr), max_training_label_end=str(y.loc[tr, 'label_end'].max()),
                            fit_warnings=json.dumps(artifact['fit_warnings']), **m))
                        del artifact
                    side_trials.append(dict(trial=config['id'], validation_mse_bps2=squared_sum / count,
                                            validation_rows=count, config=config))
                    pd.DataFrame(validations).to_csv(out / 'validation_trials.csv', index=False)
                ordered = sorted(side_trials, key=lambda r: (r['validation_mse_bps2'], r['trial']))
                rankings.extend(dict(issue=str(issue), side=side, rank=i+1, **r) for i, r in enumerate(ordered))
                bundle = dict(issue=issue, side=side, horizon_hours=48, units='positive ordinary bps', models=[])
                for rank, chosen in enumerate(ordered[:3], 1):
                    tr = train_keys(y, side_mask, issue, chosen['config']['history_days'])
                    artifact = fit(x, target, tr, chosen['config'], args.seed)
                    artifact['max_training_label_end'] = y.loc[tr, 'label_end'].max()
                    bundle['models'].append(artifact)
                    diagnostics.append(dict(issue=str(issue), side=side, rank=rank, trial=chosen['trial'],
                        stage='train_in_sample', **metric(target.loc[tr], predict(artifact, x.loc[tr]))))
                path = out / f'{issue:%Y-%m}_{side}_mean_models.joblib'
                before = predict_bundle(bundle, x.loc[te])
                joblib.dump(bundle, path, compress=3)
                replay = predict_bundle(joblib.load(path), x.loc[te])
                np.testing.assert_allclose(before, replay, rtol=0, atol=1e-6)
                for name in ['best', 'ensemble']:
                    frame[f'{side}_{name}_mean_bps'] = before[f'{name}_mean_bps']
                    frame[f'{side}_{name}_mean_pct'] = before[f'{name}_mean_bps'] / 100.
                baseline_keys = train_keys(y, side_mask, issue, None)
                frame[f'{side}_training_mean_bps'] = target.loc[baseline_keys].mean()
                for name in ['mlp', 'rf', 'reference']:
                    frame[f'{side}_previous_{name}_bps'] = ref.loc[te, f'{side}_{name}_mean_bps']
                audits.append(dict(issue=issue, side=side, model_file=path.name, sha256=sha(path),
                    replay_passed=True, best_trial=ordered[0]['trial'], ensemble_trials=[r['trial'] for r in ordered[:3]],
                    fit_details=[dict(config=a['config'], rows=a['training_rows'], train_first=a['train_first'],
                        train_last=a['train_last'], max_training_label_end=a['max_training_label_end'],
                        warnings=a['fit_warnings']) for a in bundle['models']]))
                dump(out / 'FIT_AUDIT.json', audits)
                dump(out / 'SELECTIONS.json', rankings)
                print(f'DONE {issue.date()} {side}: best={ordered[0]["trial"]}, validation RMSE={np.sqrt(ordered[0]["validation_mse_bps2"]):.3f}', flush=True)
                del bundle
                gc.collect()
            predictions.append(frame)
            frame.to_csv(out / f'{issue:%Y-%m}_magnitude_predictions.csv', index_label='decision_time')
    result = pd.concat(predictions).sort_index()
    check(result.index.equals(dev), 'Final prediction population mismatch')
    result.to_csv(out / 'magnitude_predictions.csv', index_label='decision_time')
    metrics, calibration = [], []
    names = ['best_mean', 'ensemble_mean', 'previous_mlp', 'previous_rf', 'previous_reference', 'training_mean']
    for period, f in [('ALL', result)] + [(str(k), g) for k, g in result.groupby('issue')]:
        for side in ['UP', 'DOWN']:
            g = f.loc[f.actual_direction.eq(side)]
            if g.empty:
                continue
            for name in names:
                column = f'{side}_{name}_bps'
                metrics.append(dict(period=period, side=side, model=name, **metric(g.actual_magnitude_bps, g[column])))
                if period != 'ALL' and name in ['best_mean', 'ensemble_mean']:
                    # Descriptive forecast bins, not future policy thresholds.
                    lo, hi = np.quantile(f[column], [1/3, 2/3])
                    labels = np.where(g[column] <= lo, 'low', np.where(g[column] <= hi, 'medium', 'high'))
                    for label in ['low', 'medium', 'high']:
                        h = g.loc[labels == label]
                        if len(h):
                            calibration.append(dict(issue=period, side=side, model=name, group=label,
                                rows=len(h), mean_prediction_bps=float(h[column].mean()),
                                mean_actual_bps=float(h.actual_magnitude_bps.mean())))
    table = pd.DataFrame(metrics)
    table.to_csv(out / 'magnitude_metrics.csv', index=False)
    pd.DataFrame(calibration).to_csv(out / 'calibration_by_fold.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(out / 'training_diagnostics.csv', index=False)
    summary = ('CONDITIONAL MEAN MAGNITUDE TUNING - no signed-return forecasts\n'
        'best_mean and ensemble_mean chosen solely using earlier validation MSE; report both, no test-based winner.\n'
        'previous_reference = earlier UP GRU / DOWN hybrid. Means and errors are ordinary bps.\n'
        'Training-average benchmark is not a selection candidate. Improvement is not guaranteed.\n\n'
        + table[table.period.eq('ALL')].to_string(index=False))
    (out / 'SUMMARY.txt').write_text(summary, encoding='utf-8')
    dump(out / 'COMPLETE.json', dict(status='COMPLETE', rows=len(result), folds=4,
        training_performed=True, signed_return_forecasts=False, model_replay_passed=True,
        source_sha256=sha(Path(__file__)), predictions_sha256=sha(out / 'magnitude_predictions.csv')))
    print(summary)
    print(f'Outputs saved: {out}')


if __name__ == '__main__':
    main()
