"""Registered magnitude experiment. Default command verifies data without fitting."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read_config():
    return json.loads((HERE / 'run_config.json').read_text())


def load_data(config, labels=False):
    features = config['features']['pair_variance']
    xs, ys = [], []
    for part in ['train', 'development']:
        xs.append(pd.read_parquet(ROOT / f'data/X_{part}.parquet', columns=features))
        columns = ['label_end', 'magnitude_log_bps'] if labels else ['label_end']
        ys.append(pd.read_parquet(ROOT / f'data/y_{part}.parquet', columns=columns))
    X, y = pd.concat(xs).sort_index(), pd.concat(ys).sort_index()
    assert X.index.equals(y.index) and X.index.is_unique
    assert len(X) == 48096 and str(X.index.tz) == 'UTC'
    assert (y.label_end - y.index == pd.Timedelta(hours=48)).all()
    if labels:
        assert np.isfinite(y.magnitude_log_bps).all() and (y.magnitude_log_bps >= 0).all()
    return X.replace([np.inf, -np.inf], np.nan), y


def folds(y, config):
    for number, (start, end) in enumerate(config['validation_windows'], 1):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        train = y.index[y.label_end < start]
        valid = y.index[(y.index >= start) & (y.index < end)]
        assert len(train) and len(valid)
        assert y.loc[train, 'label_end'].max() < valid.min()
        assert not len(train.intersection(valid))
        yield number, train, valid


def preflight(config):
    X, y = load_data(config)
    ledger, summaries, admissions, keys = [], [], [], []
    for number, train, valid in folds(y, config):
        keys.extend(valid)
        purged = y.index[(y.index < valid.min()) & (y.label_end >= valid.min())]
        for role, index in [('train', train), ('validation', valid), ('purged', purged)]:
            frame = y.loc[index].copy()
            frame['fold'], frame['role'] = number, role
            ledger.append(frame)
        summaries.append({'fold': number, 'train_rows': len(train), 'validation_rows': len(valid),
                          'purged_rows_present': len(purged), 'validation_first': str(valid.min()),
                          'validation_last': str(valid.max()),
                          'latest_training_label_end': str(y.loc[train, 'label_end'].max())})
        for feature in X:
            values = X.loc[train, feature]
            count, unique = int(values.count()), int(values.nunique())
            assert count >= 1000 and unique >= 2, f'Fold {number}: inadmissible {feature}'
            admissions.append({'fold': number, 'feature': feature, 'finite_training': count,
                               'unique_training': unique, 'admitted': True})
    expected = y.index[y.index >= pd.Timestamp(config['validation_windows'][0][0])]
    assert pd.DatetimeIndex(keys).equals(expected)
    assert len(keys) == len(set(keys)) == 18481
    assert len(summaries) * 4 + config['reserved_final_fits'] + config['failure_reserve'] <= 20
    receipt = HERE / 'preflight'
    receipt.mkdir(exist_ok=False)
    pd.concat(ledger).to_csv(receipt / 'fold_ledger.csv', index_label='decision_time')
    pd.DataFrame(summaries).to_csv(receipt / 'fold_summary.csv', index=False)
    pd.DataFrame(admissions).to_csv(receipt / 'feature_admission.csv', index=False)
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [ROOT / f'data/{kind}_{part}.parquet'
                        for kind in ['X', 'y'] for part in ['train', 'development']]}
    hashes['run_config.json'] = hashlib.sha256((HERE / 'run_config.json').read_bytes()).hexdigest()
    result = {'status': 'PASS', 'new_fits': 0, 'target_values_loaded': False,
              'fresh_test_accessed': False, 'input_hashes': hashes, 'folds': summaries}
    (receipt / 'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def pair_design(X, metadata=None):
    metadata = {} if metadata is None else metadata
    columns = []
    for f in X:
        v = X[f]
        if f not in metadata:
            filled = v.fillna(v.median())
            metadata[f] = {'median': float(v.median()), 'center': float(filled.mean()),
                           'scale': float(filled.std(ddof=0)) or 1.0}
        a = metadata[f]
        columns.extend([(v.fillna(a['median']).to_numpy() - a['center']) / a['scale'],
                        v.isna().to_numpy(float)])
    return np.column_stack(columns), metadata


def run(config):
    from sklearn.linear_model import QuantileRegressor
    from xgboost import XGBRegressor
    import sklearn, xgboost, scipy, pyarrow

    receipt = json.loads((HERE / 'preflight/verification.json').read_text())
    for name, digest in receipt['input_hashes'].items():
        path = HERE / name if name == 'run_config.json' else ROOT / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f'Changed input: {name}'
    out = HERE / 'run_001'
    out.mkdir(exist_ok=False)  # Never reset a fit ledger by rerunning.
    X, y = load_data(config, labels=True)
    events, predictions = [], []
    environment = {'python': platform.python_version(), **{m.__name__: m.__version__
                   for m in [np, pd, sklearn, xgboost, scipy, pyarrow]}}
    (out / 'environment.json').write_text(json.dumps(environment, indent=2))
    (out / 'run_config.json').write_text(json.dumps(config, indent=2))
    (out / 'runner_sha256.txt').write_text(hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def attempt(fold, name, operation):
        assert len(events) < 16, 'Registered walk-forward fit budget exhausted'
        event = {'fit_number': len(events) + 1, 'fold': fold, 'model': name,
                 'status': 'started', 'time_utc': pd.Timestamp.now(tz='UTC').isoformat()}
        events.append(event)
        ledger = out / 'fit_ledger.json'
        ledger.write_text(json.dumps(events, indent=2))
        try:
            result = operation()
            event['status'] = 'completed'
            return result
        except Exception as exc:
            event.update(status='failed', error=repr(exc))
            raise
        finally:
            ledger.write_text(json.dumps(events, indent=2))

    for number, train, valid in folds(y, config):
        target = y.loc[train, 'magnitude_log_bps'].to_numpy()
        frame = pd.DataFrame({'fold': number, 'actual': y.loc[valid, 'magnitude_log_bps']}, index=valid)
        median = attempt(number, 'training_median', lambda: float(np.median(target)))
        frame['training_median'] = median
        pair = config['features']['pair']
        A, metadata = pair_design(X.loc[train, pair])
        B, _ = pair_design(X.loc[valid, pair], metadata)
        reg = attempt(number, 'refitted_pair', lambda: QuantileRegressor(
            quantile=0.5, alpha=0, solver='highs').fit(A, target))
        frame['refitted_pair'] = np.maximum(reg.predict(B), 0)
        (out / f'fold_{number}_pair.json').write_text(json.dumps({
            'preprocessing': metadata, 'intercept': float(reg.intercept_),
            'coefficients': reg.coef_.tolist(), 'training_median': median}, indent=2))
        for name in ['pair', 'pair_variance']:
            features = config['features'][name]
            model = attempt(number, f'xgb_{name}', lambda: XGBRegressor(**config['xgb']).fit(
                X.loc[train, features], target))
            frame[f'xgb_{name}'] = np.maximum(model.predict(X.loc[valid, features]), 0)
            model.save_model(out / f'fold_{number}_xgb_{name}.json')
        assert np.isfinite(frame.to_numpy()).all()
        frame.to_parquet(out / f'fold_{number}_predictions.parquet')
        predictions.append(frame)
        print(f'Completed fold {number}; {len(events)} fits', flush=True)
    result = pd.concat(predictions).sort_index()
    assert len(result) == 18481 and result.index.is_unique
    result.to_parquet(out / 'oof_predictions.parquet')
    names = ['training_median', 'refitted_pair', 'xgb_pair', 'xgb_pair_variance']
    losses = result[names].sub(result.actual, axis=0).abs()
    overall = losses.mean()
    losses.groupby(result.fold).mean().to_csv(out / 'fold_MAE.csv')
    losses.groupby(losses.index.strftime('%Y-%m')).mean().to_csv(out / 'monthly_MAE.csv')
    deltas = losses[['xgb_pair', 'xgb_pair_variance']].sub(losses.refitted_pair, axis=0)
    intervals = []
    for days in [7, 14]:
        groups = ((deltas.index - deltas.index.min()).total_seconds() // (days * 86400)).astype(int)
        sizes = deltas.groupby(groups).size().to_numpy()
        sums = deltas.groupby(groups).sum()
        draws = np.random.default_rng(48 + days).integers(0, len(sizes), (2000, len(sizes)))
        for name in deltas:
            samples = sums[name].to_numpy()[draws].sum(axis=1) / sizes[draws].sum(axis=1)
            lo, hi = np.quantile(samples, [0.025, 0.975])
            intervals.append({'model': name, 'block_days': days, 'low': lo, 'high': hi})
    winner = min(['xgb_pair', 'xgb_pair_variance'], key=lambda name: (overall[name], names.index(name)))
    report = {'pooled_MAE': overall.to_dict(), 'best_xgb': winner,
              'best_xgb_beats_both_controls': bool(overall[winner] < min(overall.training_median, overall.refitted_pair)),
              'nominal_delta_vs_refitted_pair_intervals': intervals,
              'new_fits': len(events), 'fresh_test_accessed': False,
              'interpretation': 'Selected on exposed walk-forward development. Not independent confirmation; intervals are not search-adjusted. Negative delta is improvement. No final model refit performed.'}
    (out / 'RESULT.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['preflight', 'run'])
    args = parser.parse_args()
    {'preflight': preflight, 'run': run}[args.command](read_config())
