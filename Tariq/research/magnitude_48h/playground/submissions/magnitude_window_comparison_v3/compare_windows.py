"""Two-month walk-forward comparison of training histories. No fresh test access."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
import sklearn
import scipy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = HERE.parent / 'magnitude_walkforward_v2'


def design(frame, metadata=None):
    metadata = {} if metadata is None else metadata
    columns = []
    for name in frame:
        values = frame[name]
        if name not in metadata:
            median = float(values.median())
            filled = values.fillna(median)
            metadata[name] = {'median': median, 'center': float(filled.mean()),
                              'scale': float(filled.std(ddof=0)) or 1.0}
        info = metadata[name]
        columns.extend([(values.fillna(info['median']) - info['center']) / info['scale'],
                        values.isna().astype(float)])
    return np.column_stack(columns), metadata


def load(config, labels):
    X = pd.concat([pd.read_parquet(ROOT / f'data/X_{part}.parquet', columns=config['features'])
                   for part in ['train', 'development']]).sort_index()
    columns = ['label_end', 'magnitude_log_bps'] if labels else ['label_end']
    y = pd.concat([pd.read_parquet(ROOT / f'data/y_{part}.parquet', columns=columns)
                   for part in ['train', 'development']]).sort_index()
    assert X.index.equals(y.index) and X.index.is_unique and len(X) == 48096
    assert (y.label_end - y.index == pd.Timedelta(hours=48)).all()
    return X.replace([np.inf, -np.inf], np.nan), y


def splits(config, y):
    for fold, (start, end) in enumerate(config['validation_windows'], 1):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        valid = y.index[(y.index >= start) & (y.index < end)]
        assert len(valid)
        for name, months in config['training_histories'].items():
            mask = y.label_end < start
            if months is not None:
                mask &= y.index >= start - pd.DateOffset(months=months)
            train = y.index[mask]
            assert len(train) and y.loc[train, 'label_end'].max() < valid.min()
            yield fold, name, train, valid


def preflight(config):
    X, y = load(config, False)
    summary, ledgers = [], []
    for fold, name, train, valid in splits(config, y):
        for feature in X:
            assert X.loc[train, feature].count() >= 1000
            assert X.loc[train, feature].nunique() >= 2
        summary.append({'fold': fold, 'history': name, 'training_rows': len(train),
                        'training_first': str(train.min()), 'training_last': str(train.max()),
                        'latest_training_label_end': str(y.loc[train, 'label_end'].max()),
                        'validation_rows': len(valid), 'validation_first': str(valid.min()),
                        'validation_last': str(valid.max())})
        for role, index in [('train', train), ('validation', valid)]:
            frame = y.loc[index].copy()
            frame['fold'], frame['history'], frame['role'] = fold, name, role
            ledgers.append(frame)
    for name in config['training_histories']:
        keys = pd.DatetimeIndex([t for _, n, _, valid in splits(config, y) if n == name for t in valid])
        assert keys.equals(y.loc['2024-01-01':].index) and len(keys) == 18481
    # Only exact matching previously fitted training populations may be reused.
    for fold_str, old_fold in config['cached_expanding_models'].items():
        start = pd.Timestamp(config['validation_windows'][int(fold_str)-1][0])
        previous_config = json.loads((PREVIOUS / 'run_config.json').read_text())
        assert start == pd.Timestamp(previous_config['validation_windows'][old_fold-1][0])
        assert config['features'] == previous_config['features']['pair']
        assert json.loads((PREVIOUS / 'run_001/verification.json').read_text())['status'] == 'PASS'
    out = HERE / 'preflight'
    out.mkdir(exist_ok=False)
    pd.DataFrame(summary).to_csv(out / 'fold_summary.csv', index=False)
    pd.concat(ledgers).to_csv(out / 'fold_ledger.csv', index_label='decision_time')
    paths = [ROOT / f'data/{kind}_{part}.parquet' for kind in ['X', 'y'] for part in ['train', 'development']]
    paths += [HERE / 'config.json']
    paths += [PREVIOUS / f'run_001/fold_{n}_pair.json' for n in config['cached_expanding_models'].values()]
    receipt = {'status': 'PASS', 'new_fits': 0, 'target_values_loaded': False,
               'hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
               'folds': 7, 'histories': 3, 'rows_per_history': 18481, 'planned_new_fits': 19}
    (out / 'verification.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


def run(config):
    receipt = json.loads((HERE / 'preflight/verification.json').read_text())
    for path, digest in receipt['hashes'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    out = HERE / 'run_001'
    out.mkdir(exist_ok=False)
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    (out / 'environment.json').write_text(json.dumps({'python': platform.python_version(),
        'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__,
        'scipy': scipy.__version__}, indent=2))
    X, y = load(config, True)
    ledger, pieces = [], []
    for fold, name, train, valid in splits(config, y):
        cache = config['cached_expanding_models'].get(str(fold)) if name == 'expanding' else None
        A, metadata = design(X.loc[train])
        if cache is not None:
            card = json.loads((PREVIOUS / f'run_001/fold_{cache}_pair.json').read_text())
            assert card['preprocessing'] == metadata
        else:
            assert len(ledger) < config['new_fit_budget']
            entry = {'fit_number': len(ledger)+1, 'fold': fold, 'history': name,
                     'status': 'started', 'time_utc': pd.Timestamp.now(tz='UTC').isoformat()}
            ledger.append(entry)
            (out / 'fit_ledger.json').write_text(json.dumps(ledger, indent=2))
            try:
                model = QuantileRegressor(quantile=0.5, alpha=0, solver='highs').fit(
                    A, y.loc[train, 'magnitude_log_bps'].to_numpy())
                card = {'intercept': float(model.intercept_), 'coefficients': model.coef_.tolist(),
                        'preprocessing': metadata}
                entry['status'] = 'completed'
            except Exception as exc:
                entry.update(status='failed', error=repr(exc))
                raise
            finally:
                (out / 'fit_ledger.json').write_text(json.dumps(ledger, indent=2))
        card.update(fold=fold, history=name, training_rows=len(train),
                    training_label_end_max=str(y.loc[train, 'label_end'].max()), reused=cache is not None)
        path = out / f'fold_{fold}_{name}.json'
        path.write_text(json.dumps(card, indent=2))
        B, _ = design(X.loc[valid], card['preprocessing'])
        pred = np.maximum(card['intercept'] + B @ card['coefficients'], 0)
        assert np.isfinite(pred).all()
        frame = pd.DataFrame({'fold': fold, 'history': name, 'actual': y.loc[valid, 'magnitude_log_bps'],
                              'prediction': pred}, index=valid)
        frame.to_parquet(out / f'fold_{fold}_{name}.parquet')
        pieces.append(frame)
        print(f'Fold {fold}/7 {name}: completed ({len(ledger)} new fits)', flush=True)
    assert len(ledger) == 19
    long = pd.concat(pieces)
    predictions = long.pivot(columns='history', values='prediction').sort_index()
    truth = y.loc[predictions.index, 'magnitude_log_bps']
    frozen = pd.read_parquet(ROOT / 'benchmarks/development_predictions.parquet')
    prior = pd.read_parquet(PREVIOUS / 'run_001/oof_predictions.parquet')
    assert predictions.index.equals(frozen.index) and predictions.index.equals(prior.index)
    predictions['frozen_MP0004'] = frozen.magnitude_log_bps
    predictions['expanding_3month'] = prior.refitted_pair
    losses = predictions.sub(truth, axis=0).abs()
    monthly = losses.groupby(losses.index.strftime('%Y-%m')).mean()
    monthly.to_csv(out / 'monthly_MAE.csv')
    losses.groupby(long[long.history == 'expanding'].fold).mean().to_csv(out / 'fold_MAE.csv')
    predictions.assign(actual=truth).to_parquet(out / 'predictions.parquet')
    intervals = []
    for name in config['training_histories']:
        for reference in ['frozen_MP0004', 'expanding_3month', 'expanding']:
            if name == reference:
                continue
            delta = losses[name] - losses[reference]
            for days in [7, 14]:
                group = ((delta.index - delta.index.min()).total_seconds() // (days*86400)).astype(int)
                sums, sizes = delta.groupby(group).sum().to_numpy(), delta.groupby(group).size().to_numpy()
                draws = np.random.default_rng(48+days).integers(0, len(sizes), (2000, len(sizes)))
                sample = sums[draws].sum(1) / sizes[draws].sum(1)
                lo, hi = np.quantile(sample, [.025, .975])
                intervals.append({'candidate': name, 'reference': reference, 'block_days': days,
                                  'mean_delta': float(delta.mean()), 'low': lo, 'high': hi})
    overall = losses.mean()
    winner = min(config['training_histories'], key=lambda name: overall[name])
    report = {'pooled_MAE': overall.to_dict(), 'best_new_schedule': winner,
              'new_fits': 19, 'cached_fits_reused': 2, 'total_session_fits': 39,
              'nominal_paired_intervals': intervals,
              'months_better_frozen': {n: int((monthly[n] < monthly.frozen_MP0004-1e-10).sum())
                                       for n in config['training_histories']},
              'interpretation': 'Exposed development comparison; negative deltas favor candidate. Intervals do not correct adaptive search. All candidates score identical rows; fresh test not accessed.'}
    (out / 'RESULT.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'pooled_MAE': report['pooled_MAE'], 'new_fits': 19}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['preflight', 'run'])
    args = parser.parse_args()
    config = json.loads((HERE / 'config.json').read_text())
    {'preflight': preflight, 'run': run}[args.command](config)
