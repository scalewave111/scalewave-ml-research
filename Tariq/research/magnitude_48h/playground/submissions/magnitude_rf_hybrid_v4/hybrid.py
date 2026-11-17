"""Frozen-plan RF blends with nested chronological weight selection."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import QuantileRegressor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = HERE.parent / 'magnitude_walkforward_v2'
sys.path.insert(0, str(PREVIOUS))
from walkforward import pair_design


def load(config, labels=False):
    X = pd.concat([pd.read_parquet(ROOT / f'data/X_{s}.parquet', columns=config['features'])
                   for s in ['train', 'development']]).sort_index().replace([np.inf, -np.inf], np.nan)
    columns = ['label_end', 'magnitude_log_bps'] if labels else ['label_end']
    y = pd.concat([pd.read_parquet(ROOT / f'data/y_{s}.parquet', columns=columns)
                   for s in ['train', 'development']]).sort_index()
    assert X.index.equals(y.index) and X.index.is_unique and len(X) == 48096
    assert (y.label_end - y.index == pd.Timedelta(hours=48)).all()
    return X, y


def splits(config, y):
    for fold, (start, end) in enumerate(config['validation_windows'], 1):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        inner_start = start - pd.DateOffset(months=3)
        outer_train = y.index[y.label_end < start]
        outer_valid = y.index[(y.index >= start) & (y.index < end)]
        inner_train = y.index[y.label_end < inner_start]
        inner_valid = outer_train[outer_train >= inner_start]
        assert len(inner_train) and len(inner_valid) and len(outer_valid)
        assert y.loc[inner_train, 'label_end'].max() < inner_valid.min()
        assert y.loc[inner_valid, 'label_end'].max() < outer_valid.min()
        assert y.loc[outer_train, 'label_end'].max() < outer_valid.min()
        yield fold, outer_train, outer_valid, inner_train, inner_valid


def preflight(config):
    X, y = load(config)
    old = json.loads((PREVIOUS / 'run_config.json').read_text())
    assert config['validation_windows'] == old['validation_windows']
    assert config['features'] == old['features']['pair']
    assert json.loads((PREVIOUS / 'run_001/verification.json').read_text())['status'] == 'PASS'
    ledger, summary, keys = [], [], []
    for fold, ot, ov, it, iv in splits(config, y):
        keys.extend(ov)
        for role, index in [('outer_train', ot), ('outer_validation', ov),
                            ('inner_train', it), ('inner_validation', iv)]:
            frame = y.loc[index].copy()
            frame['fold'], frame['role'] = fold, role
            ledger.append(frame)
            summary.append({'fold': fold, 'role': role, 'rows': len(index),
                            'first': str(index.min()), 'last': str(index.max()),
                            'max_label_end': str(y.loc[index, 'label_end'].max())})
        for index in [ot, it]:
            for f in X:
                assert X.loc[index, f].count() >= 1000 and X.loc[index, f].nunique() >= 2
        _, prep = pair_design(X.loc[ot])
        card = json.loads((PREVIOUS / f'run_001/fold_{fold}_pair.json').read_text())
        assert prep == card['preprocessing']
    assert pd.DatetimeIndex(keys).equals(y.loc['2024-01-01':].index) and len(keys) == 18481
    out = HERE / 'preflight'
    out.mkdir(exist_ok=False)
    pd.concat(ledger).to_csv(out / 'split_ledger.csv', index_label='decision_time')
    pd.DataFrame(summary).to_csv(out / 'split_summary.csv', index=False)
    paths = [ROOT / f'data/{kind}_{s}.parquet' for kind in ['X', 'y'] for s in ['train', 'development']]
    paths += [HERE / 'config.json', PREVIOUS / 'walkforward.py', PREVIOUS / 'run_001/oof_predictions.parquet']
    paths += [PREVIOUS / f'run_001/fold_{fold}_pair.json' for fold in range(1, 5)]
    receipt = {'status': 'PASS', 'new_fits': 0, 'target_values_loaded': False,
               'hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (out / 'verification.json').write_text(json.dumps(receipt, indent=2))
    print('Preflight PASS: four nested purged folds, identical scored rows, reusable pair transforms.', flush=True)


def run(config):
    receipt = json.loads((HERE / 'preflight/verification.json').read_text())
    for p, digest in receipt['hashes'].items():
        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest
    out = HERE / 'run_001'
    out.mkdir(exist_ok=False)
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    (out / 'environment.json').write_text(json.dumps({'python': platform.python_version(),
        'sklearn': sklearn.__version__, 'numpy': np.__version__, 'pandas': pd.__version__}, indent=2))
    X, y = load(config, True)
    prior = pd.read_parquet(PREVIOUS / 'run_001/oof_predictions.parquet')
    ledger, parts = [], []

    def fit(fold, name, fn):
        assert len(ledger) < config['new_fit_budget']
        event = {'fit_number': len(ledger)+1, 'fold': fold, 'name': name,
                 'status': 'started', 'time_utc': pd.Timestamp.now(tz='UTC').isoformat()}
        ledger.append(event)
        path = out / 'fit_ledger.json'
        path.write_text(json.dumps(ledger, indent=2))
        print(f'Fold {fold}: starting {name} (fit {len(ledger)}/16)', flush=True)
        try:
            value = fn()
            event['status'] = 'completed'
            return value
        except Exception as exc:
            event.update(status='failed', error=repr(exc))
            raise
        finally:
            path.write_text(json.dumps(ledger, indent=2))

    for fold, ot, ov, it, iv in splits(config, y):
        A, prep = pair_design(X.loc[it])
        B, _ = pair_design(X.loc[iv], prep)
        target = y.loc[it, 'magnitude_log_bps'].to_numpy()
        pair = fit(fold, 'inner_pair', lambda: QuantileRegressor(
            quantile=.5, alpha=0, solver='highs').fit(A, target))
        pair_pred = np.maximum(pair.predict(B), 0)
        inner_rf = fit(fold, 'inner_forest', lambda: RandomForestRegressor(**config['forest']).fit(A, target))
        forest_pred = np.maximum(inner_rf.predict(B), 0)
        truth = y.loc[iv, 'magnitude_log_bps'].to_numpy()
        scores = [{'weight': w, 'MAE': float(np.abs(truth-((1-w)*pair_pred+w*forest_pred)).mean())}
                  for w in config['forest_weight_candidates']]
        weight = fit(fold, 'weight_selection', lambda: min(scores, key=lambda s: (s['MAE'], s['weight']))['weight'])
        pd.DataFrame({'actual': truth, 'pair': pair_pred, 'forest': forest_pred}, index=iv).to_parquet(
            out / f'fold_{fold}_inner_predictions.parquet')
        joblib.dump(inner_rf, out / f'fold_{fold}_inner_forest.joblib')
        inner_card = {'preprocessing': prep, 'intercept': float(pair.intercept_),
                      'coefficients': pair.coef_.tolist(), 'weight_scores': scores, 'chosen_weight': weight}
        (out / f'fold_{fold}_inner.json').write_text(json.dumps(inner_card, indent=2))
        A, prep = pair_design(X.loc[ot])
        B, _ = pair_design(X.loc[ov], prep)
        forest = fit(fold, 'outer_forest', lambda: RandomForestRegressor(**config['forest']).fit(
            A, y.loc[ot, 'magnitude_log_bps'].to_numpy()))
        forest_pred = np.maximum(forest.predict(B), 0)
        joblib.dump(forest, out / f'fold_{fold}_outer_forest.joblib')
        (out / f'fold_{fold}_outer.json').write_text(json.dumps({'preprocessing': prep,
            'weight': weight, 'training_rows': len(ot)}, indent=2))
        pair_pred = prior.loc[ov, 'refitted_pair'].to_numpy()
        fixed = config['fixed_forest_weight']
        frame = pd.DataFrame({'fold': fold, 'actual': y.loc[ov, 'magnitude_log_bps'],
            'refitted_pair': pair_pred, 'forest_only': forest_pred,
            'fixed_blend': (1-fixed)*pair_pred+fixed*forest_pred,
            'selected_blend': (1-weight)*pair_pred+weight*forest_pred,
            'selected_weight': weight}, index=ov)
        assert np.isfinite(frame.to_numpy()).all()
        frame.to_parquet(out / f'fold_{fold}_predictions.parquet')
        parts.append(frame)
        print(f'Fold {fold} complete; training-selected forest weight={weight:.2f}', flush=True)
    predictions = pd.concat(parts).sort_index()
    assert predictions.index.equals(prior.index) and len(predictions) == 18481
    frozen = pd.read_parquet(ROOT / 'benchmarks/development_predictions.parquet')
    assert predictions.index.equals(frozen.index)
    predictions['frozen_MP0004'] = frozen.magnitude_log_bps
    predictions.to_parquet(out / 'predictions.parquet')
    names = ['frozen_MP0004', 'refitted_pair', 'forest_only', 'fixed_blend', 'selected_blend']
    losses = predictions[names].sub(predictions.actual, axis=0).abs()
    monthly = losses.groupby(losses.index.strftime('%Y-%m')).mean()
    monthly.to_csv(out / 'monthly_MAE.csv')
    losses.groupby(predictions.fold).mean().to_csv(out / 'fold_MAE.csv')
    intervals = []
    for candidate in ['fixed_blend', 'selected_blend']:
        for reference in ['refitted_pair', 'frozen_MP0004']:
            delta = losses[candidate] - losses[reference]
            for days in [7, 14]:
                groups = ((delta.index-delta.index.min()).total_seconds()//(days*86400)).astype(int)
                sums, sizes = delta.groupby(groups).sum().to_numpy(), delta.groupby(groups).size().to_numpy()
                draws = np.random.default_rng(48+days).integers(0, len(sizes), (2000, len(sizes)))
                samples = sums[draws].sum(1)/sizes[draws].sum(1)
                lo, hi = np.quantile(samples, [.025, .975])
                intervals.append({'candidate': candidate, 'reference': reference, 'block_days': days,
                                  'mean_delta': float(delta.mean()), 'low': lo, 'high': hi})
    report = {'pooled_MAE': losses.mean().to_dict(), 'rows': len(predictions),
        'selected_forest_weights': predictions.groupby('fold').selected_weight.first().to_dict(),
        'months_better_refitted_pair': {n: int((monthly[n]<monthly.refitted_pair-1e-10).sum())
                                      for n in ['fixed_blend', 'selected_blend']},
        'nominal_paired_intervals': intervals, 'new_fits_including_weight_selection': len(ledger),
        'prior_fits': 39, 'cumulative_fits_including_weight_selection': 39+len(ledger),
        'interpretation': 'Development comparison only. Forest settings fixed; blend weights selected using earlier inner-validation labels only. Nominal intervals do not correct adaptive project search. Fresh test not accessed.'}
    (out / 'RESULT.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'pooled_MAE': report['pooled_MAE'], 'weights': report['selected_forest_weights']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['preflight', 'run'])
    args = parser.parse_args()
    config = json.loads((HERE / 'config.json').read_text())
    {'preflight': preflight, 'run': run}[args.command](config)
