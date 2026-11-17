"""Four registered residual-correction fits; reuses already fitted pair controls."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from walkforward import HERE, ROOT, load_data, folds, pair_design

config = json.loads((HERE / 'run_config.json').read_text())
plan = json.loads((HERE / 'residual_config.json').read_text())
original = HERE / 'run_001'
previous = json.loads((original / 'fit_ledger.json').read_text())
assert len(previous) == 16 and all(x['status'] == 'completed' for x in previous)
assert json.loads((original / 'verification.json').read_text())['status'] == 'PASS'
out = HERE / 'run_002_residual'
out.mkdir(exist_ok=False)
(out / 'registered_config.json').write_text(json.dumps(plan, indent=2))
X, y = load_data(config, labels=True)
ledger, pieces = [], []
for number, train, valid in folds(y, config):
    pair = json.loads((original / f'fold_{number}_pair.json').read_text())
    A, _ = pair_design(X.loc[train, config['features']['pair']], pair['preprocessing'])
    train_base = np.maximum(pair['intercept'] + A @ np.asarray(pair['coefficients']), 0)
    frame = pd.read_parquet(original / f'fold_{number}_predictions.parquet')
    assert frame.index.equals(valid)
    residual = y.loc[train, 'magnitude_log_bps'].to_numpy() - train_base
    entry = {'fit_number': 17 + len(ledger), 'fold': number, 'model': 'residual_xgb',
             'status': 'started', 'time_utc': pd.Timestamp.now(tz='UTC').isoformat()}
    ledger.append(entry)
    (out / 'fit_ledger.json').write_text(json.dumps(ledger, indent=2))
    try:
        model = XGBRegressor(**plan['xgb']).fit(X.loc[train], residual)
        model.save_model(out / f'fold_{number}_model.json')
        correction = model.predict(X.loc[valid])
        frame['residual_xgb'] = np.maximum(frame.refitted_pair + correction, 0)
        assert np.isfinite(frame.residual_xgb).all()
        replay = XGBRegressor()
        replay.load_model(out / f'fold_{number}_model.json')
        np.testing.assert_array_equal(replay.predict(X.loc[valid]), correction)
        frame.to_parquet(out / f'fold_{number}_predictions.parquet')
        pieces.append(frame)
        entry['status'] = 'completed'
    except Exception as exc:
        entry.update(status='failed', error=repr(exc))
        raise
    finally:
        (out / 'fit_ledger.json').write_text(json.dumps(ledger, indent=2))
    print(f'Residual fold {number} completed; total fits {16 + len(ledger)}', flush=True)
result = pd.concat(pieces).sort_index()
assert len(result) == 18481 and result.index.is_unique
frozen = pd.read_parquet(ROOT / 'benchmarks/development_predictions.parquet')
assert frozen.index.equals(result.index)
result['frozen_MP0004'] = frozen.magnitude_log_bps
names = ['training_median', 'frozen_MP0004', 'refitted_pair', 'xgb_pair', 'xgb_pair_variance', 'residual_xgb']
losses = result[names].sub(result.actual, axis=0).abs()
losses.groupby(result.fold).mean().to_csv(out / 'fold_MAE.csv')
monthly = losses.groupby(losses.index.strftime('%Y-%m')).mean()
monthly.to_csv(out / 'monthly_MAE.csv')
result.to_parquet(out / 'all_predictions.parquet')
intervals = []
for comparator in ['frozen_MP0004', 'refitted_pair']:
    delta = losses.residual_xgb - losses[comparator]
    for days in [7, 14]:
        group = ((delta.index - delta.index.min()).total_seconds() // (days * 86400)).astype(int)
        sums, sizes = delta.groupby(group).sum().to_numpy(), delta.groupby(group).size().to_numpy()
        draws = np.random.default_rng(48 + days).integers(0, len(sizes), (2000, len(sizes)))
        samples = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
        low, high = np.quantile(samples, [0.025, 0.975])
        intervals.append({'comparator': comparator, 'block_days': days, 'low': low, 'high': high})
report = {'pooled_MAE': losses.mean().to_dict(), 'new_fits': 4, 'total_fits': 20,
          'months_residual_beats_refitted_pair': int((monthly.residual_xgb < monthly.refitted_pair).sum()),
          'nominal_residual_delta_intervals': intervals,
          'saved_model_replay': 'PASS', 'fresh_test_accessed': False,
          'interpretation': 'Adaptive development follow-up, registered after initial results; not independent confirmation. Negative deltas are improvements. All fits stay within 20 total.'}
(out / 'RESULT.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
