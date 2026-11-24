"""Independent saved-coefficient replay, chronology audit and MAE verification."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'run_001'
config = json.loads((OUT / 'config.json').read_text())
features = config['features']
X = pd.concat([pd.read_parquet(ROOT / f'data/X_{s}.parquet', columns=features)
               for s in ['train', 'development']]).sort_index().replace([np.inf, -np.inf], np.nan)
y = pd.concat([pd.read_parquet(ROOT / f'data/y_{s}.parquet', columns=['label_end', 'magnitude_log_bps'])
               for s in ['train', 'development']]).sort_index()
predictions = pd.read_parquet(OUT / 'predictions.parquet')
assert predictions.index.is_unique and predictions.index.equals(y.loc['2024-01-01':].index)
np.testing.assert_array_equal(predictions.actual, y.loc[predictions.index, 'magnitude_log_bps'])
max_error = 0.0
for fold, (start, end) in enumerate(config['validation_windows'], 1):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    valid = y.index[(y.index >= start) & (y.index < end)]
    for name, months in config['training_histories'].items():
        mask = y.label_end < start
        if months is not None:
            mask &= y.index >= start - pd.DateOffset(months=months)
        train = y.index[mask]
        assert y.loc[train, 'label_end'].max() < valid.min()
        card = json.loads((OUT / f'fold_{fold}_{name}.json').read_text())
        assert card['training_rows'] == len(train)
        A = []
        for feature in features:
            a = card['preprocessing'][feature]
            train_values = X.loc[train, feature]
            assert a['median'] == train_values.median()
            filled = train_values.fillna(a['median'])
            assert abs(a['center'] - filled.mean()) < 1e-12
            assert abs(a['scale'] - filled.std(ddof=0)) < 1e-12
            v = X.loc[valid, feature]
            A.extend([(v.fillna(a['median']) - a['center']) / a['scale'], v.isna().astype(float)])
        replay = np.maximum(card['intercept'] + np.column_stack(A) @ card['coefficients'], 0)
        error = float(np.max(np.abs(replay - predictions.loc[valid, name])))
        max_error = max(max_error, error)
        assert error < 1e-10
report = json.loads((OUT / 'RESULT.json').read_text())
for name, mae in report['pooled_MAE'].items():
    assert abs(np.abs(predictions.actual - predictions[name]).mean() - mae) < 1e-10
ledger = json.loads((OUT / 'fit_ledger.json').read_text())
assert len(ledger) == 19 and all(row['status'] == 'completed' for row in ledger)
receipt = {'status': 'PASS', 'rows': len(predictions), 'model_window_combinations': 21,
           'new_fits': 19, 'reused_fits': 2, 'max_replay_error': max_error,
           'checks': ['identical evaluation keys and targets', 'strict label-end purge',
                      'rolling calendar cutoffs', 'train-only transformations',
                      'all coefficient replays', 'independent MAE recomputation'],
           'verification_fits': 0, 'fresh_data_accessed': False}
with (OUT / 'verification.json').open('x') as f:
    json.dump(receipt, f, indent=2)
print(json.dumps(receipt, indent=2))
