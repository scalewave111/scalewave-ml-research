"""Independently replay saved forecasts and recompute scores; no fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'run_001'
config = json.loads((OUT / 'run_config.json').read_text())
features = config['features']['pair_variance']
X = pd.concat([pd.read_parquet(ROOT / f'data/X_{part}.parquet', columns=features)
               for part in ['train', 'development']]).sort_index().replace([np.inf, -np.inf], np.nan)
y = pd.concat([pd.read_parquet(ROOT / f'data/y_{part}.parquet',
                              columns=['label_end', 'magnitude_log_bps'])
               for part in ['train', 'development']]).sort_index()
saved = pd.read_parquet(OUT / 'oof_predictions.parquet')
assert saved.index.is_unique and saved.index.equals(y.loc['2024-01-01':].index)
np.testing.assert_array_equal(saved.actual, y.loc[saved.index, 'magnitude_log_bps'])
largest_error = 0.0
for number, (start, end) in enumerate(config['validation_windows'], 1):
    mask = (X.index >= pd.Timestamp(start)) & (X.index < pd.Timestamp(end))
    index = X.index[mask]
    expected_training = y.loc[y.label_end < pd.Timestamp(start)]
    assert expected_training.label_end.max() < index.min()
    pair = json.loads((OUT / f'fold_{number}_pair.json').read_text())
    assert pair['training_median'] == expected_training.magnitude_log_bps.median()
    design = []
    for f in config['features']['pair']:
        a = pair['preprocessing'][f]
        training_values = X.loc[expected_training.index, f]
        assert a['median'] == training_values.median()
        filled = training_values.fillna(a['median'])
        assert abs(a['center'] - filled.mean()) < 1e-12
        assert abs(a['scale'] - filled.std(ddof=0)) < 1e-12
        values = X.loc[index, f]
        design.extend([(values.fillna(a['median']) - a['center']) / a['scale'], values.isna().astype(float)])
    replay = np.maximum(pair['intercept'] + np.column_stack(design) @ pair['coefficients'], 0)
    np.testing.assert_allclose(replay, saved.loc[index, 'refitted_pair'], rtol=0, atol=1e-10)
    for name in ['pair', 'pair_variance']:
        model = xgb.XGBRegressor()
        model.load_model(OUT / f'fold_{number}_xgb_{name}.json')
        replay = np.maximum(model.predict(X.loc[index, config['features'][name]]), 0)
        error = float(np.max(np.abs(replay - saved.loc[index, f'xgb_{name}'])))
        largest_error = max(largest_error, error)
        assert error < 1e-10
report = json.loads((OUT / 'RESULT.json').read_text())
for name, expected in report['pooled_MAE'].items():
    assert abs(np.mean(np.abs(saved.actual - saved[name])) - expected) < 1e-10
ledger = json.loads((OUT / 'fit_ledger.json').read_text())
assert len(ledger) == 16 and all(entry['status'] == 'completed' for entry in ledger)
receipt = {'status': 'PASS', 'prediction_rows': len(saved), 'saved_model_max_replay_error': largest_error,
           'checks': ['exact targets and keys', 'strict label-end purge', 'train-only baseline transforms',
                      'saved model prediction replay', 'independent MAE recomputation', '16 completed fits'],
           'new_fits': 0, 'fresh_data_accessed': False}
with (OUT / 'verification.json').open('x') as f:
    json.dump(receipt, f, indent=2)
print(json.dumps(receipt, indent=2))
