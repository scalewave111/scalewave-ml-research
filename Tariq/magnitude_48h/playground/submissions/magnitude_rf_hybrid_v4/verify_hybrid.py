"""Replay locally generated forests and independently audit nested weight selection."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'run_001'
config = json.loads((OUT / 'config.json').read_text())
X = pd.concat([pd.read_parquet(ROOT / f'data/X_{s}.parquet', columns=config['features'])
               for s in ['train', 'development']]).sort_index().replace([np.inf, -np.inf], np.nan)
y = pd.concat([pd.read_parquet(ROOT / f'data/y_{s}.parquet', columns=['label_end', 'magnitude_log_bps'])
               for s in ['train', 'development']]).sort_index()
predictions = pd.read_parquet(OUT / 'predictions.parquet')
prior = pd.read_parquet(HERE.parent / 'magnitude_walkforward_v2/run_001/oof_predictions.parquet')
assert predictions.index.equals(prior.index) and predictions.index.is_unique
np.testing.assert_array_equal(predictions.actual, y.loc[predictions.index, 'magnitude_log_bps'])
np.testing.assert_array_equal(predictions.refitted_pair, prior.refitted_pair)


def transform(train, valid, metadata):
    columns = []
    for name in config['features']:
        a = metadata[name]
        v = X.loc[train, name]
        assert a['median'] == v.median()
        filled = v.fillna(a['median'])
        assert abs(a['center']-filled.mean()) < 1e-12
        assert abs(a['scale']-filled.std(ddof=0)) < 1e-12
        v = X.loc[valid, name]
        columns.extend([(v.fillna(a['median'])-a['center'])/a['scale'], v.isna().astype(float)])
    return np.column_stack(columns)


for fold, (start, end) in enumerate(config['validation_windows'], 1):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    inner_start = start-pd.DateOffset(months=3)
    ot = y.index[y.label_end < start]
    ov = y.index[(y.index >= start) & (y.index < end)]
    it = y.index[y.label_end < inner_start]
    iv = ot[ot >= inner_start]
    assert y.loc[it, 'label_end'].max() < iv.min()
    assert y.loc[iv, 'label_end'].max() < ov.min()
    inner = json.loads((OUT / f'fold_{fold}_inner.json').read_text())
    A = transform(it, iv, inner['preprocessing'])
    pair = np.maximum(inner['intercept']+A@inner['coefficients'], 0)
    rf = joblib.load(OUT / f'fold_{fold}_inner_forest.joblib')
    forest = np.maximum(rf.predict(A), 0)
    saved = pd.read_parquet(OUT / f'fold_{fold}_inner_predictions.parquet')
    assert saved.index.equals(iv)
    np.testing.assert_allclose(pair, saved.pair, atol=1e-10, rtol=0)
    np.testing.assert_allclose(forest, saved.forest, atol=1e-10, rtol=0)
    np.testing.assert_array_equal(saved.actual, y.loc[iv, 'magnitude_log_bps'])
    scores = [(float(np.abs(saved.actual-((1-w)*pair+w*forest)).mean()), w)
              for w in config['forest_weight_candidates']]
    chosen = min(scores)[1]
    assert chosen == inner['chosen_weight']
    outer = json.loads((OUT / f'fold_{fold}_outer.json').read_text())
    assert outer['weight'] == chosen and outer['training_rows'] == len(ot)
    B = transform(ot, ov, outer['preprocessing'])
    rf = joblib.load(OUT / f'fold_{fold}_outer_forest.joblib')
    forest = np.maximum(rf.predict(B), 0)
    # Independently reconstruct forest output from its saved individual trees.
    tree_mean = np.mean([tree.predict(B) for tree in rf.estimators_], axis=0)
    np.testing.assert_allclose(forest, tree_mean, atol=1e-10, rtol=0)
    np.testing.assert_allclose(forest, predictions.loc[ov, 'forest_only'], atol=1e-10, rtol=0)
    pair = predictions.loc[ov, 'refitted_pair'].to_numpy()
    for column, weight in [('selected_blend', chosen), ('fixed_blend', config['fixed_forest_weight'])]:
        np.testing.assert_allclose((1-weight)*pair+weight*forest, predictions.loc[ov, column], atol=1e-10, rtol=0)
report = json.loads((OUT / 'RESULT.json').read_text())
for name, mae in report['pooled_MAE'].items():
    assert abs(np.abs(predictions.actual-predictions[name]).mean()-mae) < 1e-10
ledger = json.loads((OUT / 'fit_ledger.json').read_text())
assert len(ledger) == 16 and all(x['status']=='completed' for x in ledger)
receipt = {'status':'PASS', 'rows':len(predictions), 'new_verification_fits':0,
    'checks':['exact outer rows and targets', 'inner and outer label purges', 'train-only transformations',
              'inner forest and pair replay', 'training-only blend weight selection',
              'outer tree-level forest replay', 'both hybrid formulas', 'independent MAE recomputation'],
    'fresh_data_accessed':False}
with (OUT / 'verification.json').open('x') as f:
    json.dump(receipt,f,indent=2)
print(json.dumps(receipt,indent=2))
