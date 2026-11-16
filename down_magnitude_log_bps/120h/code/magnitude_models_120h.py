"""Independent 120h feature-pack/model selection using only earlier outcomes."""
import numpy as np
import pandas as pd
import focused_market_trees_120h as tree
import neural_market_120h as neural

b = tree.b
PACKS = tree.PACKS
WEIGHTS = tree.WEIGHTS
CONFIG = dict(hgb=dict(kind='hgb', **tree.CONFIG), **neural.CONFIG)
CAP = neural.CAP
EPOCHS = neural.EPOCHS
LAGS = neural.LAGS
pack_columns = tree.pack_columns
augment = neural.augment


def windows(index, issue):
    result = []
    for offset in [12, 9, 6, 3]:
        start = issue - pd.DateOffset(months=offset)
        end = start + pd.DateOffset(months=3)
        tr = np.flatnonzero(index + pd.Timedelta(hours=120) < start)
        va = np.flatnonzero((index >= start) & (index + pd.Timedelta(hours=120) < end))
        b.check(len(tr) >= 1000 and len(va) >= 100, 'Insufficient history')
        result.append((tr, va))
    return result


def fit(x, y, spec):
    if spec['model'] == 'median':
        return dict(constant=float(np.median(y)), features=[])
    backend = tree if spec['config'] == 'hgb' else neural
    model = backend.fit(x, y, spec)
    model['backend'] = 'hgb' if backend is tree else 'neural'
    return model


def predict(model, x, weight=1.):
    if 'constant' in model:
        return np.full(len(x), model['constant'])
    backend = tree if model['backend'] == 'hgb' else neural
    return backend.predict(model, x, weight)


def search(x, y, folds, folder):
    counts = [len(va) for _, va in folds]
    baseline = np.array([np.abs(y[va]-np.median(y[tr])).mean() for tr, va in folds])
    fallback = dict(model='median', validation_mae=float(np.average(baseline, weights=counts)))
    selected, candidate, trials, audits = fallback, None, [], []
    for pack in PACKS:
        for config in CONFIG:
            forecasts, usable = [], True
            for i, (tr, va) in enumerate(folds):
                model = fit(x.iloc[tr], y[tr], dict(model='learned', pack=pack, config=config))
                usable &= bool(model['features'])
                forecasts.append(predict(model, x.iloc[va]))
                audits.append(dict(pack=pack, config=config, window=i+1,
                                   features=model['features'], quality=model.get('quality', [])))
            for weight in WEIGHTS:
                errors = np.array([np.abs(y[va]-(weight*p+(1-weight)*np.median(y[tr]))).mean()
                                   for (tr, va), p in zip(folds, forecasts)])
                score = float(np.average(errors, weights=counts))
                eligible = bool(usable and score < fallback['validation_mae']*.999
                                and (errors <= baseline+1e-10).sum() >= 3
                                and (errors <= baseline*1.02+1e-10).all())
                row = dict(model='learned', pack=pack, config=config, weight=weight,
                           validation_mae=score, window_mae=errors.tolist(), eligible=eligible)
                trials.append(row)
                if usable and (candidate is None or score < candidate['validation_mae']):
                    candidate = row
                if eligible and score < selected['validation_mae']:
                    selected = row
            print(f'  {pack} / {config} complete', flush=True)
    plan = dict(selected=selected, candidate=candidate or fallback, baseline_window_mae=baseline.tolist())
    b.dump(folder/'validation_trials.json', trials)
    b.dump(folder/'quality.json', audits)
    b.dump(folder/'selection.json', plan)
    return plan
