"""UP/DOWN300h MAE refinement; both baseline MAEs, independent earlier validation."""
import numpy as np
import pandas as pd
import focused_market_trees_300h as tree
import neural_market_300h as neural

b = tree.b
PACKS = tree.PACKS
WEIGHTS = [.1, .25, .5, 1.]
ANCHOR_WEIGHTS = [0., .5, 1.]
BASE_CONFIG = dict(hgb=dict(kind='hgb', **tree.CONFIG), **neural.CONFIG)
CONFIG = {f'{name}_{suffix}': dict(base=name, loss=loss, **cfg)
          for name, cfg in BASE_CONFIG.items()
          for suffix, loss in [('l1', 'absolute_error'), ('l2', 'squared_error')]}
CAP, EPOCHS, LAGS = neural.CAP, neural.EPOCHS, neural.LAGS
pack_columns, augment = tree.pack_columns, neural.augment


def configure(fast=False):
    """Explicit smaller search; keep both losses and all four validation windows."""
    global PACKS, CONFIG, CAP, EPOCHS
    PACKS = ['combined'] if fast else list(tree.PACKS)
    tree.CAP = 3000 if fast else 6000
    tree.CONFIG['max_iter'] = 40 if fast else 60
    neural.CAP = 2048 if fast else 4096
    neural.EPOCHS = 8 if fast else 24
    CAP, EPOCHS = neural.CAP, neural.EPOCHS
    bases = dict(hgb=dict(kind='hgb', **tree.CONFIG), **neural.CONFIG)
    if fast:
        bases = {k: bases[k] for k in ['hgb', 'mlp16', 'gru8']}
    CONFIG = {f'{name}_{suffix}': dict(base=name, loss=loss, **cfg)
              for name, cfg in bases.items()
              for suffix, loss in [('l1', 'absolute_error'), ('l2', 'squared_error')]}


def windows(index, issue):
    result = []
    for offset in [12, 9, 6, 3]:
        start = issue - pd.DateOffset(months=offset)
        end = start + pd.DateOffset(months=3)
        tr = np.flatnonzero(index + pd.Timedelta(hours=300) < start)
        va = np.flatnonzero((index >= start) & (index + pd.Timedelta(hours=300) < end))
        b.check(len(tr) >= 1000 and len(va) >= 100, 'Insufficient history')
        result.append((tr, va))
    return result


def fit(x, y, spec):
    if spec['model'] == 'median':
        return dict(constant=float(np.median(y)), features=[])
    cfg = CONFIG[spec['config']]
    backend = tree if cfg['base'] == 'hgb' else neural
    model = backend.fit(x, y, dict(spec, config=cfg['base'], loss=cfg['loss']))
    model['backend'] = 'hgb' if backend is tree else 'neural'
    model['anchor'] = ((1-spec.get('anchor_weight', 0.))*float(np.median(y))
                       + spec.get('anchor_weight', 0.)*float(np.mean(y)))
    return model


def predict(model, x, weight=1.):
    if 'constant' in model:
        return np.full(len(x), model['constant'])
    backend = tree if model['backend'] == 'hgb' else neural
    p = backend.predict(model, x)
    usable = np.isfinite(x[model['features']].to_numpy(float)).mean(axis=1) >= .8
    return np.where(usable, weight*p+(1-weight)*model['anchor'], model['anchor'])


def errors(y, p):
    return np.array([np.abs(y-p).mean(), np.square(y-p).mean()])


def assess(err, baseline, counts):
    pooled = np.average(err, axis=0, weights=counts)
    base = np.average(baseline, axis=0, weights=counts)[:, 0].min()
    window_best = baseline[:, :, 0].min(axis=1)
    eligible = bool(pooled[0] < base*.999
        and (err[:, 0] <= window_best+1e-10).sum() >= 3
        and (err[:, 0] <= window_best*1.02+1e-10).all())
    return dict(validation_mae=float(pooled[0]), validation_rmse=float(np.sqrt(pooled[1])),
        objective=float(pooled[0]/max(base, 1e-12)), eligible=eligible)


def search(x, y, folds, folder):
    counts = [len(va) for _, va in folds]
    baseline = np.array([[errors(y[va], np.median(y[tr])), errors(y[va], np.mean(y[tr]))]
                         for tr, va in folds])
    fallback = dict(model='median')
    selected, candidate, trials, audits = None, None, [], []
    for pack in PACKS:
        for config in CONFIG:
            forecasts, usable, valid = [], True, []
            for i, (tr, va) in enumerate(folds):
                model = fit(x.iloc[tr], y[tr], dict(model='learned', pack=pack, config=config))
                usable &= bool(model['features'])
                forecasts.append(predict(model, x.iloc[va]))
                valid.append(np.isfinite(x.iloc[va][model['features']]).mean(axis=1).to_numpy() >= .8)
                audits.append(dict(pack=pack, config=config, window=i+1,
                                   features=model['features'], quality=model.get('quality', [])))
            for anchor_weight in ANCHOR_WEIGHTS:
                for weight in WEIGHTS:
                    err = []
                    for (tr, va), p, mask in zip(folds, forecasts, valid):
                        anchor = (1-anchor_weight)*np.median(y[tr])+anchor_weight*np.mean(y[tr])
                        err.append(errors(y[va], np.where(mask, weight*p+(1-weight)*anchor, anchor)))
                    err = np.array(err)
                    row = dict(model='learned', pack=pack, config=config, weight=weight,
                        anchor_weight=anchor_weight, window_mae=err[:, 0].tolist(),
                        window_rmse=np.sqrt(err[:, 1]).tolist(), **assess(err, baseline, counts))
                    row['eligible'] = bool(usable and row['eligible'])
                    trials.append(row)
                    if usable and (candidate is None or row['objective'] < candidate['objective']):
                        candidate = row
                    if row['eligible'] and (selected is None or row['objective'] < selected['objective']):
                        selected = row
            print(f'  {pack} / {config} complete', flush=True)
    plan = dict(selected=selected or fallback, candidate=candidate or fallback,
                accepted=selected is not None, baseline_window_errors=baseline.tolist())
    b.dump(folder/'validation_trials.json', trials)
    b.dump(folder/'quality.json', audits)
    b.dump(folder/'selection.json', plan)
    return plan


def configure_refinement():
    global PACKS, CONFIG, EPOCHS, WEIGHTS
    configure(True)
    PACKS = ['leverage', 'flow_liquidation', 'combined']
    tree.CONFIG.update(max_depth=2, max_leaf_nodes=4, min_samples_leaf=256, max_iter=60)
    neural.EPOCHS = 12
    EPOCHS = 12
    WEIGHTS = [.05, .1, .25, .5, .75, 1.]
    CONFIG = dict(hgb_l1=dict(base='hgb', kind='hgb', loss='absolute_error', **tree.CONFIG),
                  gru16_l1=dict(base='gru16', loss='absolute_error', **neural.CONFIG['gru16']))

configure_refinement()
