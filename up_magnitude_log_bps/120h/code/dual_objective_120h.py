"""Joint MAE/RMSE refinement of existing 120h magnitude models."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import discover_excursion_features_120h as b
import focused_market_trees_120h as market

CONFIGS = {
    'shallow': dict(max_depth=2, max_leaf_nodes=4, min_samples_leaf=256, max_iter=60),
    'regularized': dict(max_depth=3, max_leaf_nodes=7, min_samples_leaf=128, max_iter=80),
}
MEAN_WEIGHTS = [0., .25, .5, .75, 1.]
UPDATE_WEIGHTS = [.25, .5, .75, 1.]
LONG = ['vol_per_sqrt_hour__120h', 'vol_per_sqrt_hour__720h',
        'jump_variance_share__336h', 'net__168h', 'range__1440h',
        'up_variance_share__120h', 'drift_change_120__240h',
        'drift_change_336__1440h', 'range__240h', 'up_variance_share__1440h',
        'normalized_momentum__120h', 'range_position__120h']
LONG = ['LONG__'+c for c in LONG]


def packs(x, anchor):
    return dict(anchor=list(anchor), long_price=LONG,
                long_market=list(dict.fromkeys(LONG+market.pack_columns(x, 'combined'))))


def fit_pair(x, y, columns, config):
    quality = [dict(feature=c, coverage=float(np.isfinite(x[c]).mean()),
                    eligible=bool(np.isfinite(x[c]).mean() >= .8
                                  and np.isfinite(x[c]).sum() >= 1000
                                  and x[c].nunique() > 1)) for c in columns]
    cols = [q['feature'] for q in quality if q['eligible']]
    result = dict(features=cols, quality=quality)
    if not cols:
        return result
    # Fit both losses on ALL mature rows; do not trim or reweight large outcomes.
    a = x[cols].to_numpy(float)
    result['prep'] = b.transform_fit(a)
    z = b.transform(a, result['prep'])
    for name, loss in [('median', 'absolute_error'), ('mean', 'squared_error')]:
        estimator = HistGradientBoostingRegressor(loss=loss, learning_rate=.05,
            l2_regularization=10., max_bins=63, early_stopping=False, random_state=48,
            **CONFIGS[config])
        estimator.fit(z, y)
        result[name] = estimator
    return result


def components(pair, x, anchor_prediction):
    if not pair['features']:
        return anchor_prediction, anchor_prediction
    a = x[pair['features']].to_numpy(float)
    usable = np.isfinite(a).mean(axis=1) >= .8
    z = b.transform(a, pair['prep'])
    return tuple(np.where(usable, np.maximum(0., pair[k].predict(z)), anchor_prediction)
                 for k in ['median', 'mean'])


def blend(anchor, median, mean, mean_weight, update_weight):
    return (1-update_weight)*anchor + update_weight*((1-mean_weight)*median+mean_weight*mean)


def errors(y, prediction):
    return np.array([np.abs(y-prediction).mean(), np.square(y-prediction).mean()])


def assess(err, baseline, anchor, counts):
    pooled = np.average(err, axis=0, weights=counts)
    base = np.average(baseline, axis=0, weights=counts)
    old = np.average(anchor, axis=0, weights=counts)
    # MAE denominator is median MAE; MSE denominator is mean MSE.
    ratio = pooled/np.maximum(base, 1e-12)
    old_ratio = pooled/np.maximum(old, 1e-12)
    rank = max(float(ratio[0]), float(np.sqrt(ratio[1])),
               float(old_ratio[0]), float(np.sqrt(old_ratio[1])))
    eligible = bool(pooled[0] < base[0]*.999 and pooled[1] < base[1]*.999**2
                    and (pooled <= old+1e-10).all()
                    and (err[:, 0] <= baseline[:, 0]+1e-10).sum() >= 3
                    and (err[:, 1] <= baseline[:, 1]+1e-10).sum() >= 3
                    and (err[:, 0] <= baseline[:, 0]*1.02+1e-10).all()
                    and (err[:, 1] <= baseline[:, 1]*1.02**2+1e-10).all())
    return dict(validation_mae=float(pooled[0]), validation_rmse=float(np.sqrt(pooled[1])),
                mae_ratio_to_median=float(ratio[0]), rmse_ratio_to_mean=float(np.sqrt(ratio[1])),
                objective=rank, eligible=eligible)


def search(x, y, folds, anchor_columns, folder):
    anchor_predictions, baseline, anchor_errors, counts = [], [], [], []
    for tr, va in folds:
        model = b.fit(x.iloc[tr][anchor_columns].to_numpy(float), y[tr], 'tree')
        p = b.predict(model, x.iloc[va][anchor_columns].to_numpy(float))
        anchor_predictions.append(p)
        baseline.append([np.abs(y[va]-np.median(y[tr])).mean(),
                         np.square(y[va]-np.mean(y[tr])).mean()])
        anchor_errors.append(errors(y[va], p)); counts.append(len(va))
    baseline, anchor_errors = np.array(baseline), np.array(anchor_errors)
    fallback = dict(model='anchor', features=anchor_columns)
    selected, candidate, trials, audits = None, None, [], []
    for pack, columns in packs(x, anchor_columns).items():
        for config in CONFIGS:
            predictions, usable = [], True
            for i, (tr, va) in enumerate(folds):
                pair = fit_pair(x.iloc[tr], y[tr], columns, config)
                usable &= bool(pair['features'])
                predictions.append(components(pair, x.iloc[va], anchor_predictions[i]))
                audits.append(dict(pack=pack, config=config, window=i+1, **{
                    k:pair[k] for k in ['features', 'quality']}))
            for mean_weight in MEAN_WEIGHTS:
                for update_weight in UPDATE_WEIGHTS:
                    err = np.array([errors(y[va], blend(old, *pair, mean_weight, update_weight))
                        for (_, va), old, pair in zip(folds, anchor_predictions, predictions)])
                    row = dict(model='dual', pack=pack, config=config, features=columns,
                        mean_weight=mean_weight, update_weight=update_weight,
                        window_mae=err[:, 0].tolist(), window_rmse=np.sqrt(err[:, 1]).tolist(),
                        **assess(err, baseline, anchor_errors, counts))
                    row['eligible'] = bool(usable and row['eligible'])
                    trials.append(row)
                    if usable and (candidate is None or row['objective'] < candidate['objective']):
                        candidate = row
                    if row['eligible'] and (selected is None or row['objective'] < selected['objective']):
                        selected = row
            print(f'  {pack}/{config}: L1 + L2 complete', flush=True)
    plan = dict(selected=selected or fallback, candidate=candidate or fallback,
        accepted=selected is not None, baseline_window_errors=baseline.tolist(),
        anchor_window_errors=anchor_errors.tolist())
    b.dump(folder/'selection.json', plan)
    b.dump(folder/'validation_trials.json', trials)
    b.dump(folder/'quality.json', audits)
    return plan


def predict(bundle, x, name):
    anchor = b.predict(bundle['anchor'], x[bundle['anchor_features']].to_numpy(float))
    spec = bundle['plan'][name]
    if spec['model'] == 'anchor':
        return anchor
    median, mean = components(bundle[name], x, anchor)
    return blend(anchor, median, mean, spec['mean_weight'], spec['update_weight'])
