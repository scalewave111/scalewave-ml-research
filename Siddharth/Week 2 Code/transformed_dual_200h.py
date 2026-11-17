"""Training-only nonlinear input transforms and joint MAE/RMSE200h selection."""
import numpy as np
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import discover_excursion_features_200h as b
import focused_market_trees_200h as market
import dual_objective_200h as prior
import magnitude_models_200h as original

CONFIGS = {
    'linear_raw': dict(family='linear', transform='raw'),
    'linear_signed_log': dict(family='linear', transform='signed_log'),
    'linear_rank': dict(family='linear', transform='rank'),
    'linear_quadratic': dict(family='linear', transform='quadratic'),
    'hgb_quadratic': dict(family='hgb', transform='quadratic'),
}
MEAN_WEIGHTS = [0., .25, .5, .75, 1.]
UPDATE_WEIGHTS = [.25, .5, .75, 1.]
LONG = ['vol_per_sqrt_hour__200h', 'vol_per_sqrt_hour__720h',
        'jump_variance_share__336h', 'net__168h', 'range__1440h',
        'up_variance_share__200h', 'drift_change_200__400h',
        'drift_change_336__1440h', 'range__400h', 'up_variance_share__1440h',
        'normalized_momentum__200h', 'range_position__200h']
LONG = ['LONG__'+c for c in LONG]


def packs(x, anchor):
    return dict(anchor=list(anchor), long_price=LONG,
                long_market=list(dict.fromkeys(LONG+market.pack_columns(x, 'combined'))))


def fit_transform(a, mode):
    a=np.asarray(a,float)
    finite=np.where(np.isfinite(a),a,np.nan)
    median=np.nanmedian(finite,axis=0)
    q=np.nanquantile(finite,[.25,.75],axis=0)
    scale=np.where(q[1]-q[0]>1e-8,q[1]-q[0],1.)
    prep=dict(mode=mode,median=median,scale=scale,
        references=[np.sort(a[np.isfinite(a[:,j]),j]) for j in range(a.shape[1])])
    z=basis(a,prep)
    prep['center']=z.mean(axis=0)
    spread=z.std(axis=0)
    prep['spread']=np.where(spread>1e-8,spread,1.)
    return prep


def basis(a, prep):
    a=np.asarray(a,float);missing=~np.isfinite(a)
    values=np.where(missing,prep['median'],a)
    if prep['mode']=='rank':
        columns=[]
        for j,reference in enumerate(prep['references']):
            percentile=(np.searchsorted(reference,values[:,j],side='left')+
                        np.searchsorted(reference,values[:,j],side='right'))/(2.*len(reference))
            columns.append(2*percentile-1)
        z=np.column_stack(columns)
    else:
        z=(values-prep['median'])/prep['scale']
        if prep['mode'] in ['signed_log','quadratic']:
            z=np.sign(z)*np.log1p(np.abs(z))
        z=np.clip(z,-8,8)
        if prep['mode']=='quadratic':
            z=np.column_stack([z,z*z])
    return np.column_stack([z,missing.astype(float)])


def transform(a,prep):
    return np.clip((basis(a,prep)-prep['center'])/prep['spread'],-8,8)


def fit_pair(x, y, columns, config):
    quality=[dict(feature=c,coverage=float(np.isfinite(x[c]).mean()),
        eligible=bool(np.isfinite(x[c]).mean()>=.8 and np.isfinite(x[c]).sum()>=1000
                      and x[c].nunique()>1)) for c in columns]
    cols=[q['feature'] for q in quality if q['eligible']]
    cfg=CONFIGS[config]
    result=dict(features=cols,quality=quality,config=cfg)
    if not cols:return result
    a=x[cols].to_numpy(float)
    prep=fit_transform(a,cfg['transform']);z=transform(a,prep)
    unit=max(float(np.median(y)),1.)
    result.update(prep=prep,unit=unit,training_rows=len(y),basis_columns=z.shape[1])
    target=np.asarray(y)/unit
    for name,loss in [('median','epsilon_insensitive'),('mean','squared_error')]:
        if cfg['family']=='linear':
            if name=='mean':
                estimator=Ridge(alpha=.01*len(y))
            else:
                estimator=SGDRegressor(loss=loss,epsilon=0.,penalty='elasticnet',
                    alpha=.001,l1_ratio=.1,learning_rate='adaptive',eta0=.005,
                    max_iter=1500,tol=1e-5,n_iter_no_change=15,average=True,
                    early_stopping=False,random_state=48)
        else:
            estimator=HistGradientBoostingRegressor(loss='absolute_error' if name=='median' else 'squared_error',
                max_depth=2,max_leaf_nodes=4,min_samples_leaf=256,max_iter=60,
                learning_rate=.05,l2_regularization=10.,max_bins=63,
                early_stopping=False,random_state=48)
        estimator.fit(z,target)
        result[name]=estimator
    return result


def components(pair,x,anchor_prediction):
    if not pair['features']:return anchor_prediction,anchor_prediction
    a=x[pair['features']].to_numpy(float)
    usable=np.isfinite(a).mean(axis=1)>=.8
    z=transform(a,pair['prep'])
    return tuple(np.where(usable,np.maximum(0.,pair[k].predict(z)*pair['unit']),anchor_prediction)
                 for k in ['median','mean'])


def blend(anchor, median, mean, mean_weight, update_weight):
    return (1-update_weight)*anchor + update_weight*((1-mean_weight)*median+mean_weight*mean)


def errors(y, prediction):
    return np.array([np.abs(y-prediction).mean(), np.square(y-prediction).mean()])


def assess(err, baseline, anchor, counts):
    pooled = np.average(err, axis=0, weights=counts)
    base = np.average(baseline, axis=0, weights=counts)
    old = np.average(anchor, axis=0, weights=counts)
    # MAE uses the smaller mean/median error in each window; MSE uses mean error.
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
                mae_ratio_to_stronger_baseline=float(ratio[0]), rmse_ratio_to_mean=float(np.sqrt(ratio[1])),
                objective=rank, eligible=eligible)


def anchor_predict(anchor, x):
    if anchor.get('kind') == 'dual':
        return prior.predict(anchor['model'], x, 'candidate')
    if anchor.get('kind', 'tree') == 'tree':
        return b.predict(anchor['model'], x[anchor['features']].to_numpy(float))
    return original.predict(anchor['model'], x, anchor['spec'].get('weight', 1.))


def anchor_refit(anchor, x, y):
    if anchor.get('kind') == 'dual':
        saved = anchor['model']
        spec = saved['plan']['candidate']
        bundle = dict(saved)
        bundle['anchor'] = prior.anchor_refit(saved['anchor'], x, y)
        if spec['model'] == 'dual':
            bundle['candidate'] = prior.fit_pair(x, y, spec['features'], spec['config'])
        return dict(kind='dual', features=anchor['features'], model=bundle)
    if anchor.get('kind', 'tree') == 'tree':
        return dict(kind='tree', features=anchor['features'],
                    model=b.fit(x[anchor['features']].to_numpy(float), y, 'tree'))
    return dict(kind='neural', features=anchor['features'], spec=anchor['spec'],
                model=original.fit(x, y, anchor['spec']))


def search(x, y, folds, anchor_spec, folder):
    anchor_columns = anchor_spec['features']
    anchor_predictions, baseline, anchor_errors, counts = [], [], [], []
    for tr, va in folds:
        model = anchor_refit(anchor_spec, x.iloc[tr], y[tr])
        p = anchor_predict(model, x.iloc[va])
        anchor_predictions.append(p)
        baseline.append([min(np.abs(y[va]-np.median(y[tr])).mean(), np.abs(y[va]-np.mean(y[tr])).mean()),
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
    anchor = anchor_predict(bundle['anchor'], x)
    spec = bundle['plan'][name]
    if spec['model'] == 'anchor':
        return anchor
    median, mean = components(bundle[name], x, anchor)
    return blend(anchor, median, mean, spec['mean_weight'], spec['update_weight'])
