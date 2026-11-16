"""Small economic feature groups, fixed fast tree, earlier-only shrinkage."""
from itertools import product
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import discover_excursion_features_72h as b

CONFIG=dict(max_depth=3,max_leaf_nodes=7,min_samples_leaf=128,max_iter=60,
    learning_rate=.05,l2_regularization=10.,max_bins=63)
PACKS=['leverage','flow_liquidation','combined']
WEIGHTS=[.1,.25,.5,1.]
PRICE=['LONG__vol_per_sqrt_hour__72h','LONG__normalized_momentum__72h','LONG__range_position__72h',
    'LONG__up_variance_share__72h','LONG__drawdown_vol_units__168h','LONG__jump_variance_share__336h']


def pack_columns(x,pack):
    leverage=['funding_level','funding_change','basis_level','basis_change','oi_position','oi_unwind']
    flow=['flow_pressure','flow_activity','liq_history','peer_liq_history','liq_breadth']
    families=leverage if pack=='leverage' else flow if pack=='flow_liquidation' else leverage+flow
    interactions=['funding_oi','oi_lowflow','unwind_negative_momentum','basis_funding'] if pack=='leverage' else ['liq_volatility','flow_momentum'] if pack=='flow_liquidation' else ['funding_oi','oi_lowflow','unwind_negative_momentum','basis_funding','liq_volatility','flow_momentum']
    names=PRICE+[f'STATE__{family}_{stat}__{h}h' for family,stat,h in product(families,['mean','deviation'],[24,72])]
    names += [f'INTERACT__{name}__{h}h' for name,h in product(interactions,[24,72])]
    b.check(set(names)<=set(x),'Required focused columns missing')
    return names


def fit(x,y,spec):
    median=float(np.median(y))
    if spec['model']=='median':return dict(constant=median,features=[])
    names=pack_columns(x,spec['pack'])
    quality=[dict(feature=c,coverage=float(np.isfinite(x[c]).mean()),count=int(np.isfinite(x[c]).sum()),
        eligible=bool(np.isfinite(x[c]).mean()>=.8 and np.isfinite(x[c]).sum()>=1000 and x[c].nunique()>1)) for c in names]
    cols=[r['feature'] for r in quality if r['eligible']]
    if not cols:return dict(constant=median,features=[],quality=quality)
    a=x[cols].to_numpy(float);prep=b.transform_fit(a)
    rows=np.unique(np.linspace(0,len(y)-1,min(6000,len(y)),dtype=int))
    estimator=HistGradientBoostingRegressor(loss='absolute_error',early_stopping=False,random_state=48,**CONFIG)
    estimator.fit(b.transform(a[rows],prep),np.asarray(y)[rows])
    return dict(features=cols,quality=quality,prep=prep,model=estimator,median=median,
        estimator_training_rows=len(rows),available_training_rows=len(y))


def predict(model,x,weight=1.):
    if 'constant' in model:return np.full(len(x),model['constant'])
    a=x[model['features']].to_numpy(float)
    effective=np.where(np.isfinite(a).mean(axis=1)>=.8,weight,0.)
    return effective*b.predict(model,a)+(1-effective)*model['median']


def search(x,y,folds,folder):
    counts=[len(va) for _,va in folds]
    baseline=np.array([np.abs(y[va]-np.median(y[tr])).mean() for tr,va in folds])
    fallback=dict(model='median',validation_mae=float(np.average(baseline,weights=counts)))
    selected=fallback;candidate=None;trials=[];audits=[]
    for pack in PACKS:
        predictions=[];usable=True
        for i,(tr,va) in enumerate(folds):
            model=fit(x.iloc[tr],y[tr],dict(model='focused_hgb',pack=pack))
            usable &= bool(model['features'])
            predictions.append(predict(model,x.iloc[va]))
            audits.append(dict(pack=pack,window=i+1,features=model['features'],quality=model.get('quality',[])))
        for weight in WEIGHTS:
            errors=np.array([np.abs(y[va]-(weight*p+(1-weight)*np.median(y[tr]))).mean() for (tr,va),p in zip(folds,predictions)])
            score=float(np.average(errors,weights=counts))
            eligible=bool(usable and score<fallback['validation_mae']*.999 and np.sum(errors<=baseline+1e-10)>=3 and np.all(errors<=baseline*1.02+1e-10))
            row=dict(model='focused_hgb',pack=pack,weight=weight,validation_mae=score,window_mae=errors.tolist(),eligible=eligible)
            trials.append(row)
            if usable and (candidate is None or score<candidate['validation_mae']):candidate=row
            if eligible and score<selected['validation_mae']:selected=row
        print('  focused '+pack+' complete',flush=True)
    result=dict(selected=selected,candidate=candidate or fallback,baseline_window_mae=baseline.tolist())
    b.dump(folder/'validation_trials.json',trials);b.dump(folder/'quality.json',audits);b.dump(folder/'selection.json',result)
    return result
