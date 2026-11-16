"""Joint feature-pack testing; no marginal correlation screen or inherited subset."""
from itertools import product
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor
import discover_excursion_features_72h as b

CONFIGS={
 'hgb_regularized':dict(max_depth=3,max_leaf_nodes=7,min_samples_leaf=128,max_iter=160,learning_rate=.05,l2_regularization=10.),
 'hgb_interactions':dict(max_depth=4,max_leaf_nodes=15,min_samples_leaf=256,max_iter=240,learning_rate=.03,l2_regularization=20.),
 'extra_trees':dict(n_estimators=96,max_depth=8,min_samples_leaf=64,max_features=.7),
}
PACKS=['price_control','market_price','market_all_price']
WEIGHTS=[.5,1.]
FULL_CONFIGS={k:dict(v) for k,v in CONFIGS.items()}
FULL_PACKS=list(PACKS)
TRAIN_ROW_CAP=None


def configure(fast=False):
    global CONFIGS,PACKS,TRAIN_ROW_CAP
    CONFIGS=({'hgb_fast':dict(max_depth=3,max_leaf_nodes=7,min_samples_leaf=64,max_iter=60,
        learning_rate=.05,l2_regularization=10.,max_bins=63)} if fast else {k:dict(v) for k,v in FULL_CONFIGS.items()})
    PACKS=['price_control','market_price'] if fast else list(FULL_PACKS)
    TRAIN_ROW_CAP=6000 if fast else None


def pack_columns(x,pack):
    core=[c for c in x if c.startswith('LONG__') and c.endswith(('__24h','__72h','__168h')) and 'ratio' not in c and 'change' not in c]
    market=[c for c in x if c.startswith(('STATE__','INTERACT__'))]
    price=[c for c in x if c.startswith(('LONG__','FRESH72__'))]
    return core if pack=='price_control' else core+market if pack=='market_price' else price+market


def fit(x,y,spec):
    median=float(np.median(y))
    if spec['model']=='median':return dict(constant=median,features=[])
    cols=pack_columns(x,spec['pack'])
    quality=[dict(feature=c,coverage=float(np.isfinite(x[c]).mean()),count=int(np.isfinite(x[c]).sum()),
        eligible=bool(np.isfinite(x[c]).mean()>=.8 and np.isfinite(x[c]).sum()>=1000 and x[c].nunique()>1)) for c in cols]
    cols=[r['feature'] for r in quality if r['eligible']]
    if not cols:return dict(constant=median,features=[],quality=quality)
    a=x[cols].to_numpy(float);prep=b.transform_fit(a)
    if spec['model']=='extra_trees':
        estimator=ExtraTreesRegressor(criterion='absolute_error',bootstrap=True,max_samples=min(4096,len(y)),
            random_state=48,n_jobs=1,**CONFIGS[spec['model']])
    else:
        estimator=HistGradientBoostingRegressor(loss='absolute_error',early_stopping=False,random_state=48,**CONFIGS[spec['model']])
    rows=np.unique(np.linspace(0,len(y)-1,min(TRAIN_ROW_CAP or len(y),len(y)),dtype=int))
    estimator.fit(b.transform(a[rows],prep),np.asarray(y)[rows])
    return dict(features=cols,quality=quality,prep=prep,model=estimator,median=median,spec=spec,
        estimator_training_rows=len(rows),available_training_rows=len(y))


def predict(model,x,weight=1.):
    if 'constant' in model:return np.full(len(x),model['constant'])
    a=x[model['features']].to_numpy(float)
    raw=b.predict(model,a)
    effective=np.where(np.isfinite(a).mean(axis=1)>=.8,weight,0.)
    return effective*raw+(1-effective)*model['median']


def search(x,y,folds,folder):
    baseline=np.array([np.abs(y[va]-np.median(y[tr])).mean() for tr,va in folds])
    counts=[len(va) for _,va in folds]
    best=dict(model='median',features=[],validation_mae=float(np.average(baseline,weights=counts)))
    trials=[];quality=[]
    for pack,kind in product(PACKS,CONFIGS):
        predictions=[];usable=True
        for i,(tr,va) in enumerate(folds):
            model=fit(x.iloc[tr],y[tr],dict(model=kind,pack=pack))
            usable &= bool(model['features'])
            predictions.append(predict(model,x.iloc[va]))
            quality.append(dict(pack=pack,model=kind,window=i+1,features=model['features'],quality=model.get('quality',[])))
        for weight in WEIGHTS:
            errors=np.array([np.abs(y[va]-(weight*p+(1-weight)*np.median(y[tr]))).mean() for (tr,va),p in zip(folds,predictions)])
            eligible=bool(usable and np.average(errors,weights=counts)<np.average(baseline,weights=counts)*.999
                and np.sum(errors<=baseline+1e-10)>=3 and np.all(errors<=baseline*1.02+1e-10))
            row=dict(model=kind,pack=pack,weight=weight,validation_mae=float(np.average(errors,weights=counts)),
                window_mae=errors.tolist(),eligible=eligible)
            trials.append(row)
            if eligible and row['validation_mae']<best['validation_mae']:best=row
        print(f'  {pack}/{kind} complete',flush=True)
    b.dump(folder/'quality.json',quality);b.dump(folder/'validation_trials.json',trials)
    best=dict(best,baseline_window_mae=baseline.tolist());b.dump(folder/'selection.json',best)
    return best
