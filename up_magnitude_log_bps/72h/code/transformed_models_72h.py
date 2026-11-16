"""Training-only feature transforms and MAE-oriented alternative models."""
from itertools import product
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import SplineTransformer
import discover_excursion_features_72h as b

PACKS={
 'volatility':['vol_per_sqrt_hour__24h','vol_per_sqrt_hour__72h','vol_ratio_24__72h','up_variance_share__72h'],
 'path':['range_position__72h','drawdown_vol_units__168h','rebound_vol_units__72h','trend_efficiency__168h'],
 'mixed':['vol_per_sqrt_hour__168h','normalized_momentum__72h','range_position__168h','jump_variance_share__336h'],
}
PACKS={name:['LONG__'+c for c in cols] for name,cols in PACKS.items()}
TRANSFORMS=['raw','signed_log','rank']
CONFIGS={'spline_median':dict(knots=4,degree=2,alpha=.01),
         'extra_trees':dict(n_estimators=64,max_depth=5,min_samples_leaf=64,max_features=1.,bootstrap_rows=4096)}
WEIGHTS=[.25,.5,.75,1.]


def fit_transform(a,mode):
    a=np.asarray(a,float)
    b.check(mode in TRANSFORMS,'Unknown transform')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        median=np.nanmedian(np.where(np.isfinite(a),a,np.nan),axis=0)
        q=np.nanquantile(np.where(np.isfinite(a),a,np.nan),[.25,.75],axis=0)
    median=np.nan_to_num(median)
    scale=np.where(np.isfinite(q[1]-q[0])&((q[1]-q[0])>1e-8),q[1]-q[0],1.)
    references=[np.sort(a[np.isfinite(a[:,j]),j]) for j in range(a.shape[1])]
    return dict(mode=mode,median=median,scale=scale,references=references)


def transform(a,prep):
    a=np.asarray(a,float);missing=~np.isfinite(a)
    values=np.where(missing,prep['median'],a)
    if prep['mode']=='rank':
        cols=[]
        for j,reference in enumerate(prep['references']):
            if len(reference):
                # Midrank empirical CDF, with ties handled identically at inference.
                z=(np.searchsorted(reference,values[:,j],side='left')+
                   np.searchsorted(reference,values[:,j],side='right'))/(2.*len(reference))
            else:z=np.full(len(values),.5)
            cols.append(2*z-1)
        values=np.column_stack(cols)
    else:
        values=(values-prep['median'])/prep['scale']
        if prep['mode']=='signed_log':
            values=np.sign(values)*np.log1p(np.abs(values))
        # Raw is an affine-scaled control; no clipping or nonlinear transform.
    return np.column_stack([values,missing.astype(float)])


def fit(a,y,kind):
    if kind=='median' or a.shape[1]==0:return dict(constant=float(np.median(y)))
    family,mode,weight=kind.split('|');weight=float(weight)
    b.check(family in CONFIGS and weight in WEIGHTS,'Invalid model spec')
    prep=fit_transform(a,mode);z=transform(a,prep)
    # Positive constant target scaling preserves the absolute-error objective.
    unit=max(float(np.median(y)),1.);target=np.asarray(y)/unit
    spline=None
    if family=='spline_median':
        cfg=CONFIGS[family]
        spline=SplineTransformer(n_knots=cfg['knots'],degree=cfg['degree'],knots='quantile',
            include_bias=False,extrapolation='constant')
        count=a.shape[1]
        z=np.column_stack([spline.fit_transform(z[:,:count]),z[:,count:]])
        model=QuantileRegressor(quantile=.5,alpha=cfg['alpha'],solver='highs')
    else:
        cfg=CONFIGS[family]
        model=ExtraTreesRegressor(n_estimators=cfg['n_estimators'],criterion='absolute_error',
            max_depth=cfg['max_depth'],min_samples_leaf=cfg['min_samples_leaf'],max_features=cfg['max_features'],
            bootstrap=True,max_samples=min(cfg['bootstrap_rows'],len(y)),random_state=48,n_jobs=1)
    model.fit(z,target)
    return dict(model=model,prep=prep,spline=spline,n_features=a.shape[1],unit=unit,
        weight=weight,median=float(np.median(y)),kind=kind)


def predict(model,a):
    if 'constant' in model:return np.full(len(a),model['constant'])
    z=transform(a,model['prep']);count=model['n_features']
    if model['spline'] is not None:
        z=np.column_stack([model['spline'].transform(z[:,:count]),z[:,count:]])
    raw=np.maximum(0.,model['model'].predict(z)*model['unit'])
    result=model['weight']*raw+(1-model['weight'])*model['median']
    b.check(np.isfinite(result).all() and (result>=0).all(),'Invalid predictions')
    return result


def gate(errors,baseline,counts,args):
    return (np.average(errors,weights=counts)<np.average(baseline,weights=counts)*(1-args.min_gain)
        and int((errors<baseline).sum())>=2 and np.all(errors<=baseline*(1+args.max_window_loss)+1e-12))


def search(x,y,folds,args,folder):
    baseline=np.array([np.abs(y[va]-np.median(y[tr])).mean() for tr,va in folds])
    counts=[len(va) for _,va in folds];admitted={};quality=[]
    for pack,cols in PACKS.items():
        valid=all(np.isfinite(x.iloc[tr][c]).mean()>=.8 and np.isfinite(x.iloc[tr][c]).sum()>=1000
            and x.iloc[tr][c].nunique()>1 for tr,_ in folds for c in cols)
        quality.append(dict(pack=pack,eligible=valid,features=cols))
        if valid:admitted[pack]=cols
    pd.DataFrame(quality).to_csv(folder/'quality.csv',index=False)
    best=None;trials=[]
    for pack,family,mode in product(admitted,CONFIGS,TRANSFORMS):
        cols=admitted[pack];a=x[cols].to_numpy(float);predictions=[]
        for tr,va in folds:
            model=fit(a[tr],y[tr],family+'|'+mode+'|1.0')
            predictions.append(predict(model,a[va]))
        for weight in WEIGHTS:
            errors=np.array([np.abs(y[va]-(weight*p+(1-weight)*np.median(y[tr]))).mean()
                for (tr,va),p in zip(folds,predictions)])
            eligible=bool(gate(errors,baseline,counts,args))
            row=dict(model=family+'|'+mode+'|'+str(weight),pack=pack,features=cols,
                validation_mae=float(np.average(errors,weights=counts)),validation_window_mae=errors.tolist(),eligible=eligible)
            trials.append(row)
            if eligible and (best is None or row['validation_mae']<best['validation_mae']):best=row
        print(f'  {pack}/{family}/{mode} complete',flush=True)
    b.dump(folder/'validation_trials.json',trials)
    if best is None:best=dict(model='median',features=[],validation_mae=float(np.average(baseline,weights=counts)))
    best=dict(best,baseline_mae=float(np.average(baseline,weights=counts)),baseline_window_mae=baseline.tolist())
    b.dump(folder/'selection.json',best)
    return best
