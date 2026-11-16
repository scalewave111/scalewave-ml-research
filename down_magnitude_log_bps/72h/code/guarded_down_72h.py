"""Causal residual/sizing models and training-only stability guards for DOWN72h."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import SGDRegressor
import discover_excursion_features_72h as b
from compare_rolling_medians_72h import causal_medians
from transformed_models_72h import fit_transform, transform

FAMILIES=['residual_linear','residual_tree','quantile_map','bins3','bins5']
WEIGHTS=[.25,.5,1.]
DEADZONES=[0.,.05,.1]


def baselines(y):
    frame,counts=causal_medians(y,y.index,days=[180],minimum=48)
    rolling=frame.rolling_180d.fillna(frame.online_expanding)
    return rolling.to_numpy(float),counts.rolling_180d.to_numpy(float)


def windows(index,issue,baseline):
    result=[]
    for offset in [12,9,6,3]:
        start=issue-pd.DateOffset(months=offset);end=start+pd.DateOffset(months=3)
        tr=np.flatnonzero((index+pd.Timedelta(hours=72)<start)&np.isfinite(baseline))
        va=np.flatnonzero((index>=start)&(index+pd.Timedelta(hours=72)<end)&np.isfinite(baseline))
        b.check(len(tr)>=1000 and len(va)>=100,'Insufficient four-window history')
        result.append((tr,va))
    return result


def stable_features(x,y,rows,maximum=6):
    """Three trailing calendar-quarter-length blocks within mature training rows."""
    end=x.index[rows].max()+pd.Timedelta(minutes=30)
    records=[]
    for col in x:
        corr=[];coverage=[];counts=[]
        for offset in [9,6,3]:
            start=end-pd.DateOffset(months=offset);stop=start+pd.DateOffset(months=3)
            sub=rows[(x.index[rows]>=start)&(x.index[rows]<stop)]
            values=x.iloc[sub][col].to_numpy(float);ok=np.isfinite(values)
            coverage.append(float(ok.mean()) if len(ok) else 0.)
            counts.append(int(ok.sum()))
            c=pd.Series(values[ok]).corr(pd.Series(y[sub][ok]),method='spearman') if ok.sum()>=300 and np.std(values[ok])>0 and np.std(y[sub][ok])>0 else np.nan
            corr.append(float(c))
        same_sign=bool(np.all(np.array(corr)>.01) or np.all(np.array(corr)<-.01))
        eligible=(min(counts)>=300 and min(coverage)>=.8 and max(coverage)-min(coverage)<=.1 and same_sign)
        records.append(dict(feature=col,eligible=bool(eligible),correlations=corr,coverage=coverage,
            score=float(min(abs(c) for c in corr)) if np.isfinite(corr).all() else 0.))
    ranked=sorted([r for r in records if r['eligible']],key=lambda r:(-r['score'],r['feature']))
    return [r['feature'] for r in ranked[:maximum]],records


def tree():
    return HistGradientBoostingRegressor(loss='absolute_error',max_depth=2,max_leaf_nodes=4,
        min_samples_leaf=256,max_iter=80,learning_rate=.05,l2_regularization=10.,early_stopping=False,random_state=48)


def fit(x,y,baseline,rows,family):
    rows=np.asarray(rows);cal=np.array([],dtype=int);core=rows
    if family=='quantile_map':
        start=x.index[rows].max()-pd.Timedelta(days=90)
        core=rows[x.index[rows]+pd.Timedelta(hours=72)<start]
        cal=rows[x.index[rows]>=start]
        if len(core)<1000 or len(cal)<300:return dict(family=family,features=[],unavailable=True)
    cols,audit=stable_features(x,y,core)
    if not cols:return dict(family=family,features=[],unavailable=True,feature_audit=audit)
    a=x.iloc[core][cols].to_numpy(float)
    prep=fit_transform(a,'rank');z=transform(a,prep)
    unit=max(float(np.median(y[core])),1.)
    q=np.nanquantile(a,[.01,.99],axis=0)
    model=dict(family=family,features=cols,prep=prep,unit=unit,
        low=q[0],high=q[1],missing=np.mean(~np.isfinite(a),axis=0),
        dead_scale=max(float(np.quantile(y[core],.75)-np.quantile(y[core],.25)),1.),
        feature_audit=audit,fit_rows=len(core),fit_last=str(x.index[core].max()),
        calibration_rows=len(cal),calibration_start=str(x.index[cal].min()) if len(cal) else None)
    if family=='residual_linear':
        estimator=SGDRegressor(loss='epsilon_insensitive',epsilon=0.,penalty='elasticnet',
            alpha=.01,l1_ratio=.5,max_iter=3000,tol=1e-5,early_stopping=False,
            learning_rate='invscaling',eta0=.001,power_t=.25,random_state=48,shuffle=False)
        estimator.fit(z,(y[core]-baseline[core])/unit)
    elif family.startswith('bins'):
        n=int(family[-1]);cuts=np.unique(np.quantile(y[core],np.arange(1,n)/n))
        labels=np.searchsorted(cuts,y[core],side='right')
        classes=np.unique(labels)
        centers=np.array([np.median(y[core][labels==c]) for c in classes])
        model.update(cuts=cuts,centers=centers,classes=classes)
        if len(classes)==1:
            model['constant']=float(centers[0]);return model
        estimator=HistGradientBoostingClassifier(max_depth=2,max_leaf_nodes=4,min_samples_leaf=256,
            max_iter=80,learning_rate=.05,l2_regularization=10.,early_stopping=False,random_state=48)
        estimator.fit(z,labels)
    else:
        estimator=tree();estimator.fit(z,(y[core]-baseline[core])/unit)
    model['estimator']=estimator
    if family=='quantile_map':
        # Purged held-out calibration, no in-sample prediction ranks; core model
        # remains frozen at final inference so score scale does not silently change.
        score=baseline[cal]+estimator.predict(transform(x.iloc[cal][cols].to_numpy(float),prep))*unit
        model['score_reference']=np.sort(score);model['outcome_reference']=np.sort(y[cal])
        model['calibration_label_end_max']=str(x.index[cal].max()+pd.Timedelta(hours=72))
    return model


def raw_predict(model,x,baseline,rows):
    base=baseline[rows]
    if model.get('unavailable'):return base.copy()
    if 'constant' in model:return np.full(len(rows),model['constant'])
    z=transform(x.iloc[rows][model['features']].to_numpy(float),model['prep'])
    if model['family'].startswith('bins'):
        classes=model['estimator'].predict(z)
        return model['centers'][np.searchsorted(model['classes'],classes)]
    raw=base+model['estimator'].predict(z)*model['unit']
    if model['family']=='quantile_map':
        ref=model['score_reference']
        percentile=(np.searchsorted(ref,raw,'left')+np.searchsorted(ref,raw,'right'))/(2.*len(ref))
        # Winsorized percentiles reduce tail sensitivity; this is marginal quantile
        # mapping, not calibrated conditional medians, and can worsen MAE.
        raw=np.quantile(model['outcome_reference'],np.clip(percentile,.05,.95))
    return raw


def dynamic_weight(model,x,rows):
    if model.get('unavailable'):return np.zeros(len(rows))
    cols=model['features'];a=x.iloc[rows][cols].to_numpy(float)
    missing=~np.isfinite(a)
    recent=(~np.isfinite(x[cols])).astype(float).rolling('7D',closed='left',min_periods=48).mean().iloc[rows].to_numpy()
    spike=np.any(recent>model['missing']+.1,axis=1)
    outside=np.mean((a<model['low'])|(a>model['high']),axis=1)
    weight=1.-.75*outside
    weight[missing.any(axis=1)|spike|~np.isfinite(recent).all(axis=1)]=0.
    return weight


def adjust(raw,base,quality,scale,weight,deadzone,dynamic):
    delta=raw-base
    effective=weight*(quality if dynamic else np.ones(len(base)))
    effective=np.where(np.abs(delta)>=deadzone*scale,effective,0.)
    return np.maximum(0.,base+effective*delta),effective


def gate(errors,controls,counts,gain=.001):
    return all(np.average(errors,weights=counts)<np.average(c,weights=counts)*(1-gain)
        and np.sum(errors<=c+1e-10)>=3 and np.all(errors<=c*1.02+1e-10) for c in controls)


def search(x,y,baseline,folds,folder):
    counts=[len(va) for _,va in folds]
    # Quarterly control uses ALL mature eligible labels, even the first labels
    # that lack the 48-history minimum required for residual targets.
    frozen=[float(np.median(y[x.index+pd.Timedelta(hours=72)<x.index[va].min()])) for _,va in folds]
    constant=np.array([np.abs(y[va]-m).mean() for (_,va),m in zip(folds,frozen)])
    rolling=np.array([np.abs(y[va]-baseline[va]).mean() for _,va in folds])
    trials=[];audit=[];best=dict(family='quarterly_median',validation_mae=float(np.average(constant,weights=counts)))
    if gate(rolling,[constant],counts):
        best=dict(family='rolling_median',validation_mae=float(np.average(rolling,weights=counts)))
    trials.append(dict(family='rolling_median',window_mae=rolling.tolist(),eligible=gate(rolling,[constant],counts)))
    for family in FAMILIES:
        fitted=[]
        for i,(tr,va) in enumerate(folds):
            model=fit(x,y,baseline,tr,family)
            raw=raw_predict(model,x,baseline,va)
            quality=dynamic_weight(model,x,va)
            fitted.append((model,raw,quality))
            audit.append(dict(family=family,window=i+1,features=model['features'],
                unavailable=model.get('unavailable',False),feature_audit=model.get('feature_audit',[]),
                fit_last=model.get('fit_last'),calibration_start=model.get('calibration_start'),
                calibration_label_end_max=model.get('calibration_label_end_max')))
        if any(m.get('unavailable') for m,_,_ in fitted):continue
        for weight in WEIGHTS:
            for deadzone in DEADZONES:
                for dynamic in [False,True]:
                    errors=np.array([np.abs(y[va]-adjust(raw,baseline[va],q,m['dead_scale'],weight,deadzone,dynamic)[0]).mean()
                        for (_,va),(m,raw,q) in zip(folds,fitted)])
                    allowed=bool(gate(errors,[constant,rolling],counts))
                    row=dict(family=family,weight=weight,deadzone=deadzone,dynamic=dynamic,
                        window_mae=errors.tolist(),validation_mae=float(np.average(errors,weights=counts)),eligible=allowed)
                    trials.append(row)
                    if allowed and row['validation_mae']<best['validation_mae']:best=row
        print('  '+family+' completed',flush=True)
    # JSON NaN is forbidden; undefined training correlations remain explicit null.
    clean=lambda v: {k:clean(a) for k,a in v.items()} if isinstance(v,dict) else [clean(a) for a in v] if isinstance(v,list) else None if isinstance(v,float) and not np.isfinite(v) else v
    b.dump(folder/'feature_stability.json',clean(audit))
    b.dump(folder/'validation_trials.json',trials)
    best=dict(best,quarterly_window_mae=constant.tolist(),rolling_window_mae=rolling.tolist())
    b.dump(folder/'selection.json',best)
    return best
