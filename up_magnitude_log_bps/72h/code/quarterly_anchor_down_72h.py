"""Controlled baseline-anchor comparison with fixed residual-tree machinery."""
import numpy as np
import pandas as pd
import guarded_down_72h as previous

b=previous.b
WEIGHTS=[.1,.25,.5]


def anchors(labels):
    index=labels.index
    issue=index.tz_localize(None).to_period('Q').start_time.tz_localize('UTC')
    # Existing evaluation contract keeps October2024 model/anchor through Jan2025.
    issue=pd.DatetimeIndex([min(t,b.ISSUES[-1]) for t in issue])
    result=np.full(len(labels),np.nan);ledger=[]
    for date in issue.unique():
        mature=labels[index+pd.Timedelta(hours=72)<date]
        median=float(mature.median()) if len(mature)>=48 else np.nan
        result[issue==date]=median
        ledger.append(dict(issue=date,mature_rows=len(mature),median=median,
            latest_label_end=mature.index.max()+pd.Timedelta(hours=72) if len(mature) else None))
    rolling,_=previous.baselines(labels)
    return dict(quarterly=result,rolling=rolling),pd.DataFrame(ledger)


def features(x,y,rows):
    # Stable signs/coverage screen unchanged; redundancy is measured on training only.
    ranked,audit=previous.stable_features(x,y,rows,maximum=len(x.columns))
    selected=[];rejected=[]
    for col in ranked:
        duplicate=None
        for kept in selected:
            pair=x.iloc[rows][[col,kept]].dropna()
            if len(pair)>=1000 and abs(pair.iloc[:,0].corr(pair.iloc[:,1],method='spearman'))>=.9:
                duplicate=kept;break
        if duplicate:rejected.append(dict(feature=col,redundant_with=duplicate))
        else:selected.append(col)
        if len(selected)>=6:break
    return selected,audit,rejected


def fit(x,y,baseline,rows):
    cols,audit,redundant=features(x,y,rows)
    result=dict(features=cols,feature_audit=audit,redundant=redundant)
    if not cols:return dict(result,unavailable=True)
    a=x.iloc[rows][cols].to_numpy(float)
    prep=previous.fit_transform(a,'rank')
    unit=max(float(np.median(y[rows])),1.)
    model=previous.tree()
    model.fit(previous.transform(a,prep),(y[rows]-baseline[rows])/unit)
    return dict(result,prep=prep,unit=unit,model=model)


def predict(model,x,baseline,rows,weight):
    anchor=baseline[rows]
    if model.get('unavailable'):return anchor.copy(),np.zeros(len(rows))
    a=x.iloc[rows][model['features']].to_numpy(float)
    missing=(~np.isfinite(a)).any(axis=1)
    effective=np.where(missing,0.,weight)
    adjustment=model['model'].predict(previous.transform(a,model['prep']))*model['unit']
    return np.maximum(0.,anchor+effective*adjustment),effective


def clean(v):
    if isinstance(v,dict):return {k:clean(a) for k,a in v.items()}
    if isinstance(v,list):return [clean(a) for a in v]
    return None if isinstance(v,float) and not np.isfinite(v) else v


def search(x,y,base,folds,folder):
    counts=[len(va) for _,va in folds]
    control=np.array([np.abs(y[va]-base['quarterly'][va]).mean() for _,va in folds])
    best=dict(anchor='quarterly',weight=0.,validation_mae=float(np.average(control,weights=counts)))
    trials=[];audits=[]
    for anchor in ['quarterly','rolling']:
        fitted=[]
        for i,(tr,va) in enumerate(folds):
            model=fit(x,y,base[anchor],tr)
            fitted.append(model)
            audits.append(dict(anchor=anchor,window=i+1,features=model['features'],
                unavailable=model.get('unavailable',False),feature_audit=model['feature_audit'],redundant=model['redundant']))
        for weight in WEIGHTS:
            errors=np.array([np.abs(y[va]-predict(m,x,base[anchor],va,weight)[0]).mean() for (_,va),m in zip(folds,fitted)])
            eligible=bool(previous.gate(errors,[control],counts)) and not any(m.get('unavailable') for m in fitted)
            row=dict(anchor=anchor,weight=weight,validation_mae=float(np.average(errors,weights=counts)),
                window_mae=errors.tolist(),eligible=eligible)
            trials.append(row)
            if anchor=='quarterly' and eligible and row['validation_mae']<best['validation_mae']:best=row
    b.dump(folder/'feature_audit.json',clean(audits))
    b.dump(folder/'validation_trials.json',trials)
    best=dict(best,quarterly_window_mae=control.tolist())
    b.dump(folder/'selection.json',best)
    return best


def bootstrap(frame,model,days,draws=5000):
    """Paired nonoverlapping calendar blocks, stratified by model issue date.

    Fixed stratum weights preserve the original evaluation-period mixture.
    Conditional on saved forecasts; no model reselection or retraining.
    """
    b.check(draws>=100 and days>=3,'Invalid bootstrap options')
    rng=np.random.default_rng(48+days);samples=np.zeros(draws);blocks=[]
    gain=np.abs(frame.actual-frame.training_median)-np.abs(frame.actual-frame[model])
    for issue,part in frame.groupby('issue',sort=True):
        block=(part.decision_time-pd.Timestamp(issue))//pd.Timedelta(days=days)
        grouped=pd.DataFrame(dict(block=block.to_numpy(),gain=gain.loc[part.index].to_numpy(),n=np.ones(len(part)))).groupby('block')[['gain','n']].sum().to_numpy()
        count=len(grouped);b.check(count>0,'Empty bootstrap stratum')
        selection=rng.integers(0,count,size=(draws,count))
        totals=grouped[selection].sum(axis=1)
        samples+=(len(part)/len(frame))*(totals[:,0]/totals[:,1])
        blocks.append(dict(issue=str(issue),blocks=count,rows=len(part)))
    return dict(model=model,block_days=days,draws=draws,mae_gain=float(gain.mean()),
        lower95=float(np.quantile(samples,.025)),upper95=float(np.quantile(samples,.975)),
        bootstrap_fraction_positive=float(np.mean(samples>0)),blocks=blocks)
