"""Robust descriptive statistics and chronological residual-group diagnostics; no RF/MLP training."""
from pathlib import Path
import itertools,json,hashlib
import numpy as np
import pandas as pd
import joblib
from scipy.stats import rankdata,spearmanr
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'Code/experiments/eight_feature_statistics_v1'
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')
def block_ci(a,b,index,seed=48):
    # Correlation of fixed empirical ranks; time-block pairs bootstrap.
    a=rankdata(a);b=rankdata(b);g=(index.asi8-index.asi8.min())//pd.Timedelta(days=7).value
    frame=pd.DataFrame(dict(g=g,n=1,a=a,b=b,aa=a*a,bb=b*b,ab=a*b)).groupby('g').sum()
    z=frame.to_numpy();ids=np.random.default_rng(seed).integers(0,len(z),(2000,len(z)));v=z[ids].sum(1)
    n,sa,sb,saa,sbb,sab=v.T
    cor=(sab-sa*sb/n)/np.sqrt(np.maximum((saa-sa*sa/n)*(sbb-sb*sb/n),1e-20))
    return np.quantile(cor,[.025,.975]),len(z)
def run():
    OUT.mkdir(parents=True,exist_ok=True)
    src=ROOT/'Code/experiments/eight_feature_exhaustive_v1/prepared.joblib'
    x,y,dev,ref=joblib.load(src);cols=list(x.columns)
    valpath=ROOT/'Code/experiments/baseline_single_feature_v6/validation_predictions.csv'
    v=pd.read_csv(valpath,index_col=0,parse_dates=True).sort_index();assert v.index.is_unique
    np.testing.assert_allclose(v.truth,y.loc[v.index,'magnitude_log_bps'],atol=1e-8,rtol=0)
    # Only the earlier baseline validation predictions, not final development forecasts.
    xx=x.loc[v.index];res=v.truth-v.baseline
    quality=[];stats=[];monthly=[]
    for c in cols:
        good=xx[c].notna()&np.isfinite(xx[c]);z=xx.loc[good,c];idx=z.index
        quality.append(dict(feature=c,coverage=float(good.mean()),unique=z.nunique(),median=z.median(),p01=z.quantile(.01),p99=z.quantile(.99)))
        for outcome,target in [('magnitude',v.truth),('baseline_residual',res)]:
            ci,blocks=block_ci(z.to_numpy(),target.loc[idx].to_numpy(),idx)
            stats.append(dict(feature=c,outcome=outcome,spearman=spearmanr(z,target.loc[idx]).statistic,lower=ci[0],upper=ci[1],weekly_blocks=blocks))
        for period,part in z.groupby(z.index.strftime('%Y-%m')):
            monthly.append(dict(feature=c,month=period,rows=len(part),magnitude_ic=part.corr(v.truth.loc[part.index],method='spearman'),residual_ic=part.corr(res.loc[part.index],method='spearman')))
    pd.DataFrame(quality).to_csv(OUT/'feature_quality.csv',index=False)
    pd.DataFrame(stats).to_csv(OUT/'rank_relationships.csv',index=False)
    pd.DataFrame(monthly).to_csv(OUT/'monthly_stability.csv',index=False)
    xx.corr(method='spearman').to_csv(OUT/'feature_redundancy.csv')
    cut1=pd.Timestamp('2024-01-01',tz='UTC');cut2=pd.Timestamp('2024-04-01',tz='UTC')
    train=v.index[y.loc[v.index,'label_end']<cut1]
    valid=v.index[(v.index>=cut1)&(y.loc[v.index,'label_end']<cut2)]
    test=v.index[v.index>=cut2];assert len(train)>1000 and len(valid)>1000 and len(test)>1000
    def design(keys,fitkeys):
        base=xx.loc[fitkeys];med=base.median();scale=base.quantile(.75)-base.quantile(.25);scale=scale.where(scale>1e-8,1)
        return ((xx.loc[keys].fillna(med)-med)/scale).clip(-5,5).to_numpy()
    # A small diagnostic adjustment, penalized toward the saved baseline.
    # Each subset uses the same fixed ridge penalty and robust residual clipping.
    def fit_predict(sub,fitkeys,predkeys):
        a=design(fitkeys,fitkeys)[:,sub];b=design(predkeys,fitkeys)[:,sub]
        center=a.mean(0);a=a-center;b=b-center
        r=res.loc[fitkeys].to_numpy();lo,hi=np.quantile(r,[.01,.99]);r=np.clip(r,lo,hi)
        coef=np.linalg.solve(a.T@a+len(a)*.1*np.eye(len(sub)),a.T@(r-r.mean()))
        return b@coef  # No intercept: isolate feature-dependent corrections.
    trials=[]
    for n in range(1,9):
        for sub in itertools.combinations(range(8),n):
            correction=fit_predict(list(sub),train,valid)
            options=[(float(np.mean(abs(v.loc[valid,'baseline']+w*correction-v.loc[valid,'truth']))),w) for w in [0,.1,.25,.5,1]]
            mae,w=min(options);trials.append(dict(indices=list(sub),features=[cols[i] for i in sub],n_features=n,weight=w,validation_mae=mae))
    trials.sort(key=lambda r:(r['validation_mae'],r['n_features'],r['indices']))
    dump('ALL_255_GROUPS.json',trials)
    # Freeze three nonidentical finalists on validation only before later evaluation.
    finalists=trials[:3];dump('FROZEN_GROUPS.json',finalists)
    fitkeys=v.index[y.loc[v.index,'label_end']<cut2]
    predictions=pd.DataFrame(dict(truth=v.loc[test,'truth'],baseline=v.loc[test,'baseline']),index=test)
    results=[]
    for i,row in enumerate(finalists):
        corr=fit_predict(row['indices'],fitkeys,test);pred=np.maximum(0,v.loc[test,'baseline'].to_numpy()+row['weight']*corr)
        predictions[f'group_{i+1}']=pred
        delta=abs(pred-v.loc[test,'truth'].to_numpy())-abs(v.loc[test,'baseline']-v.loc[test,'truth']).to_numpy()
        intervals={}
        for days in [7,14]:
            blocks=(test.asi8-test.asi8.min())//pd.Timedelta(days=days).value
            b=pd.DataFrame(dict(g=blocks,delta=delta)).groupby('g').delta.agg(['sum','count']);ids=np.random.default_rng(48).integers(0,len(b),(5000,len(b)))
            boot=b['sum'].to_numpy()[ids].sum(1)/b['count'].to_numpy()[ids].sum(1)
            intervals[str(days)]=np.quantile(boot,[.025,.975]).tolist()
        results.append(dict(group=i+1,**row,later_mae=float(np.mean(abs(pred-v.loc[test,'truth']))),later_baseline_mae=float(np.mean(abs(v.loc[test,'baseline']-v.loc[test,'truth']))),mae_difference=float(delta.mean()),block_intervals=intervals))
    predictions.to_csv(OUT/'later_diagnostic_predictions.csv')
    dump('GROUP_RESULTS.json',results)
    dump('METHOD_AND_AUDIT.json',dict(status='EXECUTED',target='Absolute48h endpoint log return in log bps',features=cols,fit_period=[str(train.min()),str(train.max())],validation_period=[str(valid.min()),str(valid.max())],assessment_period=[str(test.min()),str(test.max())],rows=dict(fit=len(train),validation=len(valid),assessment=len(test)),purge='label_end strictly before each cutoff',method='Training median/IQR scaling, clipping5IQR, ridge penalty0.1*n, training residual winsorization1/99%, no intercept, validation selects subset and shrinkage. All255 subsets tested cheaply.',limitations='Diagnostic residual regression is not the hybrid/MLP. Saved baseline validation predictions vary by quarter. This is not the18481-row original test comparison. Historical data previously examined;255-group selection is exploratory. Rank CIs are pointwise, not multiple-testing-adjusted. Block intervals descriptive and do not prove future gains.',inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [src,valpath]}))
    print(json.dumps(results,indent=2),flush=True)
if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):run()
