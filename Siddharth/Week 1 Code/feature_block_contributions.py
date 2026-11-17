"""Saved-model reliance diagnostics; no fitting, tuning or model changes."""
import json
import numpy as np
import pandas as pd
import joblib,torch
import two_group_tuning as t
s=t.s
OUT=s.HERE/'experiments/feature_block_contributions_v1'
def run():
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    x,y,dev,ref=joblib.load(t.e.OUT/'prepared.joblib');base=pd.read_csv(t.OUT/'predictions.csv',index_col=0,parse_dates=True)
    assert base.index.equals(dev)
    rows=[];pieces={};window=[]
    for method in ['rf','mlp']:
        for issue in s.q.STARTS:
            f=t.OUT/method/'eight'/f'{issue:%Y%m%d}';choice=json.loads((f/'SELECTION.json').read_text());t.e.configure(method,choice['config'])
            art=joblib.load(f/'outer/bundle.joblib');_,_,_,_,te=s.q.windows(y,dev,issue)
            original=s.predict(art,x,te);np.testing.assert_allclose(original,base.loc[te,method+'_eight'],atol=1e-8,rtol=0)
            blocks=(te.asi8-te.asi8.min())//pd.Timedelta(days=7).value
            groups=[np.flatnonzero(blocks==b) for b in np.unique(blocks)]
            for ci,c in enumerate(t.e.COLS):
                for repeat in range(5):
                    rng=np.random.default_rng(20261105+ci*100+repeat)
                    order=rng.permutation(len(groups))
                    if np.array_equal(order,np.arange(len(groups))):order=np.roll(order,1)
                    ids=np.concatenate([groups[k] for k in order]);shuffled=x.loc[te].copy();shuffled[c]=shuffled[c].to_numpy()[ids]
                    pred=s.predict(art,shuffled,te);pieces.setdefault((method,c,repeat),[]).append(pd.Series(pred,index=te))
                    window.append(dict(method=method,feature=c,repeat=repeat,issue=str(issue),mae_increase=float(np.mean(abs(pred-base.loc[te,'truth']))-np.mean(abs(original-base.loc[te,'truth'])))))
            print(method,issue,'done',flush=True)
        base_mae=float(np.mean(abs(base[method+'_eight']-base.truth)));base_ic=base[method+'_eight'].corr(base.truth,method='spearman')
        for c in t.e.COLS:
            maes=[];ics=[];losses=[]
            for repeat in range(5):
                pred=pd.concat(pieces[method,c,repeat]);assert pred.index.equals(dev)
                losses.append(abs(pred-base.truth)-abs(base[method+'_eight']-base.truth));maes.append(losses[-1].mean());ics.append(base_ic-pred.corr(base.truth,method='spearman'))
            delta=pd.concat(losses,axis=1).mean(axis=1);blocks=(dev.asi8-dev.asi8.min())//pd.Timedelta(days=7).value
            b=pd.DataFrame(dict(delta=delta.to_numpy(),block=blocks)).groupby('block').delta.agg(['sum','count']);ids=np.random.default_rng(48).integers(0,len(b),(3000,len(b)))
            boot=b['sum'].to_numpy()[ids].sum(1)/b['count'].to_numpy()[ids].sum(1);lo,hi=np.quantile(boot,[.025,.975])
            rows.append(dict(method=method,feature=c,baseline_mae=base_mae,mae_increase_when_shuffled=float(np.mean(maes)),ic_drop_when_shuffled=float(np.mean(ics)),lower=lo,upper=hi,shuffle_min=min(maes),shuffle_max=max(maes)))
    result=pd.DataFrame(rows).sort_values(['method','mae_increase_when_shuffled'],ascending=[True,False]);result.to_csv(OUT/'contributions.csv',index=False);pd.DataFrame(window).to_csv(OUT/'window_contributions.csv',index=False)
    s.dump(OUT/'COMPLETE.json',dict(status='EXECUTED',rows=len(dev),features=8,repeats=5,method='Permutation of contiguous7-day blocks separately within each quarterly evaluation window; original within-block order retained. Saved models unchanged; original predictions replayed.',interpretation='Positive MAE increase means model relies beneficially on feature; positive IC drop means ranking worsens without its alignment. Correlated features and off-distribution combinations limit interpretation. Bootstrap intervals are descriptive pointwise intervals, not multiplicity adjusted. This is not a retraining/addition test or proof of baseline improvement.'))
    print(result.to_string(index=False),flush=True)
if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):run()
