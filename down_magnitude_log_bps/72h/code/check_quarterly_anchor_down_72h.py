"""Synthetic-only checks for historical anchors, comparisons and bootstrap."""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
import quarterly_anchor_down_72h as q
import fit_quarterly_anchor_down_72h as runner


def main():
    rng=np.random.default_rng(48)
    index=pd.date_range('2021-01-01','2025-01-22',freq='3h',tz='UTC')
    signal=rng.uniform(0,10,len(index));y=100+20*signal
    x=pd.DataFrame(dict(signal=signal,signal_copy=2*signal,noise=rng.normal(size=len(index))),index=index)
    labels=pd.Series(y,index=index);base,ledger=q.anchors(labels)
    for issue in q.b.ISSUES:
        rows=(index>=issue)&(index<issue+pd.DateOffset(months=3))
        np.testing.assert_allclose(base['quarterly'][rows],labels[index+pd.Timedelta(hours=72)<issue].median())
    np.testing.assert_allclose(base['quarterly'][index>=pd.Timestamp('2025-01-01',tz='UTC')],labels[index+pd.Timedelta(hours=72)<q.b.ISSUES[-1]].median())
    cut=pd.Timestamp('2024-04-01',tz='UTC');altered=labels.copy()
    altered.loc[index+pd.Timedelta(hours=72)>=cut]=1e6
    again,_=q.anchors(altered)
    for name in base:np.testing.assert_allclose(base[name][index<=cut],again[name][index<=cut],equal_nan=True,rtol=0,atol=1e-10)
    joint=np.where(np.isfinite(base['quarterly'])&np.isfinite(base['rolling']),base['quarterly'],np.nan)
    issue=q.b.ISSUES[0];folds=q.previous.windows(index,issue,joint)
    tr=np.flatnonzero((index+pd.Timedelta(hours=72)<issue)&np.isfinite(joint))
    va=np.flatnonzero((index>=issue)&(index<issue+pd.Timedelta(days=30)))
    cols,audit,rejected=q.features(x,y,tr)
    assert 'signal' in cols and 'signal_copy' not in cols and any(r['feature']=='signal_copy' for r in rejected)
    with tempfile.TemporaryDirectory() as temp,threadpool_limits(limits=2):
        root=Path(temp)
        for anchor in base:
            model=q.fit(x,y,base[anchor],tr)
            for weight in q.WEIGHTS:
                pred,eff=q.predict(model,x,base[anchor],va,weight)
                assert np.isfinite(pred).all() and (pred>=0).all()
                np.testing.assert_allclose(eff,weight)
                missing=x.copy();missing.loc[index[va[0]],model['features'][0]]=np.nan
                p,w=q.predict(model,missing,base[anchor],va,weight)
                assert p[0]==base[anchor][va[0]] and w[0]==0
            joblib.dump(model,root/'model.joblib')
            np.testing.assert_allclose(q.predict(model,x,base[anchor],va,.25)[0],q.predict(joblib.load(root/'model.joblib'),x,base[anchor],va,.25)[0],rtol=0,atol=1e-10)
        for name in ['a','b','zero']:(root/name).mkdir()
        choice=q.search(x,y,base,folds,root/'a');assert choice['anchor']=='quarterly'
        xx=x.copy();xx.loc[index>=issue]=1e6;yy=y.copy();yy[index>=issue]=1e6
        assert choice==q.search(xx,yy,base,folds,root/'b')
        zero=q.search(x,np.zeros(len(y)),{k:v*0 for k,v in base.items()},folds,root/'zero')
        assert zero['weight']==0 and zero['anchor']=='quarterly'
    # Paired calendar-block bootstrap: exact constant gain, same block resampling
    # for predictions and control, deterministic seed, issue-stratum weighting.
    sample=pd.DataFrame(dict(decision_time=index[index>=issue],actual=10.,training_median=5.,selected=7.))
    sample['issue']=[q.b.ISSUES[q.b.ISSUES<=t][-1] for t in sample.decision_time]
    for days in [7,14,28]:
        row=q.bootstrap(sample,'selected',days,100)
        np.testing.assert_allclose([row['mae_gain'],row['lower95'],row['upper95']],[2,2,2])
        assert row==q.bootstrap(sample,'selected',days,100)
    sample['selected']=sample.training_median
    assert q.bootstrap(sample,'selected',14,100)['upper95']==0
    x.attrs['excluded_labels']=pd.DataFrame(columns=['decision_time','label_status'])
    dev=index[index>=issue]
    old=pd.DataFrame([dict(decision_time=t,training_median=base['quarterly'][index.get_loc(t)],
        previous_guarded=200.,previous_transformed=200.,previous_neural=200.) for t in dev])
    dictionary=pd.DataFrame([dict(feature=c,formula='synthetic') for c in x])
    def load(args):return x,y,dev,old,dictionary,base,ledger,[]
    def choose(*args):return dict(anchor='quarterly',weight=.25,validation_mae=1.)
    with tempfile.TemporaryDirectory() as temp,threadpool_limits(limits=2):
        with patch.object(runner,'load',load),patch.object(q,'search',choose),patch.object(sys,'argv',['runner','--output-dir',temp,'--bootstrap-draws','100']):runner.main()
        out=list(Path(temp).iterdir())[0];done=json.loads((out/'COMPLETE.json').read_text())
        assert done['replay_passed'] and done['development_origins']==len(dev)
        assert done['predictions_sha256']==q.b.sha(out/'predictions.csv')
        assert done['bootstrap_sha256']==q.b.sha(out/'bootstrap.json')
        assert len(pd.read_csv(out/'bootstrap.csv'))==21
    print('PASS: historical quarterly availability/maturity/October freeze; redundancy; matched anchor fits/missing fallback/replay; selection isolation/zero fallback; paired bootstrap; full save-score pipeline.')


if __name__=='__main__':main()
