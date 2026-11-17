"""Synthetic unit and end-to-end checks; no actual market fitting."""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
import guarded_down_72h as g
import fit_guarded_down_72h as runner


def main():
    rng=np.random.default_rng(48)
    idx=pd.date_range('2021-01-01','2025-01-22',freq='3h',tz='UTC')
    signal=rng.uniform(0,10,len(idx));y=100+20*signal+rng.normal(size=len(idx))
    x=pd.DataFrame(dict(signal=signal,noise=rng.normal(size=len(idx)),unstable=signal*np.where(idx.quarter%2,1,-1)),index=idx)
    labels=pd.Series(y,index=idx);baseline,counts=g.baselines(labels)
    issue=g.b.ISSUES[0];folds=g.windows(idx,issue,baseline)
    for tr,va in folds:
        assert (idx[tr]+pd.Timedelta(hours=72)<idx[va].min()).all()
        assert (idx[va]+pd.Timedelta(hours=72)<issue).all()
    cut=idx[5000];changed=labels.copy();changed.loc[changed.index+pd.Timedelta(hours=72)>=cut]=1e6
    np.testing.assert_allclose(g.baselines(changed)[0][idx<=cut],baseline[idx<=cut],equal_nan=True,rtol=0,atol=1e-10)
    actual,_=g.causal_medians(labels,pd.DatetimeIndex([cut]),days=[180],minimum=48)
    past=labels[(labels.index+pd.Timedelta(hours=72)>=cut-pd.Timedelta(days=180))&(labels.index+pd.Timedelta(hours=72)<cut)]
    np.testing.assert_allclose(actual.rolling_180d.iloc[0],past.median())
    tr=np.flatnonzero((idx+pd.Timedelta(hours=72)<issue)&np.isfinite(baseline))
    cols,audit=g.stable_features(x,y,tr)
    assert 'signal' in cols and 'unstable' not in cols
    broken=x.copy();broken.loc[(idx>=issue-pd.DateOffset(months=3))&(idx<issue),'signal']=np.nan
    assert 'signal' not in g.stable_features(broken,y,tr)[0]
    baseline_errors=np.ones(4)*10
    assert g.gate(np.array([8.,9.,10.,10.1]),[baseline_errors],[1]*4)
    assert not g.gate(np.array([8.,8.,10.1,10.1]),[baseline_errors],[1]*4)
    assert not g.gate(np.array([8.,9.,10.,10.3]),[baseline_errors],[1]*4)
    prediction,w=g.adjust(np.array([101.,120.]),np.array([100.,100.]),np.array([1.,.5]),100.,.5,.05,True)
    np.testing.assert_allclose(prediction,[100.,105.]);np.testing.assert_allclose(w,[0.,.25])
    va=np.flatnonzero((idx>=issue)&(idx<issue+pd.Timedelta(days=30)))
    with tempfile.TemporaryDirectory() as temp,threadpool_limits(limits=2):
        root=Path(temp)
        for family in g.FAMILIES:
            model=g.fit(x,y,baseline,tr,family)
            assert not model.get('unavailable') and 'signal' in model['features']
            pred=g.raw_predict(model,x,baseline,va)
            assert np.isfinite(pred).all()
            if family.startswith('bins'):assert np.isin(pred,model['centers']).all()
            if family=='quantile_map':
                assert pd.Timestamp(model['fit_last'])+pd.Timedelta(hours=72)<pd.Timestamp(model['calibration_start'])
                assert pd.Timestamp(model['calibration_label_end_max'])<issue
                assert pred.min()>=np.quantile(model['outcome_reference'],.05)-1e-8
                assert pred.max()<=np.quantile(model['outcome_reference'],.95)+1e-8
            joblib.dump(model,root/'model.joblib')
            np.testing.assert_allclose(pred,g.raw_predict(joblib.load(root/'model.joblib'),x,baseline,va),rtol=0,atol=1e-10)
            missing=x.copy();missing.loc[idx[va[0]],model['features'][0]]=np.nan
            assert g.dynamic_weight(model,missing,va[:1])[0]==0
            shifted=x.copy();shifted.loc[idx[va[0]],model['features']]=model['high']+1e6
            np.testing.assert_allclose(g.dynamic_weight(model,shifted,va[:1]),[.25])
            missing.loc[(idx>=idx[va[0]]-pd.Timedelta(days=7))&(idx<idx[va[0]]),model['features'][0]]=np.nan
            missing.loc[idx[va[0]],model['features'][0]]=x.loc[idx[va[0]],model['features'][0]]
            assert g.dynamic_weight(model,missing,va[:1])[0]==0
            future=x.copy();future.loc[idx>idx[va[-1]]]=1e6
            np.testing.assert_allclose(g.dynamic_weight(model,x,va),g.dynamic_weight(model,future,va))
        for name in ['a','b','none']:(root/name).mkdir()
        with patch.object(g,'FAMILIES',['residual_linear']):
            choice=g.search(x,y,baseline,folds,root/'a')
            xx=x.copy();xx.loc[idx>=issue]=1e8
            yy=y.copy();yy[idx>=issue]=1e8
            assert choice==g.search(xx,yy,baseline,folds,root/'b')
            zero=g.search(x,np.zeros(len(y)),baseline*0,folds,root/'none')
            assert zero['family']=='quarterly_median'
    # Full runner saves/compares/replays a real synthetic residual model.
    x.attrs['excluded_labels']=pd.DataFrame(columns=['decision_time','label_status'])
    dev=idx[idx>=issue];rows=[]
    for stamp in dev:
        at=g.b.ISSUES[g.b.ISSUES<=stamp][-1]
        med=float(np.median(y[idx+pd.Timedelta(hours=72)<at]))
        rows.append(dict(decision_time=stamp,issue=at,target=runner.TARGET,training_median=med,
            previous_neural=200.,previous_fresh=200.,previous_compact=200.,previous_transformed=200.))
    old=pd.DataFrame(rows);dictionary=pd.DataFrame([dict(feature=c,formula='synthetic') for c in x])
    def load(args):return x,y,dev,old,dictionary,baseline,counts,[]
    def choose(*args):return dict(family='residual_linear',weight=.5,deadzone=.05,dynamic=True,validation_mae=1.)
    with tempfile.TemporaryDirectory() as temp,threadpool_limits(limits=2):
        with patch.object(runner,'load',load),patch.object(g,'search',choose),patch.object(sys,'argv',['runner','--output-dir',temp]):runner.main()
        out=list(Path(temp).iterdir())[0];done=json.loads((out/'COMPLETE.json').read_text())
        assert done['development_origins']==len(dev) and done['replay_passed']
        assert done['predictions_sha256']==g.b.sha(out/'predictions.csv')
        assert len(pd.read_csv(out/'predictions.csv'))==len(dev)
    print('PASS: causal rolling baseline/boundaries; four-window purge; sign/coverage stability; all families and calibration purges; dynamic missingness/shift/deadzone/blend; replay/gates/outer isolation/fallback/full pipeline.')


if __name__=='__main__':main()
