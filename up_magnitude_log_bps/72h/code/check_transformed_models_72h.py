"""Synthetic checks only; no real-data model fitting."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import transformed_models_72h as s
import fit_transformed_models_72h as runner


def main():
    rng=np.random.default_rng(48)
    a=np.array([[1.,-4.],[2.,0.],[2.,4.],[3.,np.nan]])
    for mode in s.TRANSFORMS:
        prep=s.fit_transform(a,mode);z=s.transform(a,prep)
        assert np.isfinite(z).all() and z[-1,-1]==1
        if mode=='rank':
            np.testing.assert_allclose(z[:,0],[-.75,0.,0.,.75])
            np.testing.assert_allclose(s.transform(np.array([[-999.,999.]]),prep)[0,:2],[-1.,1.])
        else:
            expected=(a[:,0]-2.)/.5
            if mode=='signed_log':expected=np.sign(expected)*np.log1p(np.abs(expected))
            np.testing.assert_allclose(z[:,0],expected)
        # Inference batches do not fit ranks/scales or depend on other samples.
        np.testing.assert_array_equal(s.transform(a[:1],prep),s.transform(np.vstack([a[:1],[1e9,-1e9]]),prep)[:1])
    args=SimpleNamespace(min_gain=.001,max_window_loss=.02)
    baseline=np.array([10.,10.,10.])
    assert s.gate(np.array([8.,8.,10.1]),baseline,[1,1,1],args)
    assert not s.gate(np.array([8.,8.,10.3]),baseline,[1,1,1],args)
    assert not s.gate(baseline,baseline,[1,1,1],args)
    x=pd.DataFrame(dict(signal=rng.uniform(0,10,3500),noise=rng.normal(size=3500)))
    y=20+8*x.signal.to_numpy()
    folds=[(np.arange(k),np.arange(k,k+200)) for k in [1200,1600,2000]]
    with tempfile.TemporaryDirectory() as temp,threadpool_limits(limits=2):
        root=Path(temp)
        for mode in s.TRANSFORMS:
            for family in s.CONFIGS:
                model=s.fit(x.iloc[:1200].to_numpy(),y[:1200],family+'|'+mode+'|0.5')
                sample=x.iloc[-20:].to_numpy();sample[0]=np.nan
                before=s.predict(model,sample)
                assert np.isfinite(before).all() and (before>=0).all()
                pure=dict(model,weight=1.)
                np.testing.assert_allclose(before,.5*s.predict(pure,sample)+.5*model['median'])
                joblib.dump(model,root/'model.joblib')
                np.testing.assert_array_equal(before,s.predict(joblib.load(root/'model.joblib'),sample))
        for name in ['a','b','zero','missing']:(root/name).mkdir()
        with patch.object(s,'PACKS',{'signal':['signal','noise']}):
            chosen=s.search(x,y,folds,args,root/'a')
            assert chosen['model']!='median'
            changed=x.copy();changed.iloc[2200:]=1e8
            yy=y.copy();yy[2200:]=1e8
            assert chosen==s.search(changed,yy,folds,args,root/'b')
            assert s.search(x,np.zeros(len(y)),folds,args,root/'zero')['model']=='median'
            changed=x.copy();changed['signal']=np.nan
            assert s.search(changed,y,folds,args,root/'missing')['model']=='median'
    dates=pd.date_range('2022-04-22','2025-01-22',freq='6h',tz='UTC')
    for issue in s.b.ISSUES:
        for tr,va in s.b.windows(dates,issue):
            assert (dates[tr]+pd.Timedelta(hours=72)<dates[va].min()).all()
            assert (dates[va]+pd.Timedelta(hours=72)<issue).all()
    x=pd.DataFrame({'signal':rng.uniform(0,10,len(dates))},index=dates)
    x.attrs['excluded_labels']=pd.DataFrame(columns=['decision_time','label_status'])
    truth=pd.DataFrame({t:20+8*x.signal for t in s.b.TARGETS},index=dates)
    dev=dates[dates>=s.b.ISSUES[0]]
    rows=[dict(decision_time=stamp,issue=s.b.ISSUES[s.b.ISSUES<=stamp][-1],target=t,
        actual=truth.loc[stamp,t],selected=50.,previous_fresh=55.,previous_neural=45.,previous_compact=60.) for t in s.b.TARGETS for stamp in dev]
    old=pd.DataFrame(rows);dictionary=pd.DataFrame([dict(feature='signal',formula='synthetic')])
    def fake_load(args):return x,truth,dev,old,dictionary,[]
    def choice(*args):return dict(model='spline_median|rank|0.5',features=['signal'],validation_mae=1.)
    with tempfile.TemporaryDirectory() as temp,threadpool_limits(limits=2):
        with patch.object(runner,'load',fake_load),patch.object(s,'search',choice),patch.object(sys,'argv',['runner','--output-dir',temp]):
            runner.main()
        out=list(Path(temp).iterdir())[0]
        done=json.loads((out/'COMPLETE.json').read_text())
        assert done['replay_passed'] and done['development_origins']==len(dev)
        assert done['predictions_sha256']==s.b.sha(out/'predictions.csv')
        pred=pd.read_csv(out/'predictions.csv')
        assert len(pred)==2*len(dev) and pred.previous_compact.eq(60.).all()
    print('PASS: transform formulas/ties/missingness/batch isolation; all model-transform replays/blends; gates/signal/fallback/outer isolation; 72h purges and full synthetic pipeline.')


if __name__=='__main__':main()
