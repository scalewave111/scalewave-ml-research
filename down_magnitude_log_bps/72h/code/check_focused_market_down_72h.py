"""Synthetic checks for focused fast candidates and guarded policy separation."""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
import focused_market_trees_72h as m
import fit_focused_market_down_72h as runner


def main():
    rng=np.random.default_rng(48)
    families=['funding_level','funding_change','basis_level','basis_change','oi_position','oi_unwind','flow_pressure','flow_activity','liq_history','peer_liq_history','liq_breadth']
    cross=['funding_oi','oi_lowflow','unwind_negative_momentum','basis_funding','liq_volatility','flow_momentum']
    cols=m.PRICE+[f'STATE__{f}_{s}__{h}h' for f in families for s in ['mean','deviation'] for h in [24,72]]+[f'INTERACT__{f}__{h}h' for f in cross for h in [24,72]]
    x=pd.DataFrame(rng.uniform(size=(3500,len(cols))),columns=cols);y=20+100*x.iloc[:,0].to_numpy()
    assert [len(m.pack_columns(x,p)) for p in m.PACKS]==[38,30,62]
    folds=[(np.arange(k),np.arange(k,k+200)) for k in [1200,1600,2000,2400]]
    with tempfile.TemporaryDirectory() as tmp,threadpool_limits(limits=2):
        root=Path(tmp)
        for pack in m.PACKS:
            model=m.fit(x.iloc[:1600],y[:1600],dict(model='focused_hgb',pack=pack))
            assert len(model['features'])==len(m.pack_columns(x,pack))
            sample=x.iloc[-20:].copy();sample.iloc[0]=np.nan
            pred=m.predict(model,sample,.1);assert pred[0]==model['median']
            np.testing.assert_allclose(pred,.1*m.predict(model,sample)+.9*model['median'])
            joblib.dump(model,root/'model.joblib')
            np.testing.assert_allclose(pred,m.predict(joblib.load(root/'model.joblib'),sample,.1),rtol=0,atol=1e-10)
        for name in ['a','b','zero']:(root/name).mkdir()
        with patch.object(m,'PACKS',['combined']):
            result=m.search(x,y,folds,root/'a')
            assert result['candidate']['model']=='focused_hgb'
            xx=x.copy();xx.iloc[2600:]=1e8;yy=y.copy();yy[2600:]=1e8
            assert result==m.search(xx,yy,folds,root/'b')
            assert m.search(x,np.zeros(len(y)),folds,root/'zero')['selected']['model']=='median'
    index=pd.date_range('2021-01-01','2025-01-22',freq='6h',tz='UTC')
    x=pd.DataFrame(rng.uniform(size=(len(index),len(cols))),columns=cols,index=index);y=20+100*x.iloc[:,0].to_numpy()
    x.attrs['excluded_labels']=pd.DataFrame(columns=['decision_time','label_status']);dev=index[index>=m.b.ISSUES[0]]
    old=pd.DataFrame([dict(decision_time=t,training_median=np.median(y[index+pd.Timedelta(hours=72)<m.b.ISSUES[m.b.ISSUES<=t][-1]]),
        previous_neural=50.,previous_transformed=50.,previous_guarded=50.,previous_anchor=50.,previous_fast=50.) for t in dev])
    def load(args):return x,y,dev,old,pd.DataFrame([dict(feature=c,formula='synthetic') for c in cols]),[]
    def choose(*args):return dict(selected=dict(model='median',validation_mae=10.),candidate=dict(model='focused_hgb',pack='combined',weight=.25,validation_mae=11.))
    with tempfile.TemporaryDirectory() as tmp,threadpool_limits(limits=2):
        with patch.object(runner,'load',load),patch.object(m,'search',choose),patch.object(sys,'argv',['runner','--output-dir',tmp]):runner.main()
        out=list(Path(tmp).iterdir())[0];done=json.loads((out/'COMPLETE.json').read_text())
        assert done['replay_passed'] and done['predictions_sha256']==m.b.sha(out/'predictions.csv')
        pred=pd.read_csv(out/'predictions.csv')
        np.testing.assert_allclose(pred.selected,pred.training_median)
        assert np.max(np.abs(pred.candidate-pred.selected))>1.
    print('PASS: focused pack sizes, missingness/blends/replay, earlier-only selection, fallback, separate candidate vs policy and full save pipeline.')


if __name__=='__main__':main()
