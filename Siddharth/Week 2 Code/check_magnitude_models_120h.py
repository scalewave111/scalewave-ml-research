"""Synthetic model, split, selection and both-target save/replay checks; no real fitting."""
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import fit_magnitude_models_120h as r


def main():
    torch.set_num_threads(2)
    rng = np.random.default_rng(120)
    index = pd.date_range('2022-01-01', '2025-01-22', freq='2h', tz='UTC')
    families = list(r.market_features.SOURCES)
    cross = ['funding_oi','oi_lowflow','unwind_negative_momentum','basis_funding','liq_volatility','flow_momentum']
    cols = r.models.tree.PRICE + [f'STATE__{f}_{s}__{h}h' for f in families for s in ['mean','deviation'] for h in [24,120]]
    cols += [f'INTERACT__{f}__{h}h' for f in cross for h in [24,120]]
    raw = pd.DataFrame(rng.uniform(size=(len(index),len(cols))), index=index, columns=cols)
    x = r.models.augment(raw)
    x.attrs['excluded_labels'] = pd.DataFrame(columns=['decision_time'])
    truth = pd.DataFrame({r.b.TARGETS[0]:20+100*raw.iloc[:,0], r.b.TARGETS[1]:200-100*raw.iloc[:,0]},index=index)
    changed = raw.copy(); changed.iloc[2000:] = 1e6
    pd.testing.assert_frame_equal(x.iloc[:2000],r.models.augment(changed).iloc[:2000])
    for lag in r.models.LAGS:
        c = cols[0]
        np.testing.assert_allclose(x[f'SEQ{lag}__{c}'],raw[c].reindex(index-pd.Timedelta(hours=lag)),equal_nan=True)
    for issue in r.b.ISSUES:
        for tr,va in r.models.windows(index,issue):
            assert (index[tr]+pd.Timedelta(hours=120)<index[va].min()).all()
            assert (index[va]+pd.Timedelta(hours=120)<issue).all()
            boundary=index[va].min()-pd.Timedelta(hours=120)
            assert boundary not in index[tr]
    for config in r.models.CONFIG:
        with patch.object(r.models.neural,'EPOCHS',2):
            model=r.models.fit(x.iloc[20:1620],truth.iloc[20:1620,0].to_numpy(),dict(model='learned',config=config,pack='combined'))
        assert model['features'],config
        pred=r.models.predict(model,x.iloc[1620:1720])
        assert np.isfinite(pred).all() and np.ptp(pred)>0,config
        missing=x.iloc[1620:1622].copy();missing[:]=np.nan
        np.testing.assert_allclose(r.models.predict(model,missing),model['median'])
        np.testing.assert_allclose(r.models.predict(model,x.iloc[1620:1720],.25),.25*pred+.75*model['median'])
    # Exercise the real selection loop with deterministic synthetic forecasts.
    with tempfile.TemporaryDirectory() as tmp:
        folds=r.models.windows(index,r.b.ISSUES[0]);y=truth.iloc[:,0].to_numpy()
        fake_fit=lambda a,y,s:dict(features=[cols[0]],median=float(np.median(y)))
        exact=lambda model,a:20+100*a.iloc[:,0].to_numpy()
        with patch.object(r.models,'fit',fake_fit),patch.object(r.models,'predict',exact):
            first=r.models.search(x,y,folds,Path(tmp))
            altered=y.copy();altered[index>=r.b.ISSUES[0]]=1e9
            second=r.models.search(x,altered,folds,Path(tmp))
            assert first==second and first['selected']['model']=='learned'
        with patch.object(r.models,'fit',fake_fit),patch.object(r.models,'predict',lambda m,a:np.full(len(a),1e9)):
            plan=r.models.search(x,y,folds,Path(tmp))
            assert plan['selected']['model']=='median' and plan['candidate']['model']=='learned'
    dev=index[index>=r.b.ISSUES[0]]
    old=pd.DataFrame([dict(target=target,decision_time=t,actual=truth.loc[t,target],
        training_median=truth.loc[index+pd.Timedelta(hours=120)<r.b.ISSUES[r.b.ISSUES<=t][-1],target].median(),
        selected=50.,previous_long=60.) for target in r.b.TARGETS for t in dev])
    def load(args):return x,truth,dev,old,pd.DataFrame({'feature':cols}),[]
    def choose(*args):return dict(selected=dict(model='median'),candidate=dict(model='learned',config='gru8',pack='flow_liquidation',weight=.5))
    with tempfile.TemporaryDirectory() as tmp:
        with patch.object(r.models.neural,'EPOCHS',1),patch.object(r,'load',load),patch.object(r.models,'search',choose),patch.object(sys,'argv',['runner','--output-dir',tmp]):
            r.main()
        out=next(Path(tmp).iterdir());pred=pd.read_csv(out/'predictions.csv')
        done=json.loads((out/'COMPLETE.json').read_text())
        assert done['predictions_sha256']==r.b.sha(out/'predictions.csv')
        assert len(pred)==2*len(dev) and pred.groupby('target').size().eq(len(dev)).all()
        np.testing.assert_allclose(pred.selected,pred.training_median)
        for target,part in pred.groupby('target'):
            np.testing.assert_allclose(part.actual,truth.loc[dev,target])
            assert np.max(np.abs(part.candidate-part.selected))>0
        assert len(list(out.glob('*/20*/model.joblib')))==8
    print('PASS: all five configurations, 120h purges, causal sequences, selection/fallback, outer isolation, independent targets and saved-model replay.')


if __name__=='__main__':main()
