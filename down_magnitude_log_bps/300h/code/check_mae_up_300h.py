"""Synthetic model, split, selection and both-target save/replay checks; no real fitting."""
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import fit_mae_up_300h as r


def main():
    torch.set_num_threads(2)
    r.models.configure_up()
    assert len(r.models.CONFIG) == 3 and len(r.models.PACKS) == 2
    assert r.models.EPOCHS == 12
    # MAE-only accepts RMSE loss, but still requires MAE gain against BOTH constants.
    baseline = np.tile(np.array([[10., 144.], [11., 100.]]), (4, 1, 1))
    assert r.models.assess(np.tile([9., 90.], (4, 1)), baseline, [1]*4)['eligible']
    assert r.models.assess(np.tile([9., 110.], (4, 1)), baseline, [1]*4)['eligible']
    assert not r.models.assess(np.tile([10.5, 90.], (4, 1)), baseline, [1]*4)['eligible']
    reverse = baseline[:, ::-1].copy()
    assert r.models.assess(np.tile([9., 90.], (4, 1)), reverse, [1]*4)['eligible']
    rng = np.random.default_rng(200)
    index = pd.date_range('2022-01-01', '2025-01-22', freq='2h', tz='UTC')
    families = list(r.market_features.SOURCES)
    cross = ['funding_oi','oi_lowflow','unwind_negative_momentum','basis_funding','liq_volatility','flow_momentum']
    cols = r.models.tree.PRICE + [f'STATE__{f}_{s}__{h}h' for f in families for s in ['mean','deviation'] for h in [24,300]]
    cols += [f'INTERACT__{f}__{h}h' for f in cross for h in [24,300]]
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
            assert (index[tr]+pd.Timedelta(hours=300)<index[va].min()).all()
            assert (index[va]+pd.Timedelta(hours=300)<issue).all()
            boundary=index[va].min()-pd.Timedelta(hours=300)
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
        # Mean anchoring must agree with validation blending, including missing inputs.
        model['anchor'] = float(truth.iloc[20:1620,0].mean())
        np.testing.assert_allclose(r.models.predict(model,x.iloc[1620:1720],.25),.25*pred+.75*model['anchor'])
        np.testing.assert_allclose(r.models.predict(model,missing,.25),model['anchor'])
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
        training_median=truth.loc[index+pd.Timedelta(hours=300)<r.b.ISSUES[r.b.ISSUES<=t][-1],target].median(),
        selected=50.,previous_long=60.,previous_fast_selected=70.,previous_fast_candidate=80.) for target in r.b.TARGETS for t in dev])
    def load(args):return x,truth,dev,old,pd.DataFrame({'feature':cols}),[]
    def choose(*args):return dict(selected=dict(model='median'),candidate=dict(model='learned',config='gru8_l1',pack='combined',weight=.5))
    with tempfile.TemporaryDirectory() as tmp:
        with patch.object(r.models.neural,'EPOCHS',1),patch.object(r,'load',load),patch.object(r.models,'search',choose),patch.object(sys,'argv',['runner','--output-dir',tmp]):
            r.main()
        out=next(Path(tmp).iterdir());pred=pd.read_csv(out/'predictions.csv')
        done=json.loads((out/'COMPLETE.json').read_text())
        contract=json.loads((out/'CONTRACT.json').read_text())
        assert contract['targets'] == ['up_magnitude_log_bps']
        assert contract['estimator_row_caps']['hgb'] == r.models.tree.CAP
        assert len(contract['config']) == 3
        assert done['predictions_sha256']==r.b.sha(out/'predictions.csv')
        assert len(pred)==len(dev) and pred.groupby('target').size().eq(len(dev)).all()
        np.testing.assert_allclose(pred.selected,pred.training_median)
        for target,part in pred.groupby('target'):
            np.testing.assert_allclose(part.actual,truth.loc[dev,target])
            assert np.max(np.abs(part.candidate-part.selected))>0
        assert len(list(out.glob('*/20*/model.joblib')))==4
        checks=pd.read_csv(out/'dual_baseline_checks.csv')
        assert not checks.loc[checks.model.eq('selected'),'beats_both_on_both'].any()
    print(f'PASS: {len(r.models.CONFIG)} loss/configuration combinations, 300h purges, causal sequences, selection/fallback, outer isolation, independent targets and saved-model replay.')


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        main()
