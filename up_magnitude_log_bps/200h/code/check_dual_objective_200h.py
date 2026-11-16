"""Synthetic dual-objective gates, loss fits, unchanged fallback and save/replay."""
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import fit_dual_objective_200h as r


def check_real_inputs():
    """Exercise actual feature joins and exclusion export, without fitting models."""
    x, truth, dev, old, dictionary, files, anchors = r.load(SimpleNamespace())
    excluded = x.attrs['excluded_labels']
    assert len(excluded) == 608
    assert excluded.label_status.value_counts().to_dict() == {
        'INVALID_PRICE_PATH': 304, 'OUTSIDE_RELEASE': 304}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'excluded_labels.csv'
        excluded.to_csv(path,index=False)
        restored = pd.read_csv(path)
        assert len(restored)==len(excluded) and list(restored)==list(excluded)
        pd.testing.assert_series_equal(restored.label_status,excluded.label_status.reset_index(drop=True))
    assert x.index.equals(truth.index) and len(dev)==17873 and len(anchors)==8
    print('PASS: real input joins preserve all 608 exclusions; CSV export and eight reference replays passed. No training.')


def check_gru_anchors():
    import joblib
    import torch
    torch.set_num_threads(2)
    rng=np.random.default_rng(48)
    index=pd.date_range('2022-01-01',periods=1800,freq='2h',tz='UTC')
    families=['funding_level','funding_change','basis_level','basis_change','oi_position','oi_unwind']
    cross=['funding_oi','oi_lowflow','unwind_negative_momentum','basis_funding']
    cols=r.models.market.PRICE+[f'STATE__{f}_{s}__{h}h' for f in families for s in ['mean','deviation'] for h in [24,200]]
    cols += [f'INTERACT__{f}__{h}h' for f in cross for h in [24,200]]
    raw=pd.DataFrame(rng.uniform(size=(len(index),len(cols))),index=index,columns=cols)
    x=raw.join(pd.DataFrame({f'SEQ{lag}__{c}':raw[c].reindex(index-pd.Timedelta(hours=lag)).to_numpy()
                           for lag in [60,40,20,0] for c in cols},index=index))
    y=20+100*raw.iloc[:,0].to_numpy()
    for config in ['gru8','gru16','median']:
        spec=dict(model='median') if config=='median' else dict(model='learned',config=config,pack='leverage',weight=.25)
        anchor=dict(kind='neural',features=cols,spec=spec)
        with patch.object(r.models.original.neural,'EPOCHS',2):
            fitted=r.models.anchor_refit(anchor,x.iloc[40:1640],y[40:1640])
        expected=r.models.anchor_predict(fitted,x.iloc[1640:])
        assert np.isfinite(expected).all()
        bundle=dict(anchor=fitted,plan=dict(selected=dict(model='anchor')))
        np.testing.assert_array_equal(expected,r.models.predict(bundle,x.iloc[1640:],'selected'))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.joblib';joblib.dump(bundle,path)
            np.testing.assert_array_equal(expected,r.models.predict(joblib.load(path),x.iloc[1640:],'selected'))
    print('PASS: GRU8/GRU16/median anchor refits, exact fallback and serialization.')


def main():
    import torch
    torch.set_num_threads(2)
    rng=np.random.default_rng(200)
    index=pd.date_range('2021-01-01','2025-01-22',freq='6h',tz='UTC')
    x=pd.DataFrame(dict(signal=rng.uniform(size=len(index)),noise=rng.normal(size=len(index))),index=index)
    x.attrs['excluded_labels']=pd.DataFrame(columns=['decision_time','label_status'])
    truth=pd.DataFrame({r.b.TARGETS[0]:20+100*x.signal,r.b.TARGETS[1]:200-100*x.signal},index=index)
    baseline=np.array([[10.,400.]]*4);anchor=np.array([[9.,350.]]*4)
    assert r.models.assess(np.array([[8.,300.]]*4),baseline,anchor,[1]*4)['eligible']
    for bad in [np.array([[8.,450.]]*4),np.array([[11.,200.]]*4),np.array([[9.5,300.]]*4),
                np.array([[1.,10.],[1.,10.],[1.,10.],[11.,500.]])]:
        assert not r.models.assess(bad,baseline,anchor,[1]*4)['eligible']
    # All-row training, both losses, nonnegative predictions, missing-input fallback.
    for config in r.models.CONFIGS:
        pair=r.models.fit_pair(x.iloc[:1600],truth.iloc[:1600,0].to_numpy(),list(x),config)
        assert pair['mean'].loss=='squared_error' and pair['median'].loss=='absolute_error'
        a,b=r.models.components(pair,x.iloc[1600:1700],np.full(100,70.))
        assert np.isfinite(a).all() and np.isfinite(b).all() and np.ptp(a)>0 and np.ptp(b)>0
        missing=x.iloc[:2].copy();missing[:]=np.nan
        for v in r.models.components(pair,missing,np.array([70.,71.])):np.testing.assert_allclose(v,[70.,71.])
        np.testing.assert_allclose(r.models.blend(np.ones(100)*70,a,b,.25,.5),35+.375*a+.125*b)
    folds=r.source.windows(index,r.b.ISSUES[0])
    for tr,va in folds:
        assert (index[tr]+pd.Timedelta(hours=200)<index[va].min()).all()
        assert (index[va]+pd.Timedelta(hours=200)<r.b.ISSUES[0]).all()
    def packs(*args):return {'test':['signal','noise']}
    def fake_fit(*args):return dict(features=['signal'],quality=[])
    def exact(pair,a,old):return (20+100*a.signal.to_numpy(),)*2
    with tempfile.TemporaryDirectory() as tmp:
        with patch.object(r.models,'packs',packs),patch.object(r.models,'fit_pair',fake_fit),patch.object(r.models,'components',exact):
            first=r.models.search(x,truth.iloc[:,0].to_numpy(),folds,dict(kind='tree',features=['signal']),Path(tmp))
            altered=truth.iloc[:,0].to_numpy().copy();altered[index>=r.b.ISSUES[0]]=1e9
            second=r.models.search(x,altered,folds,dict(kind='tree',features=['signal']),Path(tmp))
            assert first==second and first['accepted'] and first['selected']['mean_weight'] in r.models.MEAN_WEIGHTS
        with patch.object(r.models,'packs',packs),patch.object(r.models,'fit_pair',fake_fit),patch.object(r.models,'components',lambda p,a,old:(np.full(len(a),1e8),)*2):
            failed=r.models.search(x,truth.iloc[:,0].to_numpy(),folds,dict(kind='tree',features=['signal']),Path(tmp))
            assert not failed['accepted'] and failed['selected']['model']=='anchor'
    dev=index[index>=r.b.ISSUES[0]];anchors={};controls=[]
    for target in r.b.TARGETS:
        for i,issue in enumerate(r.b.ISSUES):
            tr=np.flatnonzero(index+pd.Timedelta(hours=200)<issue)
            end=r.b.ISSUES[i+1] if i+1<len(r.b.ISSUES) else dev.max()+pd.Timedelta(hours=6)
            te=dev[(dev>=issue)&(dev<end)]
            model=r.b.fit(x.iloc[tr][['signal']].to_numpy(),truth.iloc[tr][target].to_numpy(),'tree')
            anchors[(target,issue)]=dict(kind='tree',model=model,features=['signal'])
            p=r.b.predict(model,x.loc[te,['signal']].to_numpy())
            controls.append(pd.DataFrame(dict(target=target,decision_time=te,actual=truth.loc[te,target],
                selected=p,previous_gru=p,training_median=truth.iloc[tr][target].median())))
    old=pd.concat(controls,ignore_index=True)
    def load(args):return x,truth,dev,old,pd.DataFrame({'feature':list(x)}),[],anchors
    def choose(x,y,folds,cols,folder):
        spec=dict(model='dual',pack='test',config='shallow',features=list(x),mean_weight=.5,update_weight=.5)
        accepted=folder.parent.name.startswith('up_')
        return dict(candidate=spec,selected=spec if accepted else dict(model='anchor',features=cols),accepted=accepted)
    with tempfile.TemporaryDirectory() as tmp:
        with patch.object(r,'load',load),patch.object(r.models,'search',choose),patch.object(sys,'argv',['runner','--output-dir',tmp]):r.main()
        out=next(Path(tmp).iterdir());pred=pd.read_csv(out/'predictions.csv')
        assert len(pred)==2*len(dev) and not pred.duplicated(['target','decision_time']).any()
        assert len(list(out.glob('*/20*/model.joblib')))==8
        features=pd.read_csv(out/'selected_features.csv');assert set(features.feature)==set(x)
        done=json.loads((out/'COMPLETE.json').read_text());assert done['predictions_sha256']==r.b.sha(out/'predictions.csv')
        down=pred[pred.target.str.startswith('down_')];np.testing.assert_allclose(down.selected,down.previous_best)
        checks=pd.read_csv(out/'dual_baseline_checks.csv')
        assert (checks.beats_both==(checks.beats_median_mae&checks.beats_mean_mae&checks.beats_mean_rmse)).all()
    print('PASS: L1/L2 fits, dual gates, 200h purges, selection isolation, missing fallback, unchanged anchor and eight-model both-target pipeline/replay.')


if __name__=='__main__':
    with threadpool_limits(limits=2):
        if '--check-real-inputs' in sys.argv:check_real_inputs()
        elif '--check-gru-anchors' in sys.argv:check_gru_anchors()
        else:
            check_gru_anchors()
            main()
