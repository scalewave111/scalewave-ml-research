"""Synthetic two-target integration check; no real-data fitting."""
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd
import fit_gru_market_up_72h as r
import torch


def main():
    torch.set_num_threads(2)
    rng = np.random.default_rng(72)
    index = pd.date_range('2021-01-01', '2025-01-22', freq='6h', tz='UTC')
    families = ['funding_level','funding_change','basis_level','basis_change','oi_position','oi_unwind',
                'flow_pressure','flow_activity','liq_history','peer_liq_history','liq_breadth']
    cross = ['funding_oi','oi_lowflow','unwind_negative_momentum','basis_funding','liq_volatility','flow_momentum']
    cols = r.models.tree.PRICE + [f'STATE__{f}_{s}__{h}h' for f in families for s in ['mean','deviation'] for h in [24,72]]
    cols += [f'INTERACT__{f}__{h}h' for f in cross for h in [24,72]]
    x = pd.DataFrame(rng.uniform(size=(len(index),len(cols))), index=index, columns=cols)
    x=r.models.augment(x)
    x.attrs['excluded_labels'] = pd.DataFrame(columns=['decision_time'])
    truth = pd.DataFrame({r.b.TARGETS[0]: 20+100*x.iloc[:,0], r.b.TARGETS[1]: 200-100*x.iloc[:,0]}, index=index)
    for config in ['gru8','gru16']:
        with patch.object(r.models,'EPOCHS',3):
            bundle=r.models.fit(x.iloc[:1600],truth.iloc[:1600,0].to_numpy(),dict(model='neural',config=config))
        assert bundle['features']
        p=r.models.predict(bundle,x.iloc[1600:1700])
        assert np.isfinite(p).all() and np.ptp(p)>0
        missing=x.iloc[1600:1602].copy();missing[:]=np.nan
        np.testing.assert_allclose(r.models.predict(bundle,missing),bundle['median'])
    before=r.models.augment(x.drop(columns=[c for c in x if c.startswith('SEQ')]))
    changed=x.drop(columns=[c for c in x if c.startswith('SEQ')]).copy()
    changed.iloc[2000:]=1e6
    pd.testing.assert_frame_equal(before.iloc[:2000],r.models.augment(changed).iloc[:2000])
    dev = index[index >= r.b.ISSUES[0]]
    old = pd.DataFrame([dict(target=target, decision_time=t, actual=truth.loc[t,target],
        training_median=truth.loc[index+pd.Timedelta(hours=72)<r.b.ISSUES[r.b.ISSUES<=t][-1],target].median(),
        previous_neural=50.,previous_best=60.) for target in r.b.TARGETS for t in dev])
    def load(args):
        return x, truth, dev, old, pd.DataFrame({'feature':cols}), []
    def choose(*args):
        return dict(selected=dict(model='median'), candidate=dict(model='neural',config='gru8',weight=.5))
    with tempfile.TemporaryDirectory() as tmp:
        with patch.object(r.models,'EPOCHS',2), patch.object(r,'load',load), patch.object(r.models,'search',choose), patch.object(sys,'argv',['runner','--output-dir',tmp]):
            r.main()
        out = next(Path(tmp).iterdir())
        pred = pd.read_csv(out/'predictions.csv')
        done = json.loads((out/'COMPLETE.json').read_text())
        assert done['predictions_sha256'] == r.b.sha(out/'predictions.csv')
        assert set(pred.target)=={'up_magnitude_log_bps'}
        assert len(pred)==len(dev) and pred.groupby('target').size().eq(len(dev)).all()
        np.testing.assert_allclose(pred.selected,pred.training_median)
        for target, part in pred.groupby('target'):
            np.testing.assert_allclose(part.actual,truth.loc[dev,target])
            assert np.max(np.abs(part.candidate-part.selected))>1
        assert len(list(out.glob('*/20*/model.joblib')))==4
    print('PASS: UP GRU pipeline, coverage, baselines, candidate separation and saved-model replay.')


if __name__=='__main__':
    main()
