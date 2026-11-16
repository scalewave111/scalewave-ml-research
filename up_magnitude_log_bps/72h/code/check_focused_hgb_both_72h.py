"""Synthetic two-target integration check; no real-data fitting."""
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd
import fit_focused_hgb_both_72h as r


def main():
    rng = np.random.default_rng(72)
    index = pd.date_range('2021-01-01', '2025-01-22', freq='6h', tz='UTC')
    families = ['funding_level','funding_change','basis_level','basis_change','oi_position','oi_unwind',
                'flow_pressure','flow_activity','liq_history','peer_liq_history','liq_breadth']
    cross = ['funding_oi','oi_lowflow','unwind_negative_momentum','basis_funding','liq_volatility','flow_momentum']
    cols = r.models.PRICE + [f'STATE__{f}_{s}__{h}h' for f in families for s in ['mean','deviation'] for h in [24,72]]
    cols += [f'INTERACT__{f}__{h}h' for f in cross for h in [24,72]]
    x = pd.DataFrame(rng.uniform(size=(len(index),len(cols))), index=index, columns=cols)
    x.attrs['excluded_labels'] = pd.DataFrame(columns=['decision_time'])
    truth = pd.DataFrame({r.b.TARGETS[0]: 20+100*x.iloc[:,0], r.b.TARGETS[1]: 200-100*x.iloc[:,0]}, index=index)
    dev = index[index >= r.b.ISSUES[0]]
    old = pd.DataFrame([dict(target=target, decision_time=t, actual=truth.loc[t,target],
        training_median=truth.loc[index+pd.Timedelta(hours=72)<r.b.ISSUES[r.b.ISSUES<=t][-1],target].median(),
        previous_neural=50.) for target in r.b.TARGETS for t in dev])
    def load(args):
        return x, truth, dev, old, pd.DataFrame({'feature':cols}), []
    def choose(*args):
        return dict(selected=dict(model='median'), candidate=dict(model='focused_hgb',pack='combined',weight=.5))
    with tempfile.TemporaryDirectory() as tmp:
        with patch.object(r,'load',load), patch.object(r.models,'search',choose), patch.object(sys,'argv',['runner','--output-dir',tmp]):
            r.main()
        out = next(Path(tmp).iterdir())
        pred = pd.read_csv(out/'predictions.csv')
        done = json.loads((out/'COMPLETE.json').read_text())
        assert done['predictions_sha256'] == r.b.sha(out/'predictions.csv')
        assert len(pred)==2*len(dev) and pred.groupby('target').size().eq(len(dev)).all()
        np.testing.assert_allclose(pred.selected,pred.training_median)
        for target, part in pred.groupby('target'):
            np.testing.assert_allclose(part.actual,truth.loc[dev,target])
            assert np.max(np.abs(part.candidate-part.selected))>1
        assert len(list(out.glob('*/20*/model.joblib')))==8
    print('PASS: independent UP/DOWN targets, coverage, baselines, candidate separation and saved-model replay.')


if __name__=='__main__':
    main()
