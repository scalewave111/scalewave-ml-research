"""Synthetic MLP tests; no real market fitting."""
from pathlib import Path
import tempfile
import sys
import json
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
import mlp_search_120h as m
import fit_mlp_compact_120h as runner


def main():
    m.torch.set_num_threads(2)
    rng=np.random.default_rng(48)
    n=5000
    x=pd.DataFrame({'signal':rng.uniform(-1,1,n),m.SCALE_COLUMN:rng.uniform(1,3,n)})
    x.loc[::17,'signal']=np.nan
    y=100+60*x.signal.fillna(0).to_numpy()
    m.CONFIGS={'test':dict(hidden=[8],epochs=30,decay=.1)}
    m.PACKS={'test':['signal',m.SCALE_COLUMN]}
    folds=[(np.arange(k),np.arange(k,k+400)) for k in [1800,2600,3400]]
    assert m.gate(np.array([9.,9.,10.1]),np.array([10.,10.,10.]),[1,1,1])
    assert not m.gate(np.array([8.,8.,10.3]),np.array([10.,10.,10.]),[1,1,1])
    assert not m.gate(np.array([10.,10.,10.]),np.array([10.,10.,10.]),[1,1,1])
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        for name in ['a','b','zero']:(root/name).mkdir()
        first=m.search(x,y,folds,None,root/'a')
        assert first['model']=='mlp'
        changed=x.copy();changed.iloc[3800:]=1e6
        yy=y.copy();yy[3800:]=1e6
        assert first==m.search(changed,yy,folds,None,root/'b')
        assert m.search(x,np.zeros(n),folds,None,root/'zero')['model']=='median'
        for mode in ['raw','volatility']:
            spec=dict(model='mlp',features=list(x),mode=mode,config='test',weight=.5)
            model=m.fit(x.iloc[:3400],y[:3400],spec)
            p=m.predict(model,x.iloc[3800:])
            assert np.abs(y[3800:]-p).mean()<np.abs(y[3800:]-np.median(y[:3400])).mean()
            joblib.dump(model,root/'model.joblib')
            np.testing.assert_array_equal(p,m.predict(joblib.load(root/'model.joblib'),x.iloc[3800:]))
            bad=x.iloc[3800:].copy();bad[m.SCALE_COLUMN]=np.nan
            assert np.isfinite(m.predict(model,bad)).all()
            factor,_=m.scales(x.iloc[:3400],mode,model['scale_params'])
            reconstructed=(y[:3400]/factor/model['output_unit'])*(factor*model['output_unit']/model['unit'])
            np.testing.assert_allclose(reconstructed,y[:3400]/model['unit'])
        np.testing.assert_array_equal(m.predict({'constant':5.},x),np.full(n,5.))
        dates=pd.date_range('2022-04-22','2025-01-22',freq='6h',tz='UTC')
        for issue in m.b.ISSUES:
            for tr,va in m.b.windows(dates,issue):
                assert (dates[tr]+pd.Timedelta(hours=120)<dates[va].min()).all()
                assert (dates[va]+pd.Timedelta(hours=120)<issue).all()
        xx=x.iloc[:len(dates)].copy();xx.index=dates
        xx.attrs['excluded_labels']=pd.DataFrame(columns=['decision_time','label_status'])
        targets=pd.DataFrame({t:y[:len(dates)] for t in m.b.TARGETS},index=dates)
        dev=dates[dates>=m.b.ISSUES[0]]
        old=pd.DataFrame([dict(decision_time=t,issue=m.b.ISSUES[m.b.ISSUES<=t][-1],target=target,actual=targets.loc[t,target],selected=100.,long_reference=101.) for target in m.b.TARGETS for t in dev])
        dictionary=pd.DataFrame({'feature':list(xx)})
        def load(args):return xx,targets,dev,old,dictionary,[]
        def choose(*args):return dict(model='mlp',features=list(xx),mode='volatility',config='test',weight=.5)
        with patch.object(runner,'load',load),patch.object(m,'search',choose),patch.object(sys,'argv',['runner','--output-dir',str(root/'pipeline')]):
            runner.main()
        output=next((root/'pipeline').iterdir())
        done=json.loads((output/'COMPLETE.json').read_text())
        assert done['replay_passed'] and done['predictions_sha256']==m.b.sha(output/'predictions.csv')
    print('PASS: signal learning, raw/scaled original-unit loss, deterministic serialization, missing scale fallback, gates, future isolation, zero fallback, 120h purges and synthetic full pipeline.')


if __name__=='__main__':main()
