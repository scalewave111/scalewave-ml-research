"""Synthetic search and full save/replay pipeline checks; no real-data fitting."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
import compact_trees_72h as s
import fit_compact_trees_72h as runner


def main():
    rng=np.random.default_rng(48)
    baseline=np.array([10.,10.,10.])
    assert s.accepted(np.array([8.,8.,10.1]),baseline,[1,1,1],.001,.02)
    assert not s.accepted(np.array([8.,8.,10.3]),baseline,[1,1,1],.001,.02)
    assert not s.accepted(np.array([1.,10.1,10.1]),baseline,[1,1,1],.001,.02)
    assert not s.accepted(baseline,baseline,[1,1,1],.001,.02)
    dates=pd.date_range('2021-01-01','2025-01-22',freq='3h',tz='UTC')
    x=pd.DataFrame(dict(signal=rng.uniform(0,10,len(dates)),noise=rng.normal(size=len(dates)),missing=np.nan),index=dates)
    y=20+30*x.signal.to_numpy()
    issue=s.b.ISSUES[-1]
    folds=s.b.windows(dates,issue)
    for tr,va in folds:
        assert (dates[tr]+pd.Timedelta(hours=72)<dates[va].min()).all()
        assert (dates[va]+pd.Timedelta(hours=72)<issue).all()
    args=SimpleNamespace(shortlist=3,max_features=3,screen_rows=1000,min_gain=.001,max_window_loss=.02)
    with tempfile.TemporaryDirectory() as tmp,threadpool_limits(limits=2):
        root=Path(tmp)
        for name in ['a','b','zero','none']:
            (root/name).mkdir()
        chosen=s.search(x,y,folds,args,root/'a')
        assert 'signal' in chosen['features'] and 'missing' not in chosen['features'] and len(chosen['features'])<=3
        changed, yy=x.copy(),y.copy()
        changed.loc[dates>=issue]=99999.
        yy[dates>=issue]=99999.
        assert chosen==s.search(changed,yy,folds,args,root/'b')
        assert s.search(x,np.zeros(len(y)),folds,args,root/'zero')['model']=='median'
        assert s.search(x[['missing']],y,folds,args,root/'none')['model']=='median'
        for kind in [c+'@0.5' for c in s.CONFIGS]:
            tr=folds[0][0]
            a=x[['signal','noise']].to_numpy()
            model=s.fit(a[tr],y[tr],kind)
            sample=a[-20:].copy(); sample[0]=np.nan
            before=s.predict(model,sample)
            assert np.isfinite(before).all() and (before>=0).all()
            assert model['model'].max_depth==s.CONFIGS[kind.split('@')[0]]['max_depth']
            np.testing.assert_allclose(before,.5*s.b.predict(model,sample)+.5*model['median'])
            joblib.dump(model,root/'model.joblib')
            np.testing.assert_allclose(before,s.predict(joblib.load(root/'model.joblib'),sample),rtol=0,atol=0)
    # Exercise every quarterly save, scoring, hashing and completion path.
    x=x[['signal']].copy()
    x.attrs['excluded_labels']=pd.DataFrame(columns=['decision_time','label_status'])
    truths=pd.DataFrame({t:y for t in runner.b.TARGETS},index=dates)
    dev=dates[dates>=runner.b.ISSUES[0]]
    rows=[]
    for target in runner.b.TARGETS:
        for stamp in dev:
            rows.append(dict(decision_time=stamp,issue=runner.b.ISSUES[runner.b.ISSUES<=stamp][-1],
                target=target,actual=truths.loc[stamp,target],selected=50.,previous_fresh=55.,previous_neural=45.))
    old=pd.DataFrame(rows)
    dictionary=pd.DataFrame([dict(feature='signal',definition='synthetic')])
    def fake_load(args):
        return x,truths,dev,old,dictionary,[]
    def choose(*args):
        return dict(model='depth2@0.5',features=['signal'],validation_mae=1.)
    with tempfile.TemporaryDirectory() as tmp:
        with patch.object(runner,'load',fake_load),patch.object(s,'search',choose),patch.object(sys,'argv',['runner','--output-dir',tmp]):
            runner.main()
        folders=list(Path(tmp).iterdir()); assert len(folders)==1
        done=json.loads((folders[0]/'COMPLETE.json').read_text())
        assert done['replay_passed'] and done['development_origins']==len(dev)
        assert done['predictions_sha256']==runner.b.sha(folders[0]/'predictions.csv')
        pred=pd.read_csv(folders[0]/'predictions.csv')
        assert len(pred)==2*len(dev) and pred.previous_fresh.eq(55.).all()
    # Create a genuine synthetic source-run bundle for inference-only diagnostics.
    import discover_long_features_72h as original
    import diagnose_fresh_features_72h as diagnostic
    def source_choice(*args):
        return dict(model='tree',features=['signal'],validation_mae=1.,baseline_mae=2.)
    with tempfile.TemporaryDirectory() as tmp,threadpool_limits(limits=2):
        with patch.object(original,'load',fake_load),patch.object(original.b,'search',source_choice),patch.object(sys,'argv',['runner','--output-dir',tmp]):
            original.main()
        run=list(Path(tmp).iterdir())[0]
        with patch.object(original.b,'fit',diagnostic.no_fit),patch.object(original.b,'search',diagnostic.no_fit):
            tables=diagnostic.analyze(run,x,truths,dev,old)
            assert not tables['feature_shifts'].empty
            assert len(tables['period_diagnostics'])==60
            (run/'predictions.csv').write_text('corrupted')
            try:
                diagnostic.analyze(run,x,truths,dev,old)
                raise AssertionError('corruption accepted')
            except ValueError as exc:
                assert 'hash' in str(exc)
    print('PASS: stability gate, purge, fresh selection, missing/zero fallback, outer isolation, model configs, serialization and full synthetic pipeline')


if __name__=='__main__':
    main()
