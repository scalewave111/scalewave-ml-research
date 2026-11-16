"""Synthetic end-to-end test; never reads market data."""
import argparse
import contextlib
import io
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
import diagnose_long_features_200h as d


def main():
    assert d.describe([1,3],[1,3],2)['mae_gain_vs_median']==1
    assert list(d.bins([0,1,2,np.nan],0,1))==['low','medium','high','missing']
    assert d.correlation([1,2,3],[3,2,1])==-1
    assert d.correlation([1,1,1],[3,2,1]) is None
    assert d.spacing_count(pd.date_range('2020-01-01',periods=3,freq='200h',tz='UTC'))==2
    try:
        d.no_fit()
        raise AssertionError('No-fit guard failed')
    except RuntimeError:
        pass
    rng=np.random.default_rng(48)
    index=pd.date_range('2021-01-01','2025-01-20',freq='6h',tz='UTC')
    x=pd.DataFrame({'LONGPRICE__rv__200h':rng.uniform(1,10,len(index))},index=index)
    y=pd.DataFrame({t:10+3*x.iloc[:,0]+rng.normal(size=len(x)) for t in d.base.TARGETS},index=index)
    dev=index[index>=d.base.ISSUES[0]]
    original_load=d.source.load
    original_fit,original_search=d.base.fit,d.base.search
    argv=sys.argv[:]
    try:
        with tempfile.TemporaryDirectory() as temp,threadpool_limits(limits=2):
            root=Path(temp);run=root/'run';run.mkdir()
            frames,cards=[],[]
            for i,issue in enumerate(d.base.ISSUES):
                end=d.base.ISSUES[i+1] if i<3 else dev.max()+pd.Timedelta(hours=6)
                te=dev[(dev>=issue)&(dev<end)]
                tr=np.flatnonzero(index+pd.Timedelta(hours=200)<issue)
                for target in d.base.TARGETS:
                    folder=run/f'{issue:%Y-%m}_{target}';folder.mkdir()
                    truth=y[target].to_numpy()
                    choice=dict(model='ridge',features=list(x.columns),validation_window_mae=[0. if i%2 else 1e9]*3)
                    policy=d.source.baseline_gate(choice,truth,d.base.windows(index,issue))
                    mean,median=truth[tr].mean(),np.median(truth[tr])
                    constant=mean if policy['fallback']=='mean' else median
                    model=original_fit(x.iloc[tr].to_numpy(),truth[tr],'ridge')
                    raw=d.base.predict(model,x.loc[te].to_numpy())
                    bundle=dict(model=model,features=list(x.columns),policy=policy,constant=constant,issue=issue,target=target)
                    joblib.dump(bundle,folder/'model.joblib')
                    for name,obj in [('MODEL_HASH',dict(sha256=d.base.sha(folder/'model.joblib'))),('selection',choice),('policy',policy)]:
                        d.base.dump(folder/f'{name}.json',obj)
                    cards.append(dict(issue=str(issue),target=target,search=choice,policy=policy))
                    frames.append(pd.DataFrame(dict(decision_time=te,issue=issue,target=target,actual=y.loc[te,target].to_numpy(),
                        candidate=raw,selected=raw if policy['accepted'] else constant,previous=median,training_mean=mean,training_median=median)))
            pred=pd.concat(frames,ignore_index=True)
            pred['period']=pred.decision_time.dt.tz_localize(None).dt.to_period('Q').astype(str)
            pred.loc[pred.decision_time>=pd.Timestamp('2025-01-01',tz='UTC'),'period']='2025-01'
            pred.to_csv(run/'predictions.csv',index=False)
            scores=[]
            for target,g in pred.groupby('target'):
                for period,f in [('ALL',g)]+list(g.groupby('period')):
                    for model in d.MODELS:
                        scores.append(dict(target=target,period=period,model=model,**d.base.metric(f.actual.to_numpy(),f[model].to_numpy())))
            pd.DataFrame(scores).to_csv(run/'metrics.csv',index=False)
            d.base.dump(run/'selections.json',cards)
            d.base.dump(run/'COMPLETE.json',dict(status='COMPLETE',replay_passed=True,development_origins=len(dev),predictions_sha256=d.base.sha(run/'predictions.csv')))
            d.base.dump(run/'CONTRACT.json',dict(horizon_minutes=12000,input_hashes={},arguments={k:str(root) for k in ['data_dir','y_dir','returns','reference_run']}))
            old=pred.copy();old['selected']=pred.previous
            d.source.load=lambda args:(x,y,dev,old,[],None,[])
            sys.argv=['diagnose','--run-dir',str(run),'--output-dir',str(root/'out')]
            with contextlib.redirect_stdout(io.StringIO()):
                d.main()
            complete=list((root/'out').glob('*/COMPLETE.json'))
            assert len(complete)==1 and d.read(complete[0])['metrics_reproduced']
            assert len(pd.read_csv(complete[0].parent/'feature_relationships.csv'))==8
            # Corruption must stop the run before producing another completion.
            (run/'predictions.csv').write_text('corrupted',encoding='utf-8')
            try:
                d.main()
                raise AssertionError('Corruption accepted')
            except ValueError as exc:
                assert 'hash mismatch' in str(exc)
    finally:
        d.source.load=original_load
        d.base.fit,d.base.search=original_fit,original_search
        sys.argv=argv
    print('PASS: formulas, groups, overlap counts, no-fit guard, complete synthetic inference pipeline, metrics, model replay, corruption rejection')


if __name__=='__main__':
    main()
