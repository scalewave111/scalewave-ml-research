"""One fixed four-feature challenger. Original quarterly walk-forward recipe."""
from pathlib import Path
import json,time,importlib.metadata,subprocess,sys
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
import baseline_single_feature as s
q=s.q
HERE=Path(__file__).resolve().parent
OUT=HERE/'experiments/vol_ratio_plus_level_v1'
SRC=s.OUT
COLS=q.b.PAIR+['PRICE__vol_ratio_4h_24h','PRICE__realized_vol_24h']

def run():
    start=time.monotonic();OUT.mkdir(exist_ok=True)
    if (OUT/'COMPLETE.json').exists():
        print((OUT/'RESULTS.txt').read_text());return
    contract=dict(features=COLS,primary='ratio_plus_level',target='BTCUSDT absolute 48-hour endpoint log-return magnitude in log bps',
        method='Original median-regression plus RF, same parameters and preprocessing including missingness flags. Extend both components to four source features.',
        selection='Same prior purged quarter MAE selects RF weight from [0,.1,.25,.5]. No feature search, calibration, or test-driven retries.',
        reason='24h volatility is the absolute level corresponding to the denominator of the existing 4h/24h ratio; chosen before this run.',
        evaluation='Same four expanding outer windows and 18481 already-examined historical rows; no fresh holdout.',
        source_sha=q.sha(__file__),helper_sha=q.sha(s.__file__))
    if (OUT/'CONTRACT.json').exists():assert json.loads((OUT/'CONTRACT.json').read_text())==contract
    else:
        q.dump(OUT/'CONTRACT.json',contract);(OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    x,y,dev,ref=joblib.load(SRC/'prepared.joblib')
    old=pd.read_csv(SRC/'predictions.csv',index_col=0,parse_dates=True)
    val=pd.read_csv(SRC/'validation_predictions.csv',index_col=0,parse_dates=True)
    assert old.index.equals(dev) and len(dev)==18481 and x.index.equals(y.index)
    np.testing.assert_allclose(y.magnitude_log_bps,abs(y.signed_return_log_bps),rtol=0,atol=1e-8)
    np.testing.assert_allclose(old.baseline,ref.report_hybrid,rtol=0,atol=1e-8)
    # Verify cached feature provenance and the completed causal audit.
    manifest=json.loads((SRC/'MANIFEST.json').read_text())
    assert json.loads((SRC/'AUDIT.json').read_text())['canonical_feature_causality']
    paths=[SRC/n for n in ['prepared.joblib','predictions.csv','validation_predictions.csv','MANIFEST.json','FEATURE_AUDIT.json']]
    hashes={str(p):q.sha(p) for p in paths}
    templates={k:q.b.BASE/'models'/f'2024-01__report_outer_{k}__expanding.joblib' for k in ['pair','rf']}
    prototypes={k:joblib.load(p)['model'] for k,p in templates.items()}
    fw=Path('C:/Users/Siddh/Desktop/ML/ml_framework')
    docs=[HERE/'PROJECT_CONTEXT.md',fw.parent/'AGENTS.md']+[fw/n for n in ['AGENTS.md','CONTEXT.md','docs/INDEX.md','docs/methods/WORKFLOW.md','docs/methods/DATA_AND_METRICS.md','docs/findings/RECORDED_RESULTS.md','jobs.json']]
    if not (OUT/'MANIFEST.json').exists():q.dump(OUT/'MANIFEST.json',dict(inputs=hashes,templates={str(p):q.sha(p) for p in templates.values()},instructions={str(p):q.sha(p) for p in docs},versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','joblib']},feature_provenance=manifest))
    pieces=[];validations=[];training=[];decisions=[];count=0
    for issue in q.STARTS:
        folder=OUT/issue.strftime('%Y%m%d');folder.mkdir(exist_ok=True)
        inner,ik,vk,ot,te=q.windows(y,dev,issue)
        forecasts={}
        for phase,keys,cut,pk in [('inner',ik,inner,vk),('outer',ot,issue,te)]:
            for kind in ['pair','rf']:
                q.dump(OUT/'STATUS.json',dict(stage='fitting',issue=issue,phase=phase,kind=kind,completed_fits=count))
                a=s.fit(x,y,keys,cut,COLS,kind,folder/f'{kind}_{phase}.joblib',prototypes[kind])
                assert a['train_keys'].equals(keys) and a['issuance']==cut and y.loc[keys,'label_end'].max()<cut
                expected=s.prepare(x,keys,COLS)
                assert a['prep']['columns']==COLS
                for field in ['median','center','scale']:pd.testing.assert_series_equal(a['prep'][field],expected[field])
                for k,v in prototypes[kind].get_params().items():
                    if k!='n_jobs':assert a['model'].get_params()[k]==v
                forecasts[phase,kind]=q.b.predict(a,x.loc[pk])
                if phase=='outer':forecasts['train',kind]=q.b.predict(a,x.loc[ot])
                count+=1;print('FIT',count,'of 16',issue.date(),phase,kind,flush=True)
            if phase=='inner':
                truth=y.loc[vk,'magnitude_log_bps'].to_numpy()
                w=s.choose_weight(forecasts['inner','pair'],forecasts['inner','rf'],truth)
                decision=dict(issue=str(issue),weight=w,validation_rows=len(vk),train_rows=len(ot),test_rows=len(te))
                q.dump(folder/'selection.json',decision) # freeze before outer fitting
        decisions.append(decision)
        vv=val.loc[pd.to_datetime(val.issue,utc=True)==issue,['baseline','vol_ratio','truth','issue']].copy();assert vv.index.equals(vk)
        pp=old.loc[te,['baseline','vol_ratio','truth','issue']].copy()
        for df,phase in [(vv,'inner'),(pp,'outer')]:df['ratio_plus_level']=(1-w)*forecasts[phase,'pair']+w*forecasts[phase,'rf']
        training.append(pd.DataFrame(dict(ratio_plus_level=(1-w)*forecasts['train','pair']+w*forecasts['train','rf'],truth=y.loc[ot,'magnitude_log_bps'],issue=str(issue)),index=ot))
        pieces.append(pp);validations.append(vv)
    p=pd.concat(pieces);v=pd.concat(validations);t=pd.concat(training)
    assert p.index.equals(dev)
    p.to_csv(OUT/'predictions.csv',index_label='decision_time');v.to_csv(OUT/'validation_predictions.csv',index_label='decision_time')
    q.dump(OUT/'SELECTIONS.json',decisions)
    names=['baseline','vol_ratio','ratio_plus_level']
    scores=pd.DataFrame([dict(split=split,model=n,**q.b.metric(df[n],df.truth)) for split,df in [('validation',v),('test',p)] for n in names]+[dict(split='training_in_sample',model='ratio_plus_level',**q.b.metric(t.ratio_plus_level,t.truth))])
    scores.to_csv(OUT/'scores.csv',index=False)
    windows=pd.DataFrame([dict(issue=i,model=n,**q.b.metric(df[n],df.truth)) for i,df in p.groupby('issue') for n in names]);windows.to_csv(OUT/'window_scores.csv',index=False)
    extreme=pd.DataFrame([dict(model=n,**q.b.metric(p.loc[p.truth>500,n],p.loc[p.truth>500,'truth'])) for n in names]);extreme.to_csv(OUT/'extreme_scores.csv',index=False)
    uncertainty=[]
    for reference in ['baseline','vol_ratio']:
        delta=abs(p.ratio_plus_level-p.truth)-abs(p[reference]-p.truth)
        block=(p.index.asi8-p.index.asi8.min())//pd.Timedelta(days=7).value
        b=pd.DataFrame(dict(delta=delta.to_numpy(),block=block)).groupby('block').delta.agg(['sum','count'])
        ids=np.random.default_rng(20261104).integers(0,len(b),(5000,len(b)))
        boot=b['sum'].to_numpy()[ids].sum(1)/b['count'].to_numpy()[ids].sum(1)
        lo,hi=np.quantile(boot,[.025,.975]);uncertainty.append(dict(reference=reference,mae_difference=delta.mean(),lower=lo,upper=hi))
    pd.DataFrame(uncertainty).to_csv(OUT/'uncertainty.csv',index=False)
    # Independent checkpoint replay, including validation weight choice.
    saved=pd.read_csv(OUT/'predictions.csv',index_col=0,parse_dates=True)
    for d in decisions:
        issue=pd.Timestamp(d['issue']);inner,ik,vk,ot,te=q.windows(y,dev,issue);folder=OUT/issue.strftime('%Y%m%d')
        vp={k:q.b.predict(joblib.load(folder/f'{k}_inner.joblib'),x.loc[vk]) for k in ['pair','rf']}
        assert s.choose_weight(vp['pair'],vp['rf'],y.loc[vk,'magnitude_log_bps'].to_numpy())==d['weight']
        op={k:q.b.predict(joblib.load(folder/f'{k}_outer.joblib'),x.loc[te]) for k in ['pair','rf']}
        np.testing.assert_allclose((1-d['weight'])*op['pair']+d['weight']*op['rf'],saved.loc[te,'ratio_plus_level'],rtol=0,atol=1e-8)
    assert all(q.sha(Path(path))==h for path,h in hashes.items())
    audit=dict(status='PASS',fits=count,rows=len(p),four_source_features=True,strict_label_purges=True,training_only_preprocessing=True,unchanged_model_parameters=True,baseline_reproduced=True,selection_and_checkpoint_replay=True,source_inputs_unchanged=True)
    q.dump(OUT/'AUDIT.json',audit)
    report='Four-feature fixed challenger: liquidation pair + volatility ratio + 24h realized volatility\n\n'+scores.to_string(index=False)+'\n\nWindow scores:\n'+windows.to_string(index=False)+'\n\nExtreme (>500 bps) scores:\n'+extreme.to_string(index=False)+'\n\nDescriptive weekly block bootstrap (unadjusted for previous searches):\n'+pd.DataFrame(uncertainty).to_string(index=False)+'\n\nTraining is pooled in-sample across expanding windows with repeated timestamps. Historical evaluation has already been examined; no fresh holdout. No calibration applied.\n'
    (OUT/'RESULTS.txt').write_text(report,encoding='utf-8')
    q.dump(OUT/'COMPLETE.json',dict(status='EXECUTED_AND_AUDITED',audit=audit,seconds=time.monotonic()-start,scores=scores.to_dict('records')))
    q.dump(OUT/'STATUS.json',dict(stage='complete'))
    print(report,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
