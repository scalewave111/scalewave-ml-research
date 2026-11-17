"""Two original inputs, existing hybrid/MLP engines, purged 1h target."""
import sys,json,subprocess,concurrent.futures,importlib.metadata,msvcrt
from pathlib import Path
import numpy as np
import pandas as pd
import joblib,torch
import eight_feature_search as e
import canonical_return_features as c
s=e.s
OUT=s.HERE/'experiments/two_feature_1h_v1'
CONFIGS={'rf':[{},dict(n_estimators=96,max_depth=4,min_samples_leaf=128,max_samples=2048,criterion='squared_error')], 'mlp':[dict(widths=[8,4],dropout=.05),dict(widths=[4],dropout=.15)]}

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'prepared.joblib').exists():return
    x,_,dev,_=joblib.load(e.OUT/'prepared.joblib');x=x[s.PAIR].copy()
    r=c.load_returns(s.ROOT/'Dataset/canonical_returns_5m.parquet');y=c.make_labels(r,x.index,1);x=x.loc[y.index]
    ref=pd.read_csv(s.ROOT/'Experiments/tree_magnitude_filter_v1/1h/predictions.csv',index_col=0,parse_dates=True)
    assert ref.index.equals(dev) and len(dev)==18481
    np.testing.assert_allclose(ref.truth,y.loc[dev,'magnitude_log_bps'],atol=1e-8,rtol=0)
    assert (y.label_end==y.index+pd.Timedelta(hours=1)).all()
    for origin in x.index[::1000]:
        direct=r[(r.index>origin)&(r.index<=origin+pd.Timedelta(hours=1))]
        assert len(direct)==12 and direct.notna().all()
        assert abs(float(direct.sum())-y.loc[origin,'signed_return_log_bps'])<1e-8
    assert x.index.is_unique and x.index.is_monotonic_increasing and x.index.equals(y.index)
    assert list(x.columns)==s.PAIR and len(s.PAIR)==2
    joblib.dump((x,y,dev,ref),OUT/'prepared.joblib',compress=3)
    s.dump(OUT/'CONTRACT.json',dict(features=s.PAIR,target='Absolute 1h endpoint log return in log bps',configs=CONFIGS,weights=s.WEIGHTS,mlp_tuning_seeds=[48],mlp_final_seeds=[48,149,250],max_epochs=48,selection='Minimum preceding validation MAE; strict label_end < cutoff for inner/outer training and validation. Four expanding quarterly windows.',metrics=['MAE','Spearman IC'],limitation='Previously examined historical evaluation period; no test-driven retries.'))
    fw=Path('C:/Users/Siddh/Desktop/ML/ml_framework')
    paths=[Path(__file__),Path(s.__file__),Path(e.__file__),Path(s.m.__file__),Path(s.q.__file__),Path(s.q.b.__file__),Path(c.__file__),e.OUT/'prepared.joblib',s.ROOT/'Dataset/canonical_returns_5m.parquet',s.ROOT/'Experiments/tree_magnitude_filter_v1/1h/predictions.csv']
    docs=[s.ROOT/'PROJECT_CONTEXT.md',s.HERE/'PROJECT_CONTEXT.md']+[fw/n for n in ['AGENTS.md','CONTEXT.md','docs/INDEX.md','docs/methods/WORKFLOW.md','docs/methods/DATA_AND_METRICS.md','jobs.json']]
    s.dump(OUT/'MANIFEST.json',dict(sources={str(p):s.q.sha(p) for p in paths},instructions={str(p):s.q.sha(p) for p in docs},versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','torch','scikit-learn','joblib']}))
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    s.dump(OUT/'DATA_AUDIT.json',dict(rows=len(x),evaluation_rows=len(dev),source_sha=c.EXPECTED_SHA256,target_parity=True,missing=x.isna().mean().to_dict()))

def worker(method,i):
    torch.set_num_threads(1);x,y,dev,ref=joblib.load(OUT/'prepared.joblib');issue=s.q.STARTS[i]
    inner,ik,vk,ot,te=s.q.windows(y,dev,issue);cols=sorted(s.PAIR);f=OUT/method/f'{issue:%Y%m%d}';f.mkdir(parents=True,exist_ok=True)
    if (f/'DONE.json').exists():return
    trials=[];s.SEEDS=[48]
    for ci,config in enumerate(CONFIGS[method]):
        e.configure(method,config);s.dump(f/'STATUS.json',dict(stage='validation',trial=ci+1,total=2))
        art=s.fit_pack(method,x,y,ik,inner,cols,f/f'c{ci}'/'inner',vk=vk)
        pd.DataFrame(dict(prediction=s.predict(art,x,vk),truth=y.loc[vk,'magnitude_log_bps']),index=vk).to_csv(f/f'c{ci}'/'validation.csv')
        trials.append(dict(id=ci,config=config,**s.metrics(s.predict(art,x,vk),y.loc[vk,'magnitude_log_bps'])))
        s.dump(f/'TRIALS.json',trials)
    choice=min(trials,key=lambda z:(z['mae'],z['id']));s.dump(f/'SELECTION.json',choice);e.configure(method,choice['config'])
    ai=joblib.load(f/f"c{choice['id']}"/'inner/bundle.joblib')
    if method=='mlp':
        s.SEEDS=[48,149,250];s.dump(f/'STATUS.json',dict(stage='selected_configuration_three_seeds'))
        ai=s.fit_pack(method,x,y,ik,inner,cols,f/'confirmation',vk=vk)
    pd.DataFrame(dict(prediction=s.predict(ai,x,vk),truth=y.loc[vk,'magnitude_log_bps']),index=vk).to_csv(f/'validation.csv')
    s.dump(f/'STATUS.json',dict(stage='outer_refit'))
    ao=s.fit_pack(method,x,y,ot,issue,cols,f/'outer',chosen=ai)
    pred=s.predict(ao,x,te);np.testing.assert_allclose(pred,s.predict(joblib.load(f/'outer/bundle.joblib'),x,te),atol=0,rtol=0)
    assert y.loc[ik,'label_end'].max()<inner and y.loc[vk,'label_end'].max()<issue and y.loc[ot,'label_end'].max()<issue
    assert ao['keys'].equals(ot) and ao['cols']==cols
    if method=='rf':assert ao['weight']==ai['weight']
    else:assert ao['epochs']==ai['epochs']
    pd.DataFrame(dict(prediction=pred,truth=y.loc[te,'magnitude_log_bps'],issue=str(issue)),index=te).to_csv(f/'predictions.csv')
    s.dump(f/'DONE.json',dict(status='EXECUTED_AND_REPLAYED',rows=len(te),features=cols,weight=ao.get('weight'),epochs=ao.get('epochs')))
    s.dump(f/'STATUS.json',dict(stage='complete'))

def summarize():
    x,y,dev,ref=joblib.load(OUT/'prepared.joblib');p=pd.DataFrame(dict(truth=ref.truth,rf_baseline=ref.rf_baseline),index=dev)
    selections=[];window_rows=[]
    for method in CONFIGS:
        pieces=[]
        for issue in s.q.STARTS:
            f=OUT/method/f'{issue:%Y%m%d}';assert (f/'DONE.json').exists()
            choice=json.loads((f/'SELECTION.json').read_text());e.configure(method,choice['config'])
            saved=pd.read_csv(f/'predictions.csv',index_col=0,parse_dates=True)
            inner,ik,vk,ot,te=s.q.windows(y,dev,issue)
            assert saved.index.equals(te)
            trials=json.loads((f/'TRIALS.json').read_text())
            assert choice==min(trials,key=lambda z:(z['mae'],z['id']))
            for trial in trials:
                e.configure(method,trial['config'])
                a=joblib.load(f/f"c{trial['id']}"/'inner/bundle.joblib')
                assert a['keys'].equals(ik) and a['cutoff']==inner and a['cols']==sorted(s.PAIR)
                vv=pd.read_csv(f/f"c{trial['id']}"/'validation.csv',index_col=0,parse_dates=True)
                assert vv.index.equals(vk)
                replay=s.predict(a,x,vk)
                np.testing.assert_allclose(replay,vv.prediction,atol=1e-8,rtol=0)
                assert abs(s.metrics(replay,y.loc[vk,'magnitude_log_bps'])['mae']-trial['mae'])<1e-8
            e.configure(method,choice['config'])
            ai=joblib.load(f/('confirmation' if method=='mlp' else f"c{choice['id']}/inner")/'bundle.joblib')
            art=joblib.load(f/'outer/bundle.joblib')
            vv=pd.read_csv(f/'validation.csv',index_col=0,parse_dates=True)
            assert vv.index.equals(vk)
            np.testing.assert_allclose(s.predict(ai,x,vk),vv.prediction,atol=1e-8,rtol=0)
            assert y.loc[vk,'label_end'].max()<issue
            for a,keys,cut in [(ai,ik,inner),(art,ot,issue)]:
                assert a['keys'].equals(keys) and a['cutoff']==cut and a['cols']==sorted(s.PAIR)
                assert y.loc[keys,'label_end'].max()<cut
                if method=='rf':
                    expected=s.q.b.prep(x.loc[keys],a['cols'])
                    for model in a['models'].values():
                        for field in ['median','center','scale']:pd.testing.assert_series_equal(expected[field],model['prep'][field])
                else:
                    expected=s.q.prep(x,keys,a['cols'])
                    for field in ['median','scale']:pd.testing.assert_series_equal(expected[field],a['linear']['prep'][field])
                    assert a['linear']['train_keys'].equals(keys)
                    for neural in a['neural']:assert neural['train_keys'].equals(keys) and neural['cutoff']==cut
            assert art.get('epochs',art.get('weight'))==ai.get('epochs',ai.get('weight'))
            selections.append(dict(model=method,issue=str(issue),config=choice['config'],epochs=art.get('epochs'),weight=art.get('weight')))
            window_rows.append(dict(model=method,issue=str(issue),**s.metrics(saved.prediction,saved.truth)))
            np.testing.assert_allclose(s.predict(art,x,saved.index),saved.prediction,atol=1e-8,rtol=0)
            np.testing.assert_allclose(saved.truth,y.loc[saved.index,'magnitude_log_bps'],atol=1e-8,rtol=0)
            pieces.append(saved.prediction)
        joined=pd.concat(pieces);assert joined.index.equals(dev);p['hybrid' if method=='rf' else method]=joined
    p.to_csv(OUT/'predictions.csv')
    scores=pd.DataFrame([dict(model=n,**s.metrics(p[n],p.truth)) for n in p if n!='truth'])
    baseline=float(scores.loc[scores.model=='rf_baseline','mae'].iloc[0]);assert abs(baseline-25.36740214054159)<1e-8
    scores['mae_change_vs_rf_baseline']=scores.mae-baseline;scores.to_csv(OUT/'scores.csv',index=False)
    monthly=[]
    for month,g in p.groupby(p.index.strftime('%Y-%m')):
        for n in ['rf_baseline','hybrid','mlp']:monthly.append(dict(month=month,model=n,**s.metrics(g[n],g.truth)))
    pd.DataFrame(monthly).to_csv(OUT/'monthly_scores.csv',index=False)
    manifest=json.loads((OUT/'MANIFEST.json').read_text());assert all(s.q.sha(path)==digest for path,digest in manifest['sources'].items())
    s.dump(OUT/'SELECTIONS.json',selections)
    pd.DataFrame(window_rows).to_csv(OUT/'window_scores.csv',index=False)
    s.dump(OUT/'AUDIT.json',dict(status='PASS',rows=len(dev),windows=8,all_predictions_replayed=True,selection_recomputed=True,training_only_preprocessing=True,strict_one_hour_purges=True,sources_unchanged=True))
    s.dump(OUT/'COMPLETE.json',dict(status='EXECUTED_AND_REPLAYED',rows=len(dev),scores=scores.to_dict('records'),features=s.PAIR,limitation='Historical point estimates; previously examined data.'))
    (OUT/'RESULTS.txt').write_text(scores.to_string(index=False));print(scores.to_string(index=False),flush=True)
    with (s.ROOT/'PROJECT_CONTEXT.md').open('a',encoding='utf-8') as f:f.write('\n\n## 1h original-pair hybrid/MLP completed\nSource: Code/experiments/two_feature_1h_v1/scores.csv. Four purged windows, only original two inputs, validation-selected configuration, three final MLP seeds. All saved predictions replayed.\n'+scores.to_string(index=False)+'\n')

def run_locked():
    prepare()
    if (OUT/'COMPLETE.json').exists():print((OUT/'RESULTS.txt').read_text());return
    def launch(job):
        method,i=job
        with (OUT/f'{method}_{i}.log').open('a') as log:result=subprocess.run([sys.executable,__file__,'worker',method,str(i)],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Worker failed {job}; inspect log')
        print('FINISHED',method,i,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(launch,[(m,i) for i in range(4) for m in CONFIGS]))
    summarize()

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'RUN.lock').open('a+b') as lock:
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise SystemExit('Run already active; inspect its logs.')
        run_locked()

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if len(sys.argv)>1 and sys.argv[1]=='worker':worker(sys.argv[2],int(sys.argv[3]))
        elif len(sys.argv)>1 and sys.argv[1]=='prepare':prepare()
        elif len(sys.argv)>1 and sys.argv[1]=='summarize':summarize()
        else:run()
