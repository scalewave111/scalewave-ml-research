"""Five-input 4h hybrid and SiLU MLP using the original pair plus three additions."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
import sys,json,subprocess,concurrent.futures,importlib.metadata,msvcrt
from pathlib import Path
import numpy as np
import pandas as pd
import joblib,torch
import four_feature_4h as b
s=b.s;e=b.e
OUT=s.HERE/'experiments/five_feature_4h_v1'
PREVIOUS=b.OUT
PACKS={'five':sorted(s.PAIR+['PRICE__vol_ratio_4h_24h','F4Z__liquidation__venue_breadth_percentile__1h__value','PRICE__realized_vol_168h'])}
CONFIGS={'rf':['original'],'mlp':[dict(widths=[8,4],dropout=.05)]}

def setup(pack):
    b.OUT=OUT/pack;b.COLS=PACKS[pack];b.CONFIGS=CONFIGS

def load():return joblib.load(b.old.OUT/'prepared.joblib')

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'MANIFEST.json').exists():
        manifest=json.loads((OUT/'MANIFEST.json').read_text())
        assert all(s.q.sha(p)==v for p,v in manifest['sources'].items());return
    x,y,dev,base=load()
    assert len(PACKS['five'])==5 and len(dev)==18481
    assert x.index.equals(y.index) and (y.label_end==y.index+pd.Timedelta(hours=4)).all()
    np.testing.assert_allclose(y.loc[dev,'magnitude_log_bps'],base.truth,atol=1e-8,rtol=0)
    checks=[]
    for issue in s.q.STARTS:
        inner,ik,vk,ot,te=s.q.windows(y,dev,issue)
        assert y.loc[ik,'label_end'].max()<inner and y.loc[vk,'label_end'].max()<issue and y.loc[ot,'label_end'].max()<issue
        checks.append(dict(issue=str(issue),inner_rows=len(ik),outer_rows=len(ot),validation_rows=len(vk),evaluation_rows=len(te)))
    s.dump(OUT/'CONTRACT.json',dict(packs=PACKS,target='4h absolute endpoint log return in log bps',configs=CONFIGS,mlp_seeds=[48,149,250],activation='SiLU',
        protocol='Fixed original RF recipe with preceding-validation blend choice; fixed8/4 SiLU dropout.05 MLP with prior-validation early stopping and three-seed mean. Same optimizer/preprocessing as four-feature default. No architecture or RF parameter search.',
        comparison='Five inputs: original liquidation pair plus volatility ratio, venue breadth1h and realized volatility168h. Compare saved four-feature defaults and RF baseline on identical18481 rows.',
        limitation='Previously exposed historical data; exploratory comparison, not independently confirmed improvement.',windows=checks))
    paths=[Path(__file__),Path(b.__file__),Path(b.old.__file__),Path(b.h.__file__),Path(e.__file__),Path(s.__file__),Path(s.m.__file__),Path(s.q.__file__),Path(s.q.b.__file__),b.old.OUT/'prepared.joblib',PREVIOUS/'predictions.csv']
    fw=Path('C:/Users/Siddh/Desktop/ML/ml_framework')
    docs=[s.ROOT/'PROJECT_CONTEXT.md',s.HERE/'PROJECT_CONTEXT.md']+[fw/n for n in ['AGENTS.md','CONTEXT.md','docs/INDEX.md','docs/methods/WORKFLOW.md','docs/methods/DATA_AND_METRICS.md','docs/findings/RECORDED_RESULTS.md','jobs.json']]
    s.dump(OUT/'MANIFEST.json',dict(sources={str(p):s.q.sha(p) for p in paths},instructions={str(p):s.q.sha(p) for p in docs},versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','torch','scikit-learn','joblib']}))
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    s.dump(OUT/'STATUS.json',dict(stage='prepared',jobs=8))

def worker(pack,method,i):
    setup(pack)
    b.worker(method,i)

def summarize():
    x,y,dev,base=load();previous=pd.read_csv(PREVIOUS/'predictions.csv',index_col=0,parse_dates=True)
    assert previous.index.equals(dev)
    p=pd.DataFrame(dict(truth=base.truth,rf_baseline=base.rf_baseline,hybrid_four=previous.hybrid_four_default,mlp_four=previous.mlp_four_default))
    choices=[];window_rows=[];maxerror=0.
    for pack,cols in PACKS.items():
        setup(pack)
        for method in CONFIGS:
            model='hybrid' if method=='rf' else 'mlp';pieces=[]
            for issue in s.q.STARTS:
                inner,ik,vk,ot,te=s.q.windows(y,dev,issue)
                f=OUT/pack/method/f'{issue:%Y%m%d}'
                done=json.loads((f/'DONE.json').read_text());choice=json.loads((f/'SELECTION.json').read_text())
                assert choice['selected_id']==0 and not choice['change_accepted']
                b.configure(method,0)
                a=joblib.load(done['outerpaths']['0']);b.check(a,method,x,y,ot,issue)
                ai=joblib.load(choice['innerpaths']['0']);b.check(ai,method,x,y,ik,inner,vk)
                if method=='mlp':
                    assert a['epochs']==ai['epochs'] and [n['seed'] for n in a['neural']]==[48,149,250]
                else:assert a['weight']==ai['weight']
                z=pd.read_csv(f/'predictions.csv',index_col=0,parse_dates=True)
                assert z.index.equals(te)
                replay=s.predict(a,x,te);maxerror=max(maxerror,float(np.max(abs(replay-z.default))))
                np.testing.assert_allclose(replay,z.default,atol=1e-8,rtol=0)
                np.testing.assert_allclose(z.default,z.selected,atol=0,rtol=0)
                np.testing.assert_allclose(z.truth,y.loc[te,'magnitude_log_bps'],atol=1e-8,rtol=0)
                v=pd.read_csv(f/'validation_predictions.csv',index_col=0,parse_dates=True)
                assert v.index.equals(vk)
                np.testing.assert_allclose(s.predict(ai,x,vk),v.default,atol=1e-8,rtol=0)
                choices.append(dict(pack=pack,model=model,issue=str(issue),features=cols,epochs=a.get('epochs'),blend=a.get('weight')))
                window_rows.append(dict(pack=pack,model=model,issue=str(issue),**s.metrics(z.default,z.truth)))
                pieces.append(z.default)
            pred=pd.concat(pieces);assert pred.index.equals(dev);p[f'{model}_{pack}']=pred
    scores=pd.DataFrame([dict(model=n,**s.metrics(p[n],p.truth)) for n in p if n!='truth'])
    scores.to_csv(OUT/'scores.csv',index=False);p.to_csv(OUT/'predictions.csv');pd.DataFrame(window_rows).to_csv(OUT/'window_scores.csv',index=False);s.dump(OUT/'SELECTIONS.json',choices)
    uncertainty=[]
    for model in ['hybrid','mlp']:
        for group in PACKS:
            name=f'{model}_{group}'
            for ref in [f'{model}_four','rf_baseline']:
                delta=abs(p[name]-p.truth)-abs(p[ref]-p.truth)
                for days in [7,14]:
                    blocks=(dev.asi8-dev.asi8.min())//pd.Timedelta(days=days).value
                    agg=pd.DataFrame(dict(delta=delta.to_numpy(),block=blocks)).groupby('block').delta.agg(['sum','count']);ids=np.random.default_rng(48).integers(0,len(agg),(3000,len(agg)))
                    draws=agg['sum'].to_numpy()[ids].sum(1)/agg['count'].to_numpy()[ids].sum(1);lo,hi=np.quantile(draws,[.025,.975])
                    uncertainty.append(dict(model=name,reference=ref,block_days=days,mae_change=float(delta.mean()),lower=lo,upper=hi))
    pd.DataFrame(uncertainty).to_csv(OUT/'uncertainty.csv',index=False)
    manifest=json.loads((OUT/'MANIFEST.json').read_text());assert all(s.q.sha(path)==digest for path,digest in manifest['sources'].items())
    summary=scores.to_string(index=False)+'\n\nFive inputs: original liquidation5m/15m pair plus volatility ratio4h/24h, venue breadth1h and realized volatility168h. Same18481 rows, fixed original hybrid and8/4 SiLU MLP settings. Historical exploratory evaluation.\n'
    (OUT/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    s.dump(OUT/'AUDIT.json',dict(status='PASS',rows=len(dev),windows=8,all_outer_and_validation_predictions_replayed=True,training_only_preprocessing=True,strict_purges=True,sources_unchanged=True,max_replay_error=maxerror))
    s.dump(OUT/'COMPLETE.json',dict(status='EXECUTED_AND_REPLAYED',scores=scores.to_dict('records')));s.dump(OUT/'STATUS.json',dict(stage='complete'));print(summary,flush=True)

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'RUN.lock').open('a+b') as lock:
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise SystemExit('Another five-feature run is already active; monitor its logs.')
        prepare()
        if (OUT/'COMPLETE.json').exists():print((OUT/'SUMMARY.txt').read_text());return
        def launch(job):
            pack,method,i=job
            with (OUT/f'{pack}_{method}_{i}.log').open('a') as log:r=subprocess.run([sys.executable,__file__,'worker',pack,method,str(i)],stdout=log,stderr=subprocess.STDOUT)
            if r.returncode:raise RuntimeError(f'Failed {job}; inspect log')
            print('FINISHED',pack,method,i,flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(launch,[(p,m,i) for i in range(4) for p in PACKS for m in CONFIGS]))
        summarize()

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if len(sys.argv)>1 and sys.argv[1]=='worker':worker(sys.argv[2],sys.argv[3],int(sys.argv[4]))
        elif len(sys.argv)>1 and sys.argv[1]=='prepare':prepare()
        elif len(sys.argv)>1 and sys.argv[1]=='summarize':summarize()
        else:run()
