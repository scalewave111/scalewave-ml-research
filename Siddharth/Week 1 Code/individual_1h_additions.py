"""Six separate additions to original pair; matched 1h hybrid and MLP tuning."""
import sys,json,subprocess,concurrent.futures,shutil,importlib.metadata,msvcrt
from pathlib import Path
import numpy as np
import pandas as pd
import joblib,torch
import two_feature_1h as b
import individual_hybrid_additions as h
s=b.s;e=b.e
OUT=s.HERE/'experiments/individual_1h_additions_v1'
EXTRAS=[c for c in e.COLS if c not in s.PAIR]
PACKS={f'add_{i+1}':s.PAIR+[c] for i,c in enumerate(EXTRAS)}
CONFIGS=b.CONFIGS

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'prepared.joblib').exists():return
    if not (b.OUT/'COMPLETE.json').exists():
        raise RuntimeError('Complete Code/two_feature_1h.py first; matched pair controls are required.')
    _,y,dev,ref=joblib.load(b.OUT/'prepared.joblib');x,_,olddev,_=joblib.load(e.OUT/'prepared.joblib')
    assert (y.label_end==y.index+pd.Timedelta(hours=1)).all()
    assert len(PACKS)==6 and all(len(set(v))==3 for v in PACKS.values())
    assert olddev.equals(dev);x=x.loc[y.index,e.COLS].copy()
    # Reconstruct price features independently from verified returns before reuse.
    r=b.c.load_returns(s.ROOT/'Dataset/canonical_returns_5m.parquet');derived=b.c.features(r,x.index)
    for c in [c for c in EXTRAS if c.startswith('PRICE__')]:np.testing.assert_allclose(x[c],derived[c],atol=1e-10,rtol=0,equal_nan=True)
    base=pd.read_csv(b.OUT/'predictions.csv',index_col=0,parse_dates=True);assert base.index.equals(dev)
    np.testing.assert_allclose(base.truth,y.loc[dev,'magnitude_log_bps'],atol=1e-8,rtol=0)
    joblib.dump((x,y,dev,base),OUT/'prepared.joblib',compress=3)
    s.dump(OUT/'CONTRACT.json',dict(packs=PACKS,configs=CONFIGS,weights=s.WEIGHTS,mlp_tuning_seeds=[48],mlp_final_seeds=[48,149,250],target='1h absolute endpoint log return in log bps',selection='Same two-config validation-MAE tuning budget as original-pair run; each addition retuned. Four expanding windows with strict label_end < cutoff. No test-driven retries.',comparison='Each model versus its saved tuned two-feature counterpart and saved RF baseline on identical18481 rows. Changes include feature addition and its validation retuning; not fixed-parameter effects.',cache='Identical training-only median fits reused across configs within same feature pack and cutoff.'))
    paths=[Path(__file__),Path(b.__file__),Path(h.__file__),Path(s.__file__),Path(e.__file__),Path(s.m.__file__),Path(s.q.__file__),Path(s.q.b.__file__),e.OUT/'prepared.joblib',b.OUT/'prepared.joblib',b.OUT/'predictions.csv',s.ROOT/'Dataset/canonical_returns_5m.parquet']
    fw=Path('C:/Users/Siddh/Desktop/ML/ml_framework')
    docs=[s.ROOT/'PROJECT_CONTEXT.md']+[fw/n for n in ['AGENTS.md','CONTEXT.md','docs/INDEX.md','docs/methods/WORKFLOW.md','docs/methods/DATA_AND_METRICS.md','jobs.json']]
    s.dump(OUT/'MANIFEST.json',dict(sources={str(p):s.q.sha(p) for p in paths},instructions={str(p):s.q.sha(p) for p in docs},versions={n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','scikit-learn','joblib']}))
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())

def mlp_fit(x,y,keys,cut,cols,folder,shared,vk=None,chosen=None):
    # The existing fitter is unchanged; only identical linear fits are reused.
    dest=folder/'linear'/f'{cut:%Y%m%d}.joblib';shared.mkdir(parents=True,exist_ok=True);cache=shared/f'{cut:%Y%m%d}.joblib'
    if cache.exists() and not dest.exists():
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(cache,dest)
    art=s.fit_pack('mlp',x,y,keys,cut,cols,folder,vk=vk,chosen=chosen)
    if not cache.exists():shutil.copy2(dest,cache)
    assert art['linear']['train_keys'].equals(keys) and art['linear']['cutoff']==cut
    return art

def worker(method,pack,i):
    torch.set_num_threads(1);x,y,dev,base=joblib.load(OUT/'prepared.joblib');issue=s.q.STARTS[i]
    inner,ik,vk,ot,te=s.q.windows(y,dev,issue);cols=sorted(PACKS[pack]);f=OUT/method/pack/f'{issue:%Y%m%d}';f.mkdir(parents=True,exist_ok=True)
    if (f/'DONE.json').exists():return
    trials=[];s.SEEDS=[48]
    for ci,config in enumerate(CONFIGS[method]):
        e.configure(method,config);s.dump(f/'STATUS.json',dict(stage='validation',trial=ci+1,total=2))
        if method=='rf':art=h.fit(x,y,ik,inner,cols,f/'inner'/f'c{ci}','original' if ci==0 else 'squared',vk=vk)
        else:art=mlp_fit(x,y,ik,inner,cols,f/f'c{ci}'/'inner',f/'shared_linear',vk=vk)
        trials.append(dict(id=ci,config=config,**s.metrics(s.predict(art,x,vk),y.loc[vk,'magnitude_log_bps'])))
        s.dump(f/'TRIALS.json',trials)
    choice=min(trials,key=lambda z:(z['mae'],z['id']));s.dump(f/'SELECTION.json',choice);e.configure(method,choice['config']);ci=choice['id']
    if method=='rf':
        ai=joblib.load(f/'inner'/f'c{ci}'/'bundle.joblib');s.dump(f/'STATUS.json',dict(stage='outer_refit'))
        ao=h.fit(x,y,ot,issue,cols,f/'outer'/'selected','original' if ci==0 else 'squared',weight=ai['weight']);outerpath=f/'outer'/'selected'/'bundle.joblib'
    else:
        s.SEEDS=[48,149,250];s.dump(f/'STATUS.json',dict(stage='selected_three_seeds'))
        ai=mlp_fit(x,y,ik,inner,cols,f/'confirmation',f/'shared_linear',vk=vk)
        s.dump(f/'STATUS.json',dict(stage='outer_refit'))
        ao=mlp_fit(x,y,ot,issue,cols,f/'outer',f/'shared_linear',chosen=ai);outerpath=f/'outer'/'bundle.joblib'
    pred=s.predict(ao,x,te);np.testing.assert_allclose(pred,s.predict(joblib.load(outerpath),x,te),atol=0,rtol=0)
    assert y.loc[ik,'label_end'].max()<inner and y.loc[vk,'label_end'].max()<issue and y.loc[ot,'label_end'].max()<issue
    assert ao['keys'].equals(ot) and ao['cols']==cols
    if method=='rf':assert ao['weight']==ai['weight']
    else:assert ao['epochs']==ai['epochs']
    pd.DataFrame(dict(prediction=pred,truth=y.loc[te,'magnitude_log_bps'],issue=str(issue)),index=te).to_csv(f/'predictions.csv')
    s.dump(f/'DONE.json',dict(status='EXECUTED_AND_REPLAYED',rows=len(te),features=cols,outerpath=str(outerpath),weight=ao.get('weight'),epochs=ao.get('epochs')))
    s.dump(f/'STATUS.json',dict(stage='complete'))

def summarize():
    x,y,dev,base=joblib.load(OUT/'prepared.joblib');p=base.copy();rows=[];monthly=[]
    for method in CONFIGS:
        model='hybrid' if method=='rf' else 'mlp';control=s.metrics(base[model],base.truth)
        for pack,cols in PACKS.items():
            pieces=[]
            for issue in s.q.STARTS:
                f=OUT/method/pack/f'{issue:%Y%m%d}';done=json.loads((f/'DONE.json').read_text());choice=json.loads((f/'SELECTION.json').read_text());e.configure(method,choice['config'])
                z=pd.read_csv(f/'predictions.csv',index_col=0,parse_dates=True);ao=joblib.load(done['outerpath'])
                np.testing.assert_allclose(s.predict(ao,x,z.index),z.prediction,atol=1e-8,rtol=0)
                np.testing.assert_allclose(z.truth,y.loc[z.index,'magnitude_log_bps'],atol=1e-8,rtol=0);pieces.append(z.prediction)
            pred=pd.concat(pieces);assert pred.index.equals(dev);p[f'{model}_{pack}']=pred;m=s.metrics(pred,base.truth)
            row=dict(model=model,pack=pack,added_feature=next(c for c in cols if c not in s.PAIR),**m,mae_change_vs_pair=m['mae']-control['mae'],ic_change_vs_pair=m['rank_ic']-control['rank_ic'],mae_change_vs_rf_baseline=m['mae']-s.metrics(base.rf_baseline,base.truth)['mae'])
            rows.append(row)
            for month,g in pd.DataFrame(dict(pred=pred,truth=base.truth)).groupby(dev.strftime('%Y-%m')):monthly.append(dict(model=model,pack=pack,month=month,**s.metrics(g.pred,g.truth)))
    result=pd.DataFrame(rows);result.to_csv(OUT/'individual_results.csv',index=False);p.to_csv(OUT/'predictions.csv');pd.DataFrame(monthly).to_csv(OUT/'monthly_scores.csv',index=False)
    # Paired weekly blocks reflect serial dependence; descriptive intervals only.
    uncertainty=[];blocks=(dev.asi8-dev.asi8.min())//pd.Timedelta(days=7).value
    for row in rows:
        delta=abs(p[f"{row['model']}_{row['pack']}"]-p.truth)-abs(p[row['model']]-p.truth)
        agg=pd.DataFrame(dict(delta=delta.to_numpy(),block=blocks)).groupby('block').delta.agg(['sum','count']);ids=np.random.default_rng(48).integers(0,len(agg),(3000,len(agg)))
        draws=agg['sum'].to_numpy()[ids].sum(1)/agg['count'].to_numpy()[ids].sum(1);lo,hi=np.quantile(draws,[.025,.975]);uncertainty.append(dict(model=row['model'],pack=row['pack'],lower=lo,upper=hi))
    pd.DataFrame(uncertainty).to_csv(OUT/'uncertainty.csv',index=False)
    manifest=json.loads((OUT/'MANIFEST.json').read_text());assert all(s.q.sha(path)==digest for path,digest in manifest['sources'].items())
    s.dump(OUT/'AUDIT.json',dict(status='PASS',rows=len(dev),combinations=6,model_window_jobs=48,all_outer_predictions_replayed=True,sources_unchanged=True))
    s.dump(OUT/'COMPLETE.json',dict(status='EXECUTED_AND_REPLAYED',rows=len(dev),results=rows,limitation='Previously examined historical data; individual additions retuned on prior validation.'))
    (OUT/'RESULTS.txt').write_text(result.to_string(index=False));print(result.to_string(index=False),flush=True)
    with (s.ROOT/'PROJECT_CONTEXT.md').open('a',encoding='utf-8') as f:f.write('\n\n## 1h six individual additions completed\nSource Code/experiments/individual_1h_additions_v1/individual_results.csv. Each extra added separately to original pair; two-config validation tuning, three final MLP seeds; all predictions replayed.\n'+result.to_string(index=False)+'\n')

def run_locked():
    prepare()
    if (OUT/'COMPLETE.json').exists():print((OUT/'RESULTS.txt').read_text());return
    def launch(job):
        method,pack,i=job
        with (OUT/f'{method}_{pack}_{i}.log').open('a') as log:res=subprocess.run([sys.executable,__file__,'worker',method,pack,str(i)],stdout=log,stderr=subprocess.STDOUT)
        if res.returncode:raise RuntimeError(f'Worker failed {job}; inspect log')
        print('FINISHED',method,pack,i,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(launch,[(m,p,i) for p in PACKS for i in range(4) for m in CONFIGS]))
    summarize()

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'RUN.lock').open('a+b') as lock:
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise SystemExit('This experiment is already running. Check the existing window.')
        run_locked()

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if len(sys.argv)>1 and sys.argv[1]=='worker':worker(sys.argv[2],sys.argv[3],int(sys.argv[4]))
        else:run()
