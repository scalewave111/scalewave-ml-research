"""Matched hybrid additions; validation-gated tuning with original-setting fallback."""
import sys,json,subprocess,concurrent.futures
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
import individual_mlp_additions as a
s=a.s;e=a.e
OUT=s.HERE/'experiments/individual_hybrid_additions_v1'
CONFIGS={'original':{},'regularized':dict(n_estimators=64,max_depth=2,min_samples_leaf=256,max_samples=1024),'squared':dict(n_estimators=96,max_depth=4,min_samples_leaf=128,max_samples=2048,criterion='squared_error')}
def fit(x,y,keys,cut,cols,f,config,vk=None,weight=None):
    f.mkdir(parents=True,exist_ok=True);path=f/'bundle.joblib'
    if path.exists():
        z=joblib.load(path);assert z['keys'].equals(keys) and z['cols']==cols and z['config']==config;return z
    assert y.loc[keys,'label_end'].max()<cut
    # Share identical median-regression fits across configurations.
    linearpath=f.parent/'linear.joblib'
    if linearpath.exists():linear=joblib.load(linearpath);assert linear['keys'].equals(keys) and linear['cols']==cols
    else:
        prep=s.q.b.prep(x.loc[keys],cols);tx=s.q.b.design(x.loc[keys],prep)
        model=clone(joblib.load(s.q.b.BASE/'models/2024-01__report_outer_pair__expanding.joblib')['model']);model.fit(tx,y.loc[keys,'magnitude_log_bps'])
        linear=dict(model=model,prep=prep,keys=keys,cols=cols);joblib.dump(linear,linearpath,compress=3)
    rf=clone(joblib.load(s.q.b.BASE/'models/2024-01__report_outer_rf__expanding.joblib')['model']);rf.set_params(n_jobs=1,**CONFIGS[config])
    rf.fit(s.q.b.design(x.loc[keys],linear['prep']),y.loc[keys,'magnitude_log_bps'])
    models=dict(pair=linear,rf=dict(model=rf,prep=linear['prep']))
    if vk is not None:
        pp=s.q.b.predict(models['pair'],x.loc[vk]);rr=s.q.b.predict(models['rf'],x.loc[vk]);yy=y.loc[vk,'magnitude_log_bps'].to_numpy()
        weight=min(s.WEIGHTS,key=lambda w:(np.mean(abs((1-w)*pp+w*rr-yy)),w))
    z=dict(method='rf',models=models,weight=weight,keys=keys,cols=cols,cutoff=cut,config=config);joblib.dump(z,path,compress=3);return z
def worker(name,i):
    x,y,dev,ref=joblib.load(e.OUT/'prepared.joblib');issue=s.q.STARTS[i];inner,ik,vk,ot,te=s.q.windows(y,dev,issue);cols=list(a.PACKS[name])
    f=OUT/name/f'{issue:%Y%m%d}';f.mkdir(parents=True,exist_ok=True)
    if (f/'DONE.json').exists():return
    trials=[];valid={};yy=y.loc[vk,'magnitude_log_bps'].to_numpy()
    for ci,config in enumerate(CONFIGS):
        s.dump(f/'STATUS.json',dict(stage='validation_tuning',trial=ci+1,total=3))
        z=fit(x,y,ik,inner,cols,f/'inner'/config,config,vk=vk);valid[config]=s.predict(z,x,vk)
        trials.append(dict(config=config,weight=z['weight'],**s.metrics(valid[config],yy)))
    base=trials[0];best=min(trials,key=lambda r:(r['mae'],list(CONFIGS).index(r['config'])))
    diff=pd.Series(abs(valid['original']-yy)-abs(valid[best['config']]-yy),index=vk)
    monthly=diff.groupby(diff.index.strftime('%Y-%m')).mean();gain=base['mae']-best['mae']
    selected=best['config'] if gain>=.1 and int((monthly>0).sum())>=2 else 'original'
    s.dump(f/'TRIALS.json',trials);s.dump(f/'SELECTION.json',dict(selected=selected,best_validation=best['config'],gain=gain,monthly_gains=monthly.to_dict(),tuning_used=selected!='original'))
    p=pd.DataFrame(dict(truth=y.loc[te,'magnitude_log_bps'],issue=str(issue)),index=te)
    for config in dict.fromkeys(['original',selected]):
        s.dump(f/'STATUS.json',dict(stage='outer_refit',config=config))
        inner_art=joblib.load(f/'inner'/config/'bundle.joblib');outer=fit(x,y,ot,issue,cols,f/'outer'/config,config,weight=inner_art['weight'])
        pred=s.predict(outer,x,te);np.testing.assert_allclose(pred,s.predict(joblib.load(f/'outer'/config/'bundle.joblib'),x,te),atol=0,rtol=0)
        p[config]=pred
    p['selected']=p[selected];p.to_csv(f/'predictions.csv')
    assert y.loc[vk,'label_end'].max()<issue
    s.dump(f/'DONE.json',dict(status='EXECUTED_AND_REPLAYED',rows=len(te),selected=selected));s.dump(f/'STATUS.json',dict(stage='complete'))
def summarize():
    x,y,dev,ref=joblib.load(e.OUT/'prepared.joblib');p=pd.DataFrame(dict(truth=ref.truth,baseline=ref.report_hybrid),index=dev)
    old=pd.read_csv(s.HERE/'experiments/baseline_single_feature_v6/predictions.csv',index_col=0,parse_dates=True);assert old.index.equals(dev);p['yesterday_best']=old.vol_ratio
    choices=[]
    for name in a.PACKS:
        frames=[]
        for issue in s.q.STARTS:
            f=OUT/name/f'{issue:%Y%m%d}';assert (f/'DONE.json').exists();frames.append(pd.read_csv(f/'predictions.csv',index_col=0,parse_dates=True));choices.append(dict(pack=name,issue=str(issue),**json.loads((f/'SELECTION.json').read_text())))
        z=pd.concat(frames);assert z.index.equals(dev);np.testing.assert_allclose(z.truth,p.truth,atol=1e-8,rtol=0)
        for mode in ['original','selected']:p[name+'_'+mode]=z[mode]
    p.to_csv(OUT/'predictions.csv');scores=pd.DataFrame([dict(model=n,**s.metrics(p[n],p.truth)) for n in p if n!='truth']);scores.to_csv(OUT/'scores.csv',index=False);s.dump(OUT/'SELECTIONS.json',choices)
    rows=[];uncertainty=[]
    for name,cols in a.PACKS.items():
        fixed=s.metrics(p[name+'_original'],p.truth);selected=s.metrics(p[name+'_selected'],p.truth)
        rows.append(dict(pack=name,added_feature='none' if name=='core' else next(c for c in cols if c not in a.CORE),untuned_mae=fixed['mae'],untuned_ic=fixed['rank_ic'],selected_mae=selected['mae'],selected_ic=selected['rank_ic'],tuning_mae_change=selected['mae']-fixed['mae'],addition_mae_change=selected['mae']-s.metrics(p.core_selected,p.truth)['mae']))
        for reference in [name+'_original','core_selected','baseline','yesterday_best']:
            delta=abs(p[name+'_selected']-p.truth)-abs(p[reference]-p.truth);blocks=(dev.asi8-dev.asi8.min())//pd.Timedelta(days=7).value
            b=pd.DataFrame(dict(delta=delta.to_numpy(),block=blocks)).groupby('block').delta.agg(['sum','count']);ids=np.random.default_rng(48).integers(0,len(b),(3000,len(b)));boot=b['sum'].to_numpy()[ids].sum(1)/b['count'].to_numpy()[ids].sum(1);lo,hi=np.quantile(boot,[.025,.975]);uncertainty.append(dict(pack=name,reference=reference,mae_difference=delta.mean(),lower=lo,upper=hi))
    result=pd.DataFrame(rows);result.to_csv(OUT/'individual_results.csv',index=False);pd.DataFrame(uncertainty).to_csv(OUT/'uncertainty.csv',index=False)
    manifest=json.loads((OUT/'MANIFEST.json').read_text());assert all(s.q.sha(path)==sha for path,sha in manifest.items())
    s.dump(OUT/'COMPLETE.json',dict(status='EXECUTED_AND_REPLAYED',rows=len(dev),results=rows,limitation='Previously examined historical data; exploratory individual comparisons. Tuning gated on prior validation, not outer scores.'))
    summary=result.to_string(index=False)+'\n\nNegative change means lower MAE. Selected = original settings unless tuning improves prior validation by at least0.1bps and in at least2months. Same18481 rows. Historical exploratory evaluation.\n'
    (OUT/'SUMMARY.txt').write_text(summary,encoding='utf-8');print(summary,flush=True)
    with (s.ROOT/'PROJECT_CONTEXT.md').open('a',encoding='utf-8') as f:f.write('\n\n## Individual hybrid additions finished\nExecuted24 jobs; validation-gated tuning versus original settings. Results Code/experiments/individual_hybrid_additions_v1/SUMMARY.txt and individual_results.csv.\n'+summary)
    subprocess.run([sys.executable,str(s.ROOT/'refresh_file_index.py')],stdout=subprocess.DEVNULL)
def run():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'COMPLETE.json').exists():print((OUT/'SUMMARY.txt').read_text());return
    contract=dict(packs=a.PACKS,configs=CONFIGS,weights=s.WEIGHTS,tuning_gate='Validation MAE gain>=0.1bps and positive gain in>=2months; otherwise original.',rows=18481,source_sha=s.q.sha(__file__))
    if (OUT/'CONTRACT.json').exists():assert json.loads((OUT/'CONTRACT.json').read_text())==contract
    else:s.dump(OUT/'CONTRACT.json',contract);(OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());s.dump(OUT/'MANIFEST.json',{str(p):s.q.sha(p) for p in [Path(__file__),Path(a.__file__),Path(s.__file__),e.OUT/'prepared.joblib']})
    def launch(job):
        name,i=job
        with (OUT/f'{name}_{i}.log').open('a') as log:r=subprocess.run([sys.executable,__file__,'worker',name,str(i)],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'Worker failed {job}')
        print('FINISHED',job,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(launch,[(n,i) for i in range(4) for n in a.PACKS]))
    summarize()
if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        if sys.argv[1]=='worker':worker(sys.argv[2],int(sys.argv[3]))
        else:run()
