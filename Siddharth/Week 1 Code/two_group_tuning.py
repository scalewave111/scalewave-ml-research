"""Two fixed feature groups, validation-only tuning, same historical baseline rows."""
import sys,json,subprocess,concurrent.futures
from pathlib import Path
import numpy as np
import pandas as pd
import joblib,torch
import eight_feature_search as e
s=e.s
OUT=s.HERE/'experiments/two_group_tuning_v1'
GROUPS={'eight':e.COLS,'seven':[c for c in e.COLS if c!='F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__1h__value']}
CONFIGS={'rf':[dict(n_estimators=64,max_depth=3,min_samples_leaf=64,max_samples=1024,criterion='absolute_error'),dict(n_estimators=96,max_depth=4,min_samples_leaf=128,max_samples=2048,criterion='squared_error')], 'mlp':[dict(widths=[8,4],dropout=.05),dict(widths=[4],dropout=.15)]}
def worker(method,group,i):
    torch.set_num_threads(1);x,y,dev,ref=joblib.load(e.OUT/'prepared.joblib');issue=s.q.STARTS[i]
    inner,ik,vk,ot,te=s.q.windows(y,dev,issue);cols=sorted(GROUPS[group]);f=OUT/method/group/f'{issue:%Y%m%d}';f.mkdir(parents=True,exist_ok=True)
    if (f/'DONE.json').exists():return
    trials=[];s.SEEDS=[48]
    for ci,c in enumerate(CONFIGS[method]):
        e.configure(method,c);s.dump(f/'STATUS.json',dict(stage='tuning',trial=ci+1,total=2))
        a=s.fit_pack(method,x,y,ik,inner,cols,f/f'c{ci}'/'inner',vk=vk)
        trials.append(dict(config=c,id=ci,**s.metrics(s.predict(a,x,vk),y.loc[vk,'magnitude_log_bps'])))
        s.dump(f/'TRIALS.json',trials)
    chosen=min(trials,key=lambda r:(r['mae'],r['id']));s.dump(f/'SELECTION.json',chosen);e.configure(method,chosen['config'])
    ai=joblib.load(f/f"c{chosen['id']}"/'inner/bundle.joblib')
    if method=='mlp':
        s.SEEDS=[48,149,250];ai=s.fit_pack(method,x,y,ik,inner,cols,f/'confirmation',vk=vk)
    s.dump(f/'STATUS.json',dict(stage='outer_refit'))
    ao=s.fit_pack(method,x,y,ot,issue,cols,f/'outer',chosen=ai)
    pred=s.predict(ao,x,te);saved=joblib.load(f/'outer/bundle.joblib');np.testing.assert_allclose(pred,s.predict(saved,x,te),atol=0,rtol=0)
    assert ao['keys'].equals(ot) and ao['cutoff']==issue and y.loc[ot,'label_end'].max()<issue
    assert y.loc[ik,'label_end'].max()<inner and y.loc[vk,'label_end'].max()<issue
    if method=='rf':assert ai['weight']==ao['weight']
    else:assert ai['epochs']==ao['epochs']
    pd.DataFrame(dict(prediction=pred,truth=y.loc[te,'magnitude_log_bps'],issue=str(issue)),index=te).to_csv(f/'predictions.csv')
    s.dump(f/'DONE.json',dict(status='EXECUTED_AND_REPLAYED',method=method,group=group,features=cols,validation_mae=chosen['mae']))
    s.dump(f/'STATUS.json',dict(stage='complete'))
def summarize():
    x,y,dev,ref=joblib.load(e.OUT/'prepared.joblib');p=pd.DataFrame(dict(truth=ref.truth,baseline=ref.report_hybrid),index=dev)
    old=pd.read_csv(s.HERE/'experiments/baseline_single_feature_v6/predictions.csv',index_col=0,parse_dates=True);assert old.index.equals(dev);p['yesterday_best']=old.vol_ratio
    decisions=[]
    for method in CONFIGS:
        adaptive=[]
        for group in GROUPS:
            pieces=[]
            for issue in s.q.STARTS:
                f=OUT/method/group/f'{issue:%Y%m%d}';assert (f/'DONE.json').exists()
                pieces.append(pd.read_csv(f/'predictions.csv',index_col=0,parse_dates=True).prediction)
            z=pd.concat(pieces);assert z.index.equals(dev);p[f'{method}_{group}']=z
        for issue in s.q.STARTS:
            choices={g:json.loads((OUT/method/g/f'{issue:%Y%m%d}'/'SELECTION.json').read_text()) for g in GROUPS}
            g=min(choices,key=lambda k:(choices[k]['mae'],len(GROUPS[k])))
            _,_,_,_,te=s.q.windows(y,dev,issue);adaptive.append(p.loc[te,f'{method}_{g}']);decisions.append(dict(method=method,issue=str(issue),group=g,config=choices[g]['config']))
        p[f'{method}_selected']=pd.concat(adaptive)
    p.to_csv(OUT/'predictions.csv');scores=pd.DataFrame([dict(model=c,**s.metrics(p[c],p.truth)) for c in p if c!='truth']);scores.to_csv(OUT/'scores.csv',index=False)
    assert abs(scores.set_index('model').loc['baseline','mae']-181.52599276479125)<1e-8
    uncertainty=[]
    for c in p:
        if c in ['truth','baseline','yesterday_best']:continue
        for base in ['baseline','yesterday_best']:
            delta=abs(p[c]-p.truth)-abs(p[base]-p.truth);block=(p.index.asi8-p.index.asi8.min())//pd.Timedelta(days=7).value
            b=pd.DataFrame(dict(delta=delta.to_numpy(),block=block)).groupby('block').delta.agg(['sum','count']);ids=np.random.default_rng(48).integers(0,len(b),(3000,len(b)))
            boot=b['sum'].to_numpy()[ids].sum(1)/b['count'].to_numpy()[ids].sum(1);lo,hi=np.quantile(boot,[.025,.975]);uncertainty.append(dict(model=c,reference=base,difference=delta.mean(),lower=lo,upper=hi))
    pd.DataFrame(uncertainty).to_csv(OUT/'uncertainty.csv',index=False);s.dump(OUT/'SELECTIONS.json',decisions)
    s.dump(OUT/'COMPLETE.json',dict(status='EXECUTED_AND_REPLAYED',scores=scores.to_dict('records'),rows=len(dev),limitation='Previously exposed history; exploratory comparison.'))
    (OUT/'RESULTS.txt').write_text(scores.to_string(index=False),encoding='utf-8');print(scores.to_string(index=False),flush=True)
def run():
    OUT.mkdir(parents=True,exist_ok=True)
    contract=dict(groups=GROUPS,configs=CONFIGS,seed_tuning=[48],seed_final_mlp=[48,149,250],selection='Prior purged validation MAE selects configuration per group, then group for primary selected strategy.',target='48h absolute endpoint log return in bps',historical_rows=18481,source_sha=s.q.sha(__file__))
    if (OUT/'CONTRACT.json').exists():assert json.loads((OUT/'CONTRACT.json').read_text())==contract
    else:s.dump(OUT/'CONTRACT.json',contract);(OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    def launch(job):
        m,g,i=job
        with (OUT/f'{m}_{g}_{i}.log').open('a') as log:r=subprocess.run([sys.executable,__file__,'worker',m,g,str(i)],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'Worker failed: {job}')
        print('FINISHED',job,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(launch,[(m,g,i) for i in range(4) for m in CONFIGS for g in GROUPS]))
    summarize()
if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        if sys.argv[1]=='worker':worker(sys.argv[2],sys.argv[3],int(sys.argv[4]))
        else:run()
