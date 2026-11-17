"""Matched core-plus-one MLP experiment; fixed architecture and seed, no search."""
import sys,json,subprocess,concurrent.futures,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import joblib,torch
import eight_feature_search as e
s=e.s
OUT=s.HERE/'experiments/individual_mlp_additions_v1'
CORE=s.PAIR+['PRICE__vol_ratio_4h_24h']
EXTRAS=[c for c in e.COLS if c not in CORE]
PACKS={'core':CORE,**{f'add_{i+1}':CORE+[c] for i,c in enumerate(EXTRAS)}}
CONFIG=dict(widths=[8,4],dropout=.05)
def worker(name,i):
    torch.set_num_threads(1);e.configure('mlp',CONFIG);s.SEEDS=[48]
    x,y,dev,ref=joblib.load(e.OUT/'prepared.joblib');issue=s.q.STARTS[i];inner,ik,vk,ot,te=s.q.windows(y,dev,issue)
    f=OUT/name/f'{issue:%Y%m%d}';f.mkdir(parents=True,exist_ok=True)
    if (f/'DONE.json').exists():return
    cols=sorted(PACKS[name]);s.dump(f/'STATUS.json',dict(stage='validation_fit'))
    ai=s.fit_pack('mlp',x,y,ik,inner,cols,f/'inner',vk=vk)
    s.dump(f/'SELECTION.json',dict(features=cols,epochs=ai['epochs'],seed=48,architecture=CONFIG))
    s.dump(f/'STATUS.json',dict(stage='outer_refit'))
    ao=s.fit_pack('mlp',x,y,ot,issue,cols,f/'outer',chosen=ai)
    pred=s.predict(ao,x,te);replay=joblib.load(f/'outer/bundle.joblib')
    np.testing.assert_allclose(pred,s.predict(replay,x,te),atol=0,rtol=0)
    assert y.loc[ik,'label_end'].max()<inner and y.loc[vk,'label_end'].max()<issue and y.loc[ot,'label_end'].max()<issue
    assert ai['epochs']==ao['epochs'] and ao['keys'].equals(ot) and ao['cols']==cols
    pd.DataFrame(dict(prediction=pred,truth=y.loc[te,'magnitude_log_bps'],issue=str(issue)),index=te).to_csv(f/'predictions.csv')
    s.dump(f/'DONE.json',dict(status='EXECUTED_AND_REPLAYED',epochs=ao['epochs'],rows=len(te)))
    s.dump(f/'STATUS.json',dict(stage='complete'))
def summarize():
    x,y,dev,ref=joblib.load(e.OUT/'prepared.joblib');p=pd.DataFrame(dict(truth=ref.truth,original_baseline=ref.report_hybrid),index=dev)
    previous=pd.read_csv(s.HERE/'experiments/baseline_single_feature_v6/predictions.csv',index_col=0,parse_dates=True);assert previous.index.equals(dev);p['yesterday_hybrid']=previous.vol_ratio
    for name in PACKS:
        frames=[]
        for issue in s.q.STARTS:
            f=OUT/name/f'{issue:%Y%m%d}';assert (f/'DONE.json').exists();frames.append(pd.read_csv(f/'predictions.csv',index_col=0,parse_dates=True))
        z=pd.concat(frames);assert z.index.equals(dev);np.testing.assert_allclose(z.truth,p.truth,atol=1e-8,rtol=0);p[name]=z.prediction
    scores=pd.DataFrame([dict(model=n,**s.metrics(p[n],p.truth)) for n in p if n!='truth']);scores.to_csv(OUT/'scores.csv',index=False);p.to_csv(OUT/'predictions.csv')
    core=s.metrics(p.core,p.truth);results=[]
    for i,c in enumerate(EXTRAS):
        name=f'add_{i+1}';m=s.metrics(p[name],p.truth);delta=abs(p[name]-p.truth)-abs(p.core-p.truth)
        blocks=(dev.asi8-dev.asi8.min())//pd.Timedelta(days=7).value;b=pd.DataFrame(dict(delta=delta.to_numpy(),block=blocks)).groupby('block').delta.agg(['sum','count'])
        ids=np.random.default_rng(48).integers(0,len(b),(3000,len(b)));boot=b['sum'].to_numpy()[ids].sum(1)/b['count'].to_numpy()[ids].sum(1);lo,hi=np.quantile(boot,[.025,.975])
        results.append(dict(added_feature=c,mae=m['mae'],ic=m['rank_ic'],mae_change_vs_matched_core=m['mae']-core['mae'],ic_change_vs_matched_core=m['rank_ic']-core['rank_ic'],lower=lo,upper=hi))
    result=pd.DataFrame(results);result.to_csv(OUT/'individual_contributions.csv',index=False)
    manifest=json.loads((OUT/'MANIFEST.json').read_text());assert all(s.q.sha(path)==sha for path,sha in manifest['sources'].items())
    s.dump(OUT/'COMPLETE.json',dict(status='EXECUTED_AND_REPLAYED',rows=len(dev),fits=24,matched_core=core,results=results,limitation='Single seed48, fixed architecture, identical early-stopping rule with feature-specific selected epochs; previously examined historical data. Pointwise weekly-block intervals, not corrected for5 comparisons. Does not establish causal or future gains.'))
    summary='Matched single-feature additions to three-feature MLP\n\n'+scores.to_string(index=False)+'\n\n'+result.to_string(index=False)+'\n\nNegative MAE change means adding the feature helped this matched MLP. Comparison against yesterday hybrid also changes model family. Single-seed exploratory results; no guarantee of future gain.\n'
    (OUT/'SUMMARY.txt').write_text(summary,encoding='utf-8');print(summary,flush=True)
    with (s.ROOT/'PROJECT_CONTEXT.md').open('a',encoding='utf-8') as f:f.write('\n\n## Individual MLP additions completed\nExecuted and checkpoint-replayed24 model/window jobs, fixed8/4 SiLU/dropout.05, seed48, no architecture search. Results: Code/experiments/individual_mlp_additions_v1/SUMMARY.txt and individual_contributions.csv. Compare additions against matched three-feature MLP; historical hybrid benchmarks are separate.\n'+result.to_string(index=False)+'\n')
    subprocess.run([sys.executable,str(s.ROOT/'refresh_file_index.py')],stdout=subprocess.DEVNULL)
def run():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'COMPLETE.json').exists():print((OUT/'SUMMARY.txt').read_text());return
    s.dump(OUT/'CONTRACT.json',dict(core=CORE,extras=EXTRAS,packs=PACKS,architecture=CONFIG,seeds=[48],max_epochs=48,patience=8,selection='Prior purged validation selects epochs only. No architecture or subset tuning.',comparison='Same18481 rows; primary paired difference versus same-seed/same-architecture core MLP.'))
    if not (OUT/'MANIFEST.json').exists():
        paths=[Path(__file__),Path(s.__file__),Path(e.__file__),Path(s.m.__file__),e.OUT/'prepared.joblib']
        s.dump(OUT/'MANIFEST.json',dict(sources={str(p):s.q.sha(p) for p in paths}));(OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    def launch(job):
        name,i=job
        with (OUT/f'{name}_{i}.log').open('a') as log:r=subprocess.run([sys.executable,__file__,'worker',name,str(i)],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'Worker failed {job}')
        print('FINISHED',job,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(launch,[(n,i) for i in range(4) for n in PACKS]))
    summarize()
if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        if sys.argv[1]=='worker':worker(sys.argv[2],int(sys.argv[3]))
        else:run()
