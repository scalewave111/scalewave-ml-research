"""Four-input 4h models; prior-validation gate on optional tuning."""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[n]='1'
import sys,json,shutil,subprocess,concurrent.futures,importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import joblib,torch
import individual_4h_additions as old
s=old.s;e=old.e;h=old.h
OUT=s.HERE/'experiments/four_feature_4h_v1'
COLS=sorted(s.PAIR+['PRICE__vol_ratio_4h_24h','F4Z__liquidation__venue_breadth_percentile__1h__value'])
CONFIGS={'rf':['original','regularized','squared'],'mlp':[dict(widths=[8,4],dropout=.05),dict(widths=[4],dropout=.15),dict(widths=[16,8],dropout=.15),dict(widths=[8],dropout=0.)]}
SEEDS=[48,149,250]
def load():
    x,y,dev,base=joblib.load(old.OUT/'prepared.joblib')
    return x[COLS],y,dev,base

def configure(method,ci):
    if method=='mlp':e.configure('mlp',CONFIGS[method][ci]);s.m.Net=e.TunedNet

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'MANIFEST.json').exists():
        manifest=json.loads((OUT/'MANIFEST.json').read_text())
        assert all(s.q.sha(p)==v for p,v in manifest['sources'].items());return
    x,y,dev,base=load()
    assert len(COLS)==4 and len(dev)==18481 and x.index.equals(y.index)
    assert (y.label_end==y.index+pd.Timedelta(hours=4)).all()
    np.testing.assert_allclose(y.loc[dev,'magnitude_log_bps'],base.truth,atol=1e-8,rtol=0)
    for issue in s.q.STARTS:
        inner,ik,vk,ot,te=s.q.windows(y,dev,issue)
        assert y.loc[ik,'label_end'].max()<inner and y.loc[vk,'label_end'].max()<issue and y.loc[ot,'label_end'].max()<issue
    s.dump(OUT/'CONTRACT.json',dict(features=COLS,target='4h absolute endpoint log return in log bps',configs=CONFIGS,rf_configs=h.CONFIGS,activation='SiLU',seeds=SEEDS,defaults='Original RF and 8/4 MLP dropout.05; blend validation-selected.',selection='Screen fixed candidates using prior validation. MLP confirms default and best screen candidate with3 seeds. Accept change only for >=.1bps validation MAE gain and gains in >=2months; otherwise default. Freeze before outer refits.',comparison='Default and gated strategies, same18481 rows and previous2/3-input controls.',limitation='Previously exposed history; exploratory. No test-driven changes.'))
    paths=[Path(__file__),Path(old.__file__),Path(old.b.__file__),Path(h.__file__),Path(e.__file__),Path(s.__file__),Path(s.m.__file__),Path(s.q.__file__),Path(s.q.b.__file__),old.OUT/'prepared.joblib',old.OUT/'predictions.csv',old.b.OUT/'predictions.csv']
    fw=Path('C:/Users/Siddh/Desktop/ML/ml_framework')
    docs=[s.ROOT/'PROJECT_CONTEXT.md',s.HERE/'PROJECT_CONTEXT.md']+[fw/n for n in ['AGENTS.md','CONTEXT.md','docs/INDEX.md','docs/methods/WORKFLOW.md','docs/methods/DATA_AND_METRICS.md','docs/findings/RECORDED_RESULTS.md','jobs.json']]
    s.dump(OUT/'MANIFEST.json',dict(sources={str(p):s.q.sha(p) for p in paths},instructions={str(p):s.q.sha(p) for p in docs},versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','torch','scikit-learn','joblib']}))
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())

def fit(method,ci,x,y,keys,cut,f,shared,vk=None,chosen=None):
    configure(method,ci)
    if method=='rf':return h.fit(x,y,keys,cut,COLS,f,CONFIGS[method][ci],vk=vk,weight=None if chosen is None else chosen['weight'])
    return old.mlp_fit(x,y,keys,cut,COLS,f,shared,vk=vk,chosen=chosen)

def check(a,method,x,y,keys,cut,vk=None):
    assert a['keys'].equals(keys) and a['cutoff']==cut and a['cols']==COLS
    assert y.loc[keys,'label_end'].max()<cut
    if method=='rf':
        expected=s.q.b.prep(x.loc[keys],COLS)
        for m in a['models'].values():
            for field in ['median','center','scale']:pd.testing.assert_series_equal(expected[field],m['prep'][field])
    else:
        assert a['linear']['train_keys'].equals(keys) and a['linear']['cutoff']==cut
        expected=s.q.prep(x,keys,COLS)
        for field in ['median','scale']:pd.testing.assert_series_equal(expected[field],a['linear']['prep'][field])
        for neural in a['neural']:
            assert neural['train_keys'].equals(keys) and neural['cutoff']==cut
            if vk is not None:assert neural['validation_keys'].equals(vk)

def worker(method,i):
    x,y,dev,base=load();issue=s.q.STARTS[i];inner,ik,vk,ot,te=s.q.windows(y,dev,issue)
    assert y.loc[vk,'label_end'].max()<issue
    f=OUT/method/f'{issue:%Y%m%d}';f.mkdir(parents=True,exist_ok=True)
    if (f/'DONE.json').exists():return
    s.SEEDS=[48];trials=[];valid={}
    for ci,config in enumerate(CONFIGS[method]):
        s.dump(f/'STATUS.json',dict(stage='validation_screen',trial=ci+1,total=len(CONFIGS[method])))
        a=fit(method,ci,x,y,ik,inner,f/'screen'/f'c{ci}',f/'shared_linear',vk=vk)
        check(a,method,x,y,ik,inner,vk);valid[ci]=s.predict(a,x,vk)
        trials.append(dict(id=ci,config=config,**s.metrics(valid[ci],y.loc[vk,'magnitude_log_bps'])))
        s.dump(f/'SCREEN_TRIALS.json',trials);print(method,i,'SCREEN',ci,flush=True)
    best=min(trials,key=lambda row:(row['mae'],row['id']))['id']
    considered=list(dict.fromkeys([0,best]));inners={};innerpaths={}
    for ci in considered:
        if method=='mlp':
            s.SEEDS=SEEDS;path=f/'confirmed'/f'c{ci}';path.mkdir(parents=True,exist_ok=True)
            if not (path/'seed48.pt').exists():shutil.copy2(f/'screen'/f'c{ci}'/'seed48.pt',path/'seed48.pt')
            s.dump(f/'STATUS.json',dict(stage='three_seed_confirmation',config=ci))
            a=fit(method,ci,x,y,ik,inner,path,f/'shared_linear',vk=vk)
        else:path=f/'screen'/f'c{ci}';a=joblib.load(path/'bundle.joblib')
        check(a,method,x,y,ik,inner,vk);inners[ci]=a;innerpaths[ci]=str(path/'bundle.joblib')
        valid[ci]=s.predict(a,x,vk)
    truth=y.loc[vk,'magnitude_log_bps']
    delta=pd.Series(abs(valid[0]-truth.to_numpy())-abs(valid[best]-truth.to_numpy()),index=vk)
    monthly=delta.groupby(vk.strftime('%Y-%m')).mean();gain=float(delta.mean())
    selected=best if best!=0 and gain>=.1 and int((monthly>0).sum())>=2 else 0
    decision=dict(issue=str(issue),method=method,best_screening_id=best,selected_id=selected,selected_config=CONFIGS[method][selected],change_accepted=selected!=0,validation_gain=gain,monthly_gains=monthly.to_dict(),innerpaths=innerpaths,confirmed_trials=[dict(id=ci,**s.metrics(valid[ci],truth)) for ci in considered],selected_before_outer_refits=True)
    s.dump(f/'SELECTION.json',decision)
    pd.DataFrame(dict(truth=truth,default=valid[0],candidate=valid[best],selected=valid[selected])).to_csv(f/'validation_predictions.csv')
    p=pd.DataFrame(dict(truth=y.loc[te,'magnitude_log_bps']));outerpaths={}
    for ci in dict.fromkeys([0,selected]):
        s.dump(f/'STATUS.json',dict(stage='outer_refit',config=ci))
        ao=fit(method,ci,x,y,ot,issue,f/'outer'/f'c{ci}',f/'shared_linear',chosen=inners[ci]);check(ao,method,x,y,ot,issue)
        if method=='mlp':assert ao['epochs']==inners[ci]['epochs']
        else:assert ao['weight']==inners[ci]['weight']
        outerpaths[ci]=str(f/'outer'/f'c{ci}'/'bundle.joblib');p[f'c{ci}']=s.predict(ao,x,te)
        np.testing.assert_allclose(p[f'c{ci}'],s.predict(joblib.load(outerpaths[ci]),x,te),atol=0,rtol=0)
    p['default']=p.c0;p['selected']=p[f'c{selected}'];p.to_csv(f/'predictions.csv')
    s.dump(f/'DONE.json',dict(status='EXECUTED_AND_REPLAYED',outerpaths=outerpaths,rows=len(te)))
    s.dump(f/'STATUS.json',dict(stage='complete'))

def summarize():
    x,y,dev,base=load();p=pd.DataFrame(dict(truth=base.truth,rf_baseline=base.rf_baseline,hybrid_pair=base.hybrid,mlp_pair=base.mlp))
    prior=pd.read_csv(old.OUT/'predictions.csv',index_col=0,parse_dates=True);assert prior.index.equals(dev)
    np.testing.assert_allclose(prior.truth,p.truth,atol=1e-8,rtol=0)
    for model in ['hybrid','mlp']:
        for suffix,pack in [('ratio','add_2'),('breadth','add_4')]:p[f'{model}_three_{suffix}']=prior[f'{model}_{pack}']
    choices=[];windows=[]
    for method in CONFIGS:
        model='hybrid' if method=='rf' else 'mlp';pieces=[]
        for issue in s.q.STARTS:
            f=OUT/method/f'{issue:%Y%m%d}';done=json.loads((f/'DONE.json').read_text());choice=json.loads((f/'SELECTION.json').read_text())
            z=pd.read_csv(f/'predictions.csv',index_col=0,parse_dates=True);v=pd.read_csv(f/'validation_predictions.csv',index_col=0,parse_dates=True)
            delta=abs(v.default-v.truth)-abs(v.candidate-v.truth)
            np.testing.assert_allclose(delta.mean(),choice['validation_gain'],atol=1e-8,rtol=0)
            months=delta.groupby(v.index.strftime('%Y-%m')).mean()
            expected=choice['best_screening_id'] if delta.mean()>=.1 and int((months>0).sum())>=2 else 0
            assert choice['selected_id']==expected
            for ci,path in done['outerpaths'].items():
                configure(method,int(ci));np.testing.assert_allclose(s.predict(joblib.load(path),x,z.index),z[f'c{ci}'],atol=1e-8,rtol=0)
            for ci,path in choice['innerpaths'].items():
                configure(method,int(ci));ai=joblib.load(path)
                for name,target in [('default',0),('candidate',choice['best_screening_id']),('selected',choice['selected_id'])]:
                    if int(ci)==target:np.testing.assert_allclose(s.predict(ai,x,v.index),v[name],atol=1e-8,rtol=0)
            np.testing.assert_allclose(z.selected,z[f"c{choice['selected_id']}"],atol=0,rtol=0)
            np.testing.assert_allclose(z.truth,y.loc[z.index,'magnitude_log_bps'],atol=1e-8,rtol=0)
            for mode in ['default','selected']:windows.append(dict(model=model,issue=str(issue),mode=mode,**s.metrics(z[mode],z.truth)))
            choices.append(choice);pieces.append(z)
        z=pd.concat(pieces);assert z.index.equals(dev)
        for mode in ['default','selected']:p[f'{model}_four_{mode}']=z[mode]
    scores=pd.DataFrame([dict(model=n,**s.metrics(p[n],p.truth)) for n in p if n!='truth'])
    scores.to_csv(OUT/'scores.csv',index=False);p.to_csv(OUT/'predictions.csv');pd.DataFrame(windows).to_csv(OUT/'window_scores.csv',index=False);s.dump(OUT/'SELECTIONS.json',choices)
    uncertainty=[]
    for model in ['hybrid','mlp']:
        for mode in ['default','selected']:
            name=f'{model}_four_{mode}'
            for ref in [f'{model}_three_ratio',f'{model}_three_breadth',f'{model}_four_default','rf_baseline']:
                if name==ref:continue
                delta=abs(p[name]-p.truth)-abs(p[ref]-p.truth)
                for days in [7,14]:
                    blocks=(dev.asi8-dev.asi8.min())//pd.Timedelta(days=days).value
                    agg=pd.DataFrame(dict(delta=delta.to_numpy(),block=blocks)).groupby('block').delta.agg(['sum','count']);ids=np.random.default_rng(48).integers(0,len(agg),(3000,len(agg)))
                    draws=agg['sum'].to_numpy()[ids].sum(1)/agg['count'].to_numpy()[ids].sum(1);lo,hi=np.quantile(draws,[.025,.975])
                    uncertainty.append(dict(model=name,reference=ref,block_days=days,mae_change=float(delta.mean()),lower=lo,upper=hi))
    pd.DataFrame(uncertainty).to_csv(OUT/'uncertainty.csv',index=False)
    manifest=json.loads((OUT/'MANIFEST.json').read_text());assert all(s.q.sha(path)==digest for path,digest in manifest['sources'].items())
    summary=scores.to_string(index=False)+'\n\nAccepted changes: '+str([(c['method'],c['issue'],c['selected_config']) for c in choices if c['change_accepted']])+'\nSiLU retained. Historical exploratory comparison; feature addition and validation retuning are both included.\n'
    (OUT/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    s.dump(OUT/'AUDIT.json',dict(status='PASS',rows=len(dev),all_default_and_selected_predictions_replayed=True,confirmed_validation_predictions_replayed=True,selection_gate_recomputed=True,sources_unchanged=True))
    s.dump(OUT/'COMPLETE.json',dict(status='EXECUTED_AND_REPLAYED',scores=scores.to_dict('records')));s.dump(OUT/'STATUS.json',dict(stage='complete'));print(summary,flush=True)

def run():
    prepare()
    if (OUT/'COMPLETE.json').exists():print((OUT/'SUMMARY.txt').read_text());return
    def launch(job):
        method,i=job
        with (OUT/f'{method}_{i}.log').open('a') as log:r=subprocess.run([sys.executable,__file__,'worker',method,str(i)],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'Failed {job}; inspect worker log')
        print('FINISHED',method,i,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(launch,[(m,i) for i in range(4) for m in CONFIGS]))
    summarize()

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if len(sys.argv)>1 and sys.argv[1]=='worker':worker(sys.argv[2],int(sys.argv[3]))
        elif len(sys.argv)>1 and sys.argv[1]=='prepare':prepare()
        elif len(sys.argv)>1 and sys.argv[1]=='summarize':summarize()
        else:run()
