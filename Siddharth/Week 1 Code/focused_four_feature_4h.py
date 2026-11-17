"""Screen 15 four-feature packs cheaply; fully fit validation finalists only."""
import os
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[n]='1'
import sys,json,itertools,subprocess,concurrent.futures,importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import joblib,torch
import four_feature_4h as old
s=old.s;e=old.e;h=old.h
OUT=s.HERE/'experiments/focused_four_feature_4h_v1'
EXTRAS=[c for c in e.COLS if c not in s.PAIR]
PACKS={f'p{i:02d}':sorted(s.PAIR+list(pair)) for i,pair in enumerate(itertools.combinations(EXTRAS,2),1)}
ANCHOR=next(k for k,cols in PACKS.items() if cols==old.COLS)
METHODS=['rf','mlp']
def configure(method):
    if method=='mlp':e.configure('mlp',dict(widths=[8,4],dropout=.05));s.m.Net=e.TunedNet
    else:e.configure('rf',{})
def load():
    x,y,dev,base=joblib.load(old.old.OUT/'prepared.joblib')
    return x[e.COLS],y,dev,base
def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'MANIFEST.json').exists():
        manifest=json.loads((OUT/'MANIFEST.json').read_text())
        assert all(s.q.sha(p)==v for p,v in manifest['sources'].items());return
    x,y,dev,base=load()
    assert len(PACKS)==15 and all(len(c)==4 for c in PACKS.values()) and len(dev)==18481
    assert x.index.equals(y.index) and (y.label_end==y.index+pd.Timedelta(hours=4)).all()
    np.testing.assert_allclose(y.loc[dev,'magnitude_log_bps'],base.truth,atol=1e-8,rtol=0)
    for issue in s.q.STARTS:
        inner,ik,vk,ot,te=s.q.windows(y,dev,issue)
        assert y.loc[ik,'label_end'].max()<inner and y.loc[vk,'label_end'].max()<issue and y.loc[ot,'label_end'].max()<issue
    s.dump(OUT/'CONTRACT.json',dict(target='4h absolute endpoint log return in log bps',packs=PACKS,anchor=ANCHOR,
        screen='All15 combinations, training subset up to6144 evenly spaced rows; hybrid RF24 trees and original median regression; SiLU MLP16 epochs and seed48. Same preceding purged validation for each pack.',
        shortlist='Top3 screen MAE per model and window plus anchor, ties by pack ID.',
        full='Full expanding training and original RF recipe/validation blend or fixed 8/4 SiLU/dropout.05 MLP with seeds48/149/250 and max48 epochs. Reuse anchor inner/outer artifacts from prior four-feature run.',
        selection='Lowest full preceding-validation MAE among shortlist; selected before outer predictions. Full outer refits for selected pack only. Anchor outer predictions from prior completed run on same rows.',
        output='Selected strategy and anchor on same18481 evaluation rows; screen scores for all packs and full validation scores for shortlist. No claim that all packs received outer evaluation.',
        limitation='Already exposed historical evaluation; screened candidates may be missed by reduced fits; exploratory, not independent confirmation.'))
    paths=[Path(__file__),Path(old.__file__),Path(e.__file__),Path(s.__file__),Path(s.m.__file__),Path(s.q.__file__),Path(s.q.b.__file__),old.old.OUT/'prepared.joblib',old.OUT/'predictions.csv']
    fw=Path('C:/Users/Siddh/Desktop/ML/ml_framework')
    docs=[s.ROOT/'PROJECT_CONTEXT.md',s.HERE/'PROJECT_CONTEXT.md']+[fw/n for n in ['AGENTS.md','CONTEXT.md','docs/INDEX.md','docs/methods/WORKFLOW.md','docs/methods/DATA_AND_METRICS.md','docs/findings/RECORDED_RESULTS.md','jobs.json']]
    s.dump(OUT/'MANIFEST.json',dict(sources={str(p):s.q.sha(p) for p in paths},instructions={str(p):s.q.sha(p) for p in docs},versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','torch','scikit-learn','joblib']}))
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())

def fit_full(method,cols,x,y,keys,cut,path,vk=None,chosen=None):
    configure(method)
    if method=='rf':return h.fit(x,y,keys,cut,cols,path,'original',vk=vk,weight=None if chosen is None else chosen['weight'])
    s.SEEDS=[48,149,250]
    return s.fit_pack('mlp',x,y,keys,cut,cols,path,vk=vk,chosen=chosen)

def audit(a,cols,y,keys,cut):
    assert a['cols']==cols and a['keys'].equals(keys) and a['cutoff']==cut
    assert y.loc[keys,'label_end'].max()<cut
    if a['method']=='mlp':
        assert len(a['neural'])==3 and [z['seed'] for z in a['neural']]==[48,149,250]
        assert all(z['train_keys'].equals(keys) for z in a['neural'])

def anchor_bundle(method,issue,phase):
    root=old.OUT/method/f'{issue:%Y%m%d}'
    if method=='rf':return root/('screen/c0' if phase=='inner' else 'outer/c0')/'bundle.joblib'
    return root/('confirmed/c0' if phase=='inner' else 'outer/c0')/'bundle.joblib'

def worker(method,i):
    torch.set_num_threads(1);x,y,dev,base=load();issue=s.q.STARTS[i]
    inner,ik,vk,ot,te=s.q.windows(y,dev,issue)
    f=OUT/method/f'{issue:%Y%m%d}';f.mkdir(parents=True,exist_ok=True)
    if (f/'DONE.json').exists():return
    small=ik[np.unique(np.linspace(0,len(ik)-1,min(6144,len(ik)),dtype=int))]
    truth=y.loc[vk,'magnitude_log_bps'];screen=[];configure(method)
    for pack,cols in PACKS.items():
        s.dump(f/'STATUS.json',dict(stage='screen',completed=len(screen),total=len(PACKS),pack=pack))
        a=s.fit_pack(method,x,y,small,inner,cols,f/'screen'/pack,vk=vk,screen=True)
        assert a['cols']==cols and a['keys'].equals(small)
        row=dict(pack=pack,extras=[c for c in cols if c not in s.PAIR],**s.metrics(s.predict(a,x,vk),truth))
        screen.append(row);s.dump(f/'SCREEN.json',screen)
        print(method,i,'SCREEN',pack,flush=True)
    shortlist=[z['pack'] for z in sorted(screen,key=lambda z:(z['mae'],z['pack']))[:3]]
    if ANCHOR not in shortlist:shortlist.append(ANCHOR)
    s.dump(f/'SHORTLIST.json',shortlist)
    full=[];arts={}
    for pack in shortlist:
        s.dump(f/'STATUS.json',dict(stage='full_validation',completed=len(full),total=len(shortlist),pack=pack))
        cols=PACKS[pack];configure(method)
        if pack==ANCHOR:a=joblib.load(anchor_bundle(method,issue,'inner'));path=anchor_bundle(method,issue,'inner')
        else:
            path=f/'full'/pack/'inner'/'bundle.joblib'
            a=fit_full(method,cols,x,y,ik,inner,path.parent,vk=vk)
        audit(a,cols,y,ik,inner)
        pred=s.predict(a,x,vk)
        full.append(dict(pack=pack,extras=[c for c in cols if c not in s.PAIR],path=str(path),**s.metrics(pred,truth)))
        arts[pack]=a;s.dump(f/'FULL_VALIDATION.json',full)
    choice=min(full,key=lambda z:(z['mae'],z['pack']))
    s.dump(f/'SELECTION.json',dict(issue=str(issue),method=method,selected=choice,selection_before_outer_refit=True))
    pack=choice['pack'];cols=PACKS[pack];configure(method)
    if pack==ANCHOR:a=joblib.load(anchor_bundle(method,issue,'outer'));path=anchor_bundle(method,issue,'outer')
    else:
        path=f/'full'/pack/'outer'/('fit' if method=='rf' else '')/'bundle.joblib'
        a=fit_full(method,cols,x,y,ot,issue,path.parent,chosen=arts[pack])
    audit(a,cols,y,ot,issue)
    if method=='mlp':assert a['epochs']==arts[pack]['epochs']
    else:assert a['weight']==arts[pack]['weight']
    pred=s.predict(a,x,te)
    np.testing.assert_allclose(pred,s.predict(joblib.load(path),x,te),atol=0,rtol=0)
    old_p=pd.read_csv(old.OUT/method/f'{issue:%Y%m%d}'/'predictions.csv',index_col=0,parse_dates=True)
    # The prior four-feature run stores per-window predictions in this folder.
    assert old_p.index.equals(te)
    anchor=old_p.default.to_numpy()
    if pack==ANCHOR:np.testing.assert_allclose(pred,anchor,atol=1e-8,rtol=0)
    pd.DataFrame(dict(truth=y.loc[te,'magnitude_log_bps'],selected=pred,anchor=anchor),index=te).to_csv(f/'predictions.csv')
    s.dump(f/'DONE.json',dict(status='EXECUTED_AND_REPLAYED',selected_pack=pack,rows=len(te),outerpath=str(path)))
    s.dump(f/'STATUS.json',dict(stage='complete'))

def summarize():
    x,y,dev,base=load();choices=[];frames=[];screen=[];full=[]
    for method in METHODS:
        model='hybrid' if method=='rf' else 'mlp';parts=[]
        for issue in s.q.STARTS:
            f=OUT/method/f'{issue:%Y%m%d}';done=json.loads((f/'DONE.json').read_text());choice=json.loads((f/'SELECTION.json').read_text())
            trials=json.loads((f/'FULL_VALIDATION.json').read_text());short=json.loads((f/'SHORTLIST.json').read_text())
            assert choice['selected']==min(trials,key=lambda z:(z['mae'],z['pack']))
            assert set(short)==set(z['pack'] for z in trials)
            for row in json.loads((f/'SCREEN.json').read_text()):screen.append(dict(model=model,issue=str(issue),**row))
            for row in trials:full.append(dict(model=model,issue=str(issue),**row))
            p=pd.read_csv(f/'predictions.csv',index_col=0,parse_dates=True)
            configure(method)
            a=joblib.load(done['outerpath']);audit(a,PACKS[done['selected_pack']],y,a['keys'],issue)
            np.testing.assert_allclose(s.predict(a,x,p.index),p.selected,atol=1e-8,rtol=0)
            np.testing.assert_allclose(p.truth,y.loc[p.index,'magnitude_log_bps'],atol=1e-8,rtol=0)
            old_p=pd.read_csv(old.OUT/method/f'{issue:%Y%m%d}'/'predictions.csv',index_col=0,parse_dates=True)
            np.testing.assert_allclose(p.anchor,old_p.default,atol=1e-8,rtol=0)
            p['model']=model;p['issue']=str(issue);parts.append(p)
            choices.append(dict(model=model,issue=str(issue),selected=done['selected_pack'],extras=[c for c in PACKS[done['selected_pack']] if c not in s.PAIR],validation_mae=choice['selected']['mae']))
        joined=pd.concat(parts);assert joined.index.equals(dev);frames.append(joined)
    predictions=pd.concat(frames);predictions.to_csv(OUT/'predictions.csv')
    rows=[]
    for model,g in predictions.groupby('model',sort=False):
        for name in ['selected','anchor']:rows.append(dict(model=model,strategy=name,**s.metrics(g[name],g.truth)))
    scores=pd.DataFrame(rows);scores.to_csv(OUT/'scores.csv',index=False)
    pd.DataFrame(screen).to_csv(OUT/'screen_scores.csv',index=False)
    pd.DataFrame(full).to_csv(OUT/'full_validation_scores.csv',index=False)
    s.dump(OUT/'SELECTIONS.json',choices)
    manifest=json.loads((OUT/'MANIFEST.json').read_text())
    assert all(s.q.sha(path)==digest for path,digest in manifest['sources'].items())
    summary=scores.to_string(index=False)+'\n\nSelected packs by model/window:\n'+pd.DataFrame(choices).to_string(index=False)+'\n\nOnly shortlisted packs received full validation fits; only selected packs received new outer refits. Historical exploratory evaluation.\n'
    (OUT/'SUMMARY.txt').write_text(summary,encoding='utf-8')
    s.dump(OUT/'AUDIT.json',dict(status='PASS',rows_per_model=len(dev),screened_packs_per_model_window=15,all_predictions_replayed=True,anchor_matches_previous=True,selection_recomputed=True,sources_unchanged=True))
    s.dump(OUT/'COMPLETE.json',dict(status='EXECUTED_AND_REPLAYED',scores=rows,selection=choices))
    s.dump(OUT/'STATUS.json',dict(stage='complete'));print(summary,flush=True)

def run():
    prepare()
    if (OUT/'COMPLETE.json').exists():print((OUT/'SUMMARY.txt').read_text());return
    def launch(job):
        method,i=job
        with (OUT/f'{method}_{i}.log').open('a') as log:r=subprocess.run([sys.executable,__file__,'worker',method,str(i)],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'Worker failed {job}; inspect log')
        print('FINISHED',method,i,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(launch,[(m,i) for i in range(4) for m in METHODS]))
    summarize()

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if len(sys.argv)>1 and sys.argv[1]=='worker':worker(sys.argv[2],int(sys.argv[3]))
        elif len(sys.argv)>1 and sys.argv[1]=='prepare':prepare()
        elif len(sys.argv)>1 and sys.argv[1]=='summarize':summarize()
        else:run()
