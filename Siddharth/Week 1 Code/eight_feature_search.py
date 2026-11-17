"""Exhaustive eight-input screening, validation-only tuning, frozen outer evaluation."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[name]='1'
from pathlib import Path
import itertools,json,sys,subprocess,concurrent.futures,time
import numpy as np
import pandas as pd
import joblib,torch
from torch import nn
import search_report_features as s
import canonical_return_features as price
OUT=s.HERE/'experiments/eight_feature_exhaustive_v1'
COLS=s.PAIR+['F4Z__liquidation__cross_asset_liq_activity_raw_historical_active_rate__1h__value','PRICE__vol_ratio_4h_24h','PRICE__realized_vol_168h','F4Z__liquidation__venue_breadth_percentile__1h__value','F4Z__funding__funding_persistence_percentile__5m__value','F4Z__basis__basis_level_rz__1h__value']
CONFIG={}
original_dump=s.dump
def resilient_dump(path,obj):
    for attempt in range(40):
        try:return original_dump(path,obj)
        except PermissionError:
            if attempt==39:raise
            time.sleep(.25)
s.dump=resilient_dump
original_clone=s.clone
def tuned_clone(template):
    model=original_clone(template)
    if 'n_estimators' in model.get_params(): model.set_params(**CONFIG)
    return model
s.clone=tuned_clone
WIDTHS=[8,4];DROP=.05
class TunedNet(nn.Module):
    def __init__(self,coef,intercept):
        super().__init__();n=len(coef);self.linear=nn.Linear(n,1)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor(coef,dtype=torch.float32).reshape(1,n));self.linear.bias.fill_(intercept)
        self.linear.requires_grad_(False);layers=[]
        for width in WIDTHS: layers.extend([nn.Linear(n,width),nn.SiLU(),nn.Dropout(DROP)]);n=width
        self.body=nn.Sequential(*layers);self.head=nn.Linear(n,1);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):return (self.linear(x)+self.head(self.body(x))).squeeze(1).clamp(min=0)
s.m.Net=TunedNet
def configure(method,c):
    global CONFIG,WIDTHS,DROP
    CONFIG=c if method=='rf' else {};WIDTHS=c.get('widths',[8,4]);DROP=c.get('dropout',.05)
def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'prepared.joblib').exists():return
    x,y,dev,ref,_=joblib.load(s.OUT/'prepared.joblib')
    missing=[c for c in COLS if not c.startswith('PRICE__') and c not in x]
    if missing:
        raw,raw_y,raw_dev,_=s.q.b.old.load()
        assert raw.index.equals(x.index) and raw_dev.equals(dev)
        pd.testing.assert_frame_equal(raw_y,y)
        for c in missing:x[c]=raw[c]
    r=price.load_returns(s.ROOT/'Dataset/canonical_returns_5m.parquet')
    audit=price.validate(r,y);derived=price.features(r,x.index)
    for c in COLS:
        if c.startswith('PRICE__'):x[c]=derived[c]
    x=x[COLS].copy();assert len(COLS)==len(set(COLS))==8
    for issue in s.q.STARTS:
        inner,ik,vk,ot,te=s.q.windows(y,dev,issue)
        assert y.loc[ik,'label_end'].max()<inner and y.loc[vk,'label_end'].max()<issue
        assert all(x.loc[ik,c].notna().sum()>=100 and x.loc[ik,c].nunique()>=2 for c in COLS)
    joblib.dump((x,y,dev,ref),OUT/'prepared.joblib',compress=3)
    s.dump(OUT/'CONTRACT.json',dict(features=COLS,subsets_per_model_window=255,windows=4,screen_rows=6144,screen='Original reduced settings: RF24 trees and MLP16 epochs/one seed. Every subset considered.',tuning='Top four screening subsets plus original pair, yesterday three-feature winner and all eight; four configurations each. Full data, RF128 trees or MLP48epochs/three seeds.',selection='Minimum prior validation MAE; choices frozen before outer prediction',limitation='Previously exposed history, not untouched confirmation. Hyperparameters tuned on shortlisted subsets, not exhaustive joint subset/hyperparameter grid.',source_sha=s.q.sha(__file__),price_audit=audit))
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    paths=[s.OUT/'prepared.joblib',s.ROOT/'Dataset/canonical_returns_5m.parquet',Path(s.__file__),Path(s.m.__file__)]
    s.dump(OUT/'MANIFEST.json',dict(inputs={str(p):s.q.sha(p) for p in paths},versions=dict(python=sys.version,torch=torch.__version__),instructions={str(s.ROOT/'PROJECT_CONTEXT.md'):s.q.sha(s.ROOT/'PROJECT_CONTEXT.md')}))
def worker(method,i):
    torch.set_num_threads(1);x,y,dev,ref=joblib.load(OUT/'prepared.joblib');issue=s.q.STARTS[i]
    inner,ik,vk,ot,te=s.q.windows(y,dev,issue);f=OUT/method/f'{issue:%Y%m%d}';f.mkdir(parents=True,exist_ok=True)
    if (f/'DONE.json').exists():return
    small=ik[np.unique(np.linspace(0,len(ik)-1,min(6144,len(ik)),dtype=int))];truth=y.loc[vk,'magnitude_log_bps'];trials=[]
    configs=([{},dict(max_depth=2,min_samples_leaf=256),dict(max_depth=5,min_samples_leaf=64),dict(max_depth=3,min_samples_leaf=256,max_features=.75)] if method=='rf' else [dict(widths=[8,4],dropout=.05),dict(widths=[4],dropout=.15),dict(widths=[16,8],dropout=.15),dict(widths=[8],dropout=0.)])
    configure(method,configs[0])
    # User-authorized focused continuation: reuse every completed trial and add
    # all combinations of the three new candidates to each baseline anchor.
    prior=json.loads((f/'screen_trials.json').read_text()) if (f/'screen_trials.json').exists() else []
    candidates=[r['cols'] for r in prior]
    for anchor in [s.PAIR,s.PAIR+['PRICE__vol_ratio_4h_24h']]:
        for k in range(4):
            for extra in itertools.combinations(COLS[-3:],k):candidates.append(anchor+list(extra))
    candidates.extend([COLS,COLS[:5]])
    candidates=list({s.key(c):sorted(c) for c in candidates}.values())
    s.dump(f/'FOCUSED_CANDIDATES.json',candidates)
    for n,cols in enumerate(candidates,1):
        cols=sorted(cols);ident=s.key(cols)
        s.dump(f/'STATUS.json',dict(stage='focused_screen',completed=n-1,total=len(candidates)))
        a=s.fit_pack(method,x,y,small,inner,cols,f/'screen'/ident,vk=vk,screen=True)
        trials.append(dict(id=ident,cols=cols,**s.metrics(s.predict(a,x,vk),truth)))
        s.dump(f/'screen_trials.json',trials)
        print(method,i,'SCREEN',n,flush=True)
    choices=sorted(trials,key=lambda r:(r['mae'],len(r['cols']),r['id']))[:2]
    for cols in [s.PAIR,s.PAIR+['PRICE__vol_ratio_4h_24h'],COLS]:choices.append(next(r for r in trials if r['id']==s.key(cols)))
    choices=list({r['id']:r for r in choices}.values());full=[]
    configs=configs[:2]
    s.SEEDS=[48]
    for row in choices:
        for ci,c in enumerate(configs):
            configure(method,c);ident=row['id']+f'_c{ci}'
            s.dump(f/'STATUS.json',dict(stage='full_tuning',completed=len(full),total=len(choices)*len(configs)))
            a=s.fit_pack(method,x,y,ik,inner,row['cols'],f/'full'/ident/'inner',vk=vk)
            full.append(dict(id=ident,cols=row['cols'],config=c,**s.metrics(s.predict(a,x,vk),truth)))
            s.dump(f/'FULL_VALIDATION.json',full)
    best=min(full,key=lambda r:(r['mae'],len(r['cols']),r['id']));s.dump(f/'SELECTION.json',best)
    configure(method,best['config']);chosen=joblib.load(f/'full'/best['id']/'inner/bundle.joblib')
    if method=='mlp':
        s.SEEDS=[48,149,250]
        chosen=s.fit_pack(method,x,y,ik,inner,best['cols'],f/'confirmation'/best['id']/'inner',vk=vk)
    outer=s.fit_pack(method,x,y,ot,issue,best['cols'],f/'full'/best['id']/'outer',chosen=chosen)
    pred=s.predict(outer,x,te);replay=joblib.load(f/'full'/best['id']/'outer/bundle.joblib')
    np.testing.assert_allclose(pred,s.predict(replay,x,te),atol=0,rtol=0)
    assert outer['keys'].equals(ot) and y.loc[ot,'label_end'].max()<issue
    pd.DataFrame(dict(selected=pred,truth=y.loc[te,'magnitude_log_bps'],issue=str(issue)),index=te).to_csv(f/'predictions.csv')
    s.dump(f/'DONE.json',dict(status='EXECUTED_AND_REPLAYED',screened=len(trials),tuned=len(full),selected=best))
    s.dump(f/'STATUS.json',dict(stage='complete'))
def summarize():
    x,y,dev,ref=joblib.load(OUT/'prepared.joblib');p=pd.DataFrame(dict(truth=ref.truth,baseline=ref.report_hybrid),index=dev)
    old=pd.read_csv(s.HERE/'experiments/baseline_single_feature_v6/predictions.csv',index_col=0,parse_dates=True);assert old.index.equals(dev);p['yesterday_best']=old.vol_ratio
    for method in ['rf','mlp']:
        pieces=[]
        for issue in s.q.STARTS:
            f=OUT/method/f'{issue:%Y%m%d}';assert json.loads((f/'DONE.json').read_text())['screened']>=16
            frame=pd.read_csv(f/'predictions.csv',index_col=0,parse_dates=True);pieces.append(frame.selected)
        combined=pd.concat(pieces);assert combined.index.equals(dev);p[method]=combined
    scores=pd.DataFrame([dict(model=c,**s.metrics(p[c],p.truth)) for c in p if c!='truth']);p.to_csv(OUT/'predictions.csv');scores.to_csv(OUT/'scores.csv',index=False)
    assert abs(scores.set_index('model').loc['baseline','mae']-181.52599276479125)<1e-8
    total=sum(json.loads(p.read_text())['screened'] for p in OUT.glob('*/*/DONE.json'))
    s.dump(OUT/'COMPLETE.json',dict(status='EXECUTED_AND_REPLAYED',scores=scores.to_dict('records'),screened=total,limitation='User-authorized early stop of exhaustive screening; historical evaluation already exposed, not fresh confirmation.'))
    (OUT/'RESULTS.txt').write_text(scores.to_string(index=False),encoding='utf-8');print(scores.to_string(index=False),flush=True)
def run():
    prepare()
    def launch(job):
        method,i=job
        with (OUT/f'{method}_{i}.log').open('a') as log:r=subprocess.run([sys.executable,__file__,'worker',method,str(i)],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'Worker failed: {job}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(launch,[(m,i) for i in range(4) for m in ['rf','mlp']]))
    summarize()
if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        if sys.argv[1]=='worker':worker(sys.argv[2],int(sys.argv[3]))
        else:globals()[sys.argv[1]]()
