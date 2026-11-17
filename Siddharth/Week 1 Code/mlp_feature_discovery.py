"""Prior-data feature screen followed by unchanged MLP feature-addition tests."""
from pathlib import Path
import json,time,sys,importlib.metadata
import numpy as np
import pandas as pd
import torch
from torch import nn
import joblib
from threadpoolctl import threadpool_limits
import linear_initialized_mlp as m
q=m.q
HERE=Path(__file__).resolve().parent
OUT=HERE/'experiments/mlp_feature_discovery_v1'
BASE=HERE/'experiments/linear_initialized_mlp_v1'
SRC=HERE/'experiments/baseline_single_feature_v6'
CORE=list(m.COLS)
DICTIONARY=q.b.old.PACKAGE/'data/feature_dictionary_and_admission.csv'

# Reuse the exact training engine; only generalize input dimension. Source files remain unchanged.
class Net(nn.Module):
    def __init__(self,coef,intercept):
        super().__init__();n=len(coef);self.linear=nn.Linear(n,1)
        with torch.no_grad():
            self.linear.weight.copy_(torch.as_tensor(coef,dtype=torch.float32).reshape(1,n));self.linear.bias.fill_(float(intercept))
        self.linear.requires_grad_(False)
        self.body=nn.Sequential(nn.Linear(n,8),nn.SiLU(),nn.Dropout(.05),nn.Linear(8,4),nn.SiLU(),nn.Dropout(.05))
        self.head=nn.Linear(4,1);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):return (self.linear(x)+self.head(self.body(x))).squeeze(1).clamp(min=0)

def design(x,keys,p):
    z=((x.loc[keys,p['columns']].fillna(p['median'])-p['median'])/p['scale']).clip(-8,8).to_numpy(np.float32)
    assert z.shape==(len(keys),len(p['columns'])) and np.isfinite(z).all()
    return torch.from_numpy(z)

m.Net=Net;m.design=design

def load():
    raw,y,dev,_=q.b.old.load();cached=joblib.load(SRC/'prepared.joblib')
    assert raw.index.equals(cached[0].index) and y.index.equals(cached[1].index)
    pd.testing.assert_frame_equal(raw[q.b.PAIR],cached[0][q.b.PAIR])
    raw[CORE[-1]]=cached[0][CORE[-1]]
    return raw.replace([np.inf,-np.inf],np.nan),y,dev

def reference(issue,phase,x,keys):
    folder=BASE/issue.strftime('%Y%m%d');pred=[]
    for seed in m.SEEDS:
        a=torch.load(folder/f'{seed}_{phase}.pt',weights_only=False)
        assert a['prep']['columns']==CORE
        pred.append(m.predict(a,x,keys))
    return np.mean(pred,axis=0)

def screen_one(x,y,ik,vk,tp,vp,c):
    tr=x.loc[ik,c];va=x.loc[vk,c];good=tr.notna();valid=va.notna()
    row=dict(feature=c,family=c.split('__')[1],training_coverage=float(good.mean()),validation_coverage=float(valid.mean()),unique_training=int(tr.nunique()),quality=False,eligible=False)
    if good.mean()<.8 or valid.mean()<.8 or good.sum()<1000 or tr.nunique()<10:return row,None
    corr=x.loc[ik,CORE+[c]].corr(method='spearman',min_periods=1000).loc[c,CORE].abs().max()
    row['max_core_rank_correlation']=float(corr)
    if not np.isfinite(corr) or corr>.95:return row,None
    cuts=np.unique(np.quantile(tr[good],[.2,.4,.6,.8]))
    groups=np.searchsorted(cuts,tr.to_numpy(),side='left');vg=np.searchsorted(cuts,va.to_numpy(),side='left')
    residual=y.loc[ik,'magnitude_log_bps'].to_numpy()-tp;center=float(np.median(residual));adjust=[];counts=[]
    for b in range(len(cuts)+1):
        mask=good.to_numpy()&(groups==b);n=int(mask.sum());counts.append(n)
        adjust.append(.5*n/(n+256)*(float(np.median(residual[mask]))-center) if n>=128 else 0.)
    correction=np.asarray(adjust)[vg];correction[~valid.to_numpy()]=0
    pred=np.maximum(0,vp+correction);truth=y.loc[vk,'magnitude_log_bps'].to_numpy()
    delta=abs(vp-truth)-abs(pred-truth);months=vk.strftime('%Y-%m')
    gains=[float(delta[months==month].mean()) for month in sorted(set(months))]
    row.update(quality=True,gain_bps=float(delta.mean()),monthly_wins=sum(a>.01 for a in gains),worst_month_gain=min(gains),mae=float(abs(pred-truth).mean()),rank_ic=q.b.metric(pred,truth)['rank_ic'],eligible=bool(delta.mean()>=.1 and sum(a>.01 for a in gains)>=2))
    return row,dict(cuts=cuts.tolist(),corrections=adjust,bin_counts=counts,monthly_gains=gains)

def screen():
    OUT.mkdir(exist_ok=True)
    if (OUT/'SCREEN_COMPLETE.json').exists():print((OUT/'SCREEN_COMPLETE.json').read_text());return
    contract=dict(universe='522 original masked F4Z fields, excluding current two liquidation fields; volatility ratio retained as core input. Availability/admission rechecked inside every fold; no full-history admission filtering.',
        core=CORE,target='BTCUSDT next48h absolute endpoint log return in log bps',
        screen='For each feature: five training-quantile bins, training residual median minus global residual median, half-strength shrinkage n/(n+256), minimum128/bin. Residuals from saved inner three-feature MLP predictions on its training rows. Missing values imply zero correction. Validate on prior purged quarter.',
        quality='>=80% usable values in inner training and prior validation, >=1000 training values, >=10 unique training values, max absolute training Spearman vs core<=.95.',
        selection='At most one feature per window: lowest proxy validation MAE among >=.1bps gain and >=2/3 months improving by >.01bps; feature-name tie break. Else keep original three-feature MLP. No test-based selection.',
        verification='Add selected feature to same linear-initialized 8/4 residual MLP, same three seeds, frozen fitted-linear skip, same optimizer/dropout/preprocessing/early stopping. Report challenger regardless of gate; primary uses challenger only if actual MLP prior-validation MAE gain>=.1bps and >=2 monthly wins>.01bps, otherwise original three-feature MLP.',
        scope='New-to-model raw features, not newly collected data. Cheap bin screen is a proxy, not evidence of neural improvement. Training residuals are in-sample; earlier validation also selected old MLP epochs, so validation is selection-used.',
        limitation='Broad multiple-feature search on exposed history; no untouched holdout. No test-driven retries.',
        budget_extension='User authorized this new feature research and MLP confirmation, superseding historical package initial-fit budgets; max24 neural checkpoints and8 linear fits.',source_sha=q.sha(__file__),engine_sha=q.sha(m.__file__))
    if (OUT/'CONTRACT.json').exists():assert json.loads((OUT/'CONTRACT.json').read_text())==contract
    else:q.dump(OUT/'CONTRACT.json',contract);(OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    x,y,dev=load();dictionary=pd.read_csv(DICTIONARY);columns=sorted(set(dictionary.feature)&set(x.columns)-set(CORE))
    assert len(dictionary)==522 and len(columns)==520
    fw=Path('C:/Users/Siddh/Desktop/ML/ml_framework');docs=[HERE/'PROJECT_CONTEXT.md',fw.parent/'AGENTS.md']+[fw/n for n in ['AGENTS.md','CONTEXT.md','docs/INDEX.md','docs/methods/WORKFLOW.md','docs/methods/DATA_AND_METRICS.md','docs/findings/RECORDED_RESULTS.md','jobs.json']]
    inputs=[DICTIONARY,SRC/'prepared.joblib',BASE/'predictions.csv',BASE/'validation_predictions.csv',BASE/'AUDIT.json']+[q.b.old.PACKAGE/'data'/n for n in ['X_train.parquet','X_development.parquet','y_train.parquet','y_development.parquet']]+list(BASE.glob('*/*.pt'))
    hashes={str(p):q.sha(p) for p in inputs}
    q.dump(OUT/'MANIFEST.json',dict(inputs=hashes,instructions={str(p):q.sha(p) for p in docs},versions={n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','scikit-learn','joblib']},dependencies={str(p):q.sha(p) for p in [Path(m.__file__),Path(q.__file__),Path(q.b.__file__)]}))
    priorv=pd.read_csv(BASE/'validation_predictions.csv',index_col=0,parse_dates=True)
    rows=[];decisions=[];short=[]
    for issue in q.STARTS:
        inner,ik,vk,ot,te=q.windows(y,dev,issue)
        assert y.loc[ik,'label_end'].max()<inner and y.loc[vk,'label_end'].max()<issue
        tp=reference(issue,'inner',x,ik);vp=reference(issue,'inner',x,vk)
        np.testing.assert_allclose(vp,priorv.loc[vk,'initialized_mlp'],atol=1e-8,rtol=0)
        wr=[];details={}
        for j,c in enumerate(columns):
            row,detail=screen_one(x,y,ik,vk,tp,vp,c);row['issue']=str(issue);wr.append(row)
            if detail is not None:details[c]=detail
            if j%100==0:q.dump(OUT/'STATUS.json',dict(stage='feature_screen',issue=issue,processed=j,total=len(columns)))
        df=pd.DataFrame(wr);eligible=df[df.eligible].sort_values(['mae','feature'])
        chosen=eligible.iloc[0].feature if len(eligible) else None
        ranked=df[df.quality].sort_values(['mae','feature']);ranked.head(10).to_csv(OUT/f'{issue:%Y%m%d}_shortlist.csv',index=False)
        for rank,(_,r) in enumerate(ranked.head(10).iterrows(),1):short.append(dict(rank=rank,**r.to_dict()))
        d=dict(issue=str(issue),feature=chosen,quality_candidates=int(df.quality.sum()),eligible_candidates=len(eligible),screen_result=eligible.iloc[0].to_dict() if chosen else None,screen_parameters=details.get(chosen))
        q.dump(OUT/f'{issue:%Y%m%d}_screen_selection.json',d);decisions.append(d);rows.extend(wr)
        print('SCREEN',issue.date(),'quality',d['quality_candidates'],'eligible',len(eligible),'chosen',chosen,'gain',d['screen_result']['gain_bps'] if chosen else None,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'all_feature_screen.csv',index=False);pd.DataFrame(short).to_csv(OUT/'shortlist.csv',index=False)
    selected={d['feature'] for d in decisions if d['feature']};dictionary[dictionary.feature.isin(selected)].to_csv(OUT/'selected_feature_definitions.csv',index=False)
    q.dump(OUT/'SCREEN_COMPLETE.json',dict(status='SCREEN_EXECUTED',candidate_count=len(columns),decisions=decisions,source_sha=q.sha(__file__)))

def run():
    screen();start=time.monotonic()
    if (OUT/'COMPLETE.json').exists():print((OUT/'RESULTS.txt').read_text());return
    contract=json.loads((OUT/'CONTRACT.json').read_text());assert contract['source_sha']==q.sha(__file__) and contract['engine_sha']==q.sha(m.__file__)
    selections=json.loads((OUT/'SCREEN_COMPLETE.json').read_text())['decisions'];x,y,dev=load()
    prior=pd.read_csv(BASE/'predictions.csv',index_col=0,parse_dates=True);priorv=pd.read_csv(BASE/'validation_predictions.csv',index_col=0,parse_dates=True)
    pieces=[];validations=[];decisions=[];fits=0;seedrows=[]
    for d in selections:
        issue=pd.Timestamp(d['issue']);inner,ik,vk,ot,te=q.windows(y,dev,issue)
        folder=OUT/issue.strftime('%Y%m%d');folder.mkdir(exist_ok=True);(folder/'linear').mkdir(exist_ok=True)
        pp=prior.loc[te,['baseline','vol_ratio','initialized_mlp','truth','issue']].rename(columns={'initialized_mlp':'mlp_3features'}).copy()
        vv=priorv.loc[vk,['baseline','vol_ratio','initialized_mlp','truth','issue']].rename(columns={'initialized_mlp':'mlp_3features'}).copy()
        feature=d['feature'];decision=dict(issue=str(issue),feature=feature,use_addition=False,epochs=[])
        if feature:
            m.COLS=CORE+[feature];m.OUT=folder
            q.dump(OUT/'STATUS.json',dict(stage='inner_linear',issue=issue,feature=feature))
            bi=m.linear_fit(x,y,ik,inner);inners=[]
            for seed in m.SEEDS:
                q.dump(OUT/'STATUS.json',dict(stage='inner_mlp',issue=issue,seed=seed,completed_checkpoints=fits))
                a=m.fit(x,y,bi,seed,folder/f'{seed}_inner.pt',vk=vk);inners.append(a);fits+=1
                print('FIT',fits,issue.date(),'inner',seed,'epoch',a['epoch'],flush=True)
            vv['challenger_mlp']=np.mean([m.predict(a,x,vk) for a in inners],axis=0)
            gain=abs(vv.mlp_3features-vv.truth)-abs(vv.challenger_mlp-vv.truth);months=vk.strftime('%Y-%m')
            monthly=[float(gain[months==month].mean()) for month in sorted(set(months))]
            decision.update(epochs=[a['epoch'] for a in inners],validation_gain=float(gain.mean()),monthly_gains=monthly,use_addition=bool(gain.mean()>=.1 and sum(g>.01 for g in monthly)>=2))
            q.dump(folder/'selection.json',decision) # Freeze primary rule before outer fit/scoring.
            q.dump(OUT/'STATUS.json',dict(stage='outer_linear',issue=issue,feature=feature))
            bo=m.linear_fit(x,y,ot,issue);preds=[]
            for a in inners:
                seed=a['seed'];q.dump(OUT/'STATUS.json',dict(stage='outer_mlp',issue=issue,seed=seed,completed_checkpoints=fits))
                b=m.fit(x,y,bo,seed,folder/f'{seed}_outer.pt',epochs=a['epoch']);fits+=1
                pred=m.predict(b,x,te);preds.append(pred);seedrows.append(dict(issue=str(issue),feature=feature,seed=seed,**q.b.metric(pred,pp.truth)))
                print('FIT',fits,issue.date(),'outer',seed,'epoch',b['epoch'],flush=True)
            pp['challenger_mlp']=np.mean(preds,axis=0)
        else:
            pp['challenger_mlp']=pp.mlp_3features;vv['challenger_mlp']=vv.mlp_3features;q.dump(folder/'selection.json',decision)
        for df in [pp,vv]:df['selected_mlp']=df.challenger_mlp if decision['use_addition'] else df.mlp_3features
        pieces.append(pp);validations.append(vv);decisions.append(decision)
        print('WINDOW',issue.date(),'validation accepted',decision['use_addition'],flush=True)
    p=pd.concat(pieces);v=pd.concat(validations);assert p.index.equals(dev) and len(p)==18481
    p.to_csv(OUT/'predictions.csv',index_label='decision_time');v.to_csv(OUT/'validation_predictions.csv',index_label='decision_time');q.dump(OUT/'SELECTIONS.json',decisions)
    names=['baseline','vol_ratio','mlp_3features','challenger_mlp','selected_mlp']
    scores=pd.DataFrame([dict(split=split,model=n,**q.b.metric(df[n],df.truth)) for split,df in [('validation',v),('test',p)] for n in names]);scores.to_csv(OUT/'scores.csv',index=False)
    windows=pd.DataFrame([dict(issue=i,model=n,**q.b.metric(df[n],df.truth)) for i,df in p.groupby('issue') for n in names]);windows.to_csv(OUT/'window_scores.csv',index=False);pd.DataFrame(seedrows).to_csv(OUT/'seed_window_scores.csv',index=False)
    uncertainty=[]
    for n in ['challenger_mlp','selected_mlp']:
        for ref in ['baseline','vol_ratio','mlp_3features']:
            delta=abs(p[n]-p.truth)-abs(p[ref]-p.truth);block=(p.index.asi8-p.index.asi8.min())//pd.Timedelta(days=7).value
            b=pd.DataFrame(dict(delta=delta.to_numpy(),block=block)).groupby('block').delta.agg(['sum','count']);ids=np.random.default_rng(20261104).integers(0,len(b),(5000,len(b)))
            boot=b['sum'].to_numpy()[ids].sum(1)/b['count'].to_numpy()[ids].sum(1);lo,hi=np.quantile(boot,[.025,.975]);uncertainty.append(dict(model=n,reference=ref,mae_difference=delta.mean(),lower=lo,upper=hi))
    pd.DataFrame(uncertainty).to_csv(OUT/'uncertainty.csv',index=False)
    # Audit screening choices, frozen skip, purges, preprocessing and checkpoint replay.
    q.dump(OUT/'STATUS.json',dict(stage='audit'));saved=pd.read_csv(OUT/'predictions.csv',index_col=0,parse_dates=True)
    np.testing.assert_allclose(saved.mlp_3features,prior.initialized_mlp,atol=1e-8,rtol=0);screened=pd.read_csv(OUT/'all_feature_screen.csv');count=0
    for d in decisions:
        issue=pd.Timestamp(d['issue']);inner,ik,vk,ot,te=q.windows(y,dev,issue);feature=d['feature'];folder=OUT/issue.strftime('%Y%m%d')
        eligible=screened[(screened.issue==d['issue'])&screened.eligible].sort_values(['mae','feature'])
        assert feature==(eligible.iloc[0].feature if len(eligible) else None)
        if feature:
            m.COLS=CORE+[feature]
            # Recompute the selected proxy from earlier data, independently of test outcomes.
            tp=reference(issue,'inner',x,ik);vp=reference(issue,'inner',x,vk)
            row,detail=screen_one(x,y,ik,vk,tp,vp,feature);assert row['eligible']
            np.testing.assert_allclose(row['gain_bps'],eligible.iloc[0].gain_bps,atol=1e-8,rtol=0)
            for phase,keys,cut,pk,df in [('inner',ik,inner,vk,v),('outer',ot,issue,te,saved)]:
                base=joblib.load(folder/'linear'/f'{cut:%Y%m%d}.joblib');assert base['train_keys'].equals(keys) and y.loc[keys,'label_end'].max()<cut
                assert base['prep']['columns']==m.COLS;expected=q.prep(x,keys,m.COLS)
                for f in ['median','scale']:pd.testing.assert_series_equal(base['prep'][f],expected[f])
                preds=[]
                for j,seed in enumerate(m.SEEDS):
                    a=torch.load(folder/f'{seed}_{phase}.pt',weights_only=False);assert a['train_keys'].equals(keys) and a['cutoff']==cut and a['epoch']==d['epochs'][j]
                    assert a['prep']['columns']==m.COLS
                    for f in ['median','scale']:pd.testing.assert_series_equal(a['prep'][f],expected[f])
                    init=Net(base['coef'],base['intercept'])
                    for f in ['weight','bias']:torch.testing.assert_close(a['state']['linear.'+f],init.linear.state_dict()[f],atol=0,rtol=0)
                    if phase=='inner':
                        assert a['validation_keys'].equals(vk);best=a['history'][0]['validation_mae'];ep=0
                        for r in a['history'][1:]:
                            if r['validation_mae']<best-.01:best=r['validation_mae'];ep=r['epoch']
                        assert ep==a['epoch']
                    else:assert a['validation_keys'] is None
                    preds.append(m.predict(a,x,pk));count+=1
                np.testing.assert_allclose(np.mean(preds,axis=0),df.loc[pk,'challenger_mlp'],atol=1e-8,rtol=0)
            z=v.loc[vk];gain=abs(z.mlp_3features-z.truth)-abs(z.challenger_mlp-z.truth);months=vk.strftime('%Y-%m')
            mg=[float(gain[months==month].mean()) for month in sorted(set(months))]
            assert d['use_addition']==bool(gain.mean()>=.1 and sum(g>.01 for g in mg)>=2)
        np.testing.assert_allclose(saved.loc[te,'selected_mlp'],saved.loc[te,'challenger_mlp' if d['use_addition'] else 'mlp_3features'],atol=0,rtol=0)
    manifest=json.loads((OUT/'MANIFEST.json').read_text());assert count==fits and all(q.sha(Path(path))==h for path,h in manifest['inputs'].items())
    audit=dict(status='PASS',neural_checkpoints=count,rows=len(p),screen_selection_replayed=True,neural_selection_replayed=True,strict_label_purges=True,training_only_preprocessing=True,frozen_linear_skip=True,checkpoint_replay=True,original_inputs_unchanged=True)
    q.dump(OUT/'AUDIT.json',audit)
    report='MLP feature discovery: executed and audited\n\n'+scores.to_string(index=False)+'\n\nSelected additions:\n'+pd.DataFrame(decisions).to_string(index=False)+'\n\nWindow scores:\n'+windows.to_string(index=False)+'\n\nDescriptive block bootstrap:\n'+pd.DataFrame(uncertainty).to_string(index=False)+'\n\nBroad historical feature search; no untouched confirmation. Proxy screen scores are not neural performance. Challenger uses the screened feature each window; primary selected_mlp adds it only when actual prior-validation MLP passes the gate.\n'
    (OUT/'RESULTS.txt').write_text(report,encoding='utf-8');q.dump(OUT/'COMPLETE.json',dict(status='EXECUTED_AND_AUDITED',audit=audit,seconds=time.monotonic()-start,scores=scores.to_dict('records')));q.dump(OUT/'STATUS.json',dict(stage='complete'));print(report,flush=True)

if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):
        screen() if len(sys.argv)>1 and sys.argv[1]=='screen' else run()
