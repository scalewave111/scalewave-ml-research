"""Causal baseline-error correction using exactly the original two liquidation inputs."""
from pathlib import Path
import copy,json,time,importlib.metadata
import numpy as np
import pandas as pd
import torch
from torch import nn
import joblib
from sklearn.base import clone
from threadpoolctl import threadpool_limits
import lstm_walk_forward as q
import mlp_hyperparameter_tuning as helper

HERE=Path(__file__).resolve().parent
OUT=HERE/'experiments/mlp_baseline_correction_v5'
CACHE=HERE/'experiments/lstm_feature_architecture_v3/prepared.joblib'
SEEDS=[48,149,250]
STRENGTHS=[.05,.1,.25,.5,1.]
CONFIGS=[dict(id=f'C{i}',widths=w,lr=lr,decay=.01,dropout=.05) for i,(w,lr) in enumerate(( (w,lr) for w in [[4],[8],[8,4]] for lr in [.0003,.001]))]
MAX_EPOCHS=60

def baseline_prep(x,keys):
 z=x.loc[keys,q.b.PAIR];med=z.median();filled=z.fillna(med)
 return dict(columns=q.b.PAIR,median=med,center=filled.mean(),scale=filled.std(ddof=0).replace(0,1.))

def baseline_model(x,y,issue,kind):
 path=OUT/'baseline_models'/f'{issue:%Y%m%d}_{kind}.joblib'
 keys=y.index[y.label_end<issue];expected=baseline_prep(x,keys)
 if path.exists():a=joblib.load(path)
 else:
  candidates=[q.b.BASE/'models'/f'{issue:%Y-%m}__report_{phase}_{kind}__expanding.joblib' for phase in ['outer','inner']]
  source=next((p for p in candidates if p.exists()),None)
  if source:
   a=joblib.load(source);origin=dict(reused_source=str(source),source_sha256=q.sha(source))
  else:
   template=q.b.BASE/'models'/f'2024-01__report_outer_{kind}__expanding.joblib'
   model=clone(joblib.load(template)['model'])
   if kind=='rf':model.set_params(n_jobs=2);assert len(keys)>=4096
   print('BASELINE FIT',issue.date(),kind,len(keys),flush=True)
   model.fit(q.b.design(x.loc[keys],expected),y.loc[keys,'magnitude_log_bps'])
   a=dict(model=model,prep=expected,train_keys=keys,issuance=issue)
   origin=dict(new_fit=True,template_source=str(template),template_sha256=q.sha(template))
  joblib.dump(a,path,compress=3)
  q.dump(path.with_suffix('.json'),dict(issue=str(issue),kind=kind,train_rows=len(keys),max_label_end=str(y.loc[keys,'label_end'].max()),**origin))
 assert a['train_keys'].equals(keys) and pd.Timestamp(a['issuance'])==issue
 assert y.loc[keys,'label_end'].max()<issue and a['prep']['columns']==q.b.PAIR
 for field in ['median','center','scale']:pd.testing.assert_series_equal(a['prep'][field],expected[field])
 return a

def baseline_oof(x,y,dev,ref):
 (OUT/'baseline_models').mkdir(exist_ok=True);pieces=[];decisions=[]
 starts=pd.date_range('2023-01-01','2024-10-01',freq='QS',tz='UTC')
 for issue in starts:
  inner=issue-pd.DateOffset(months=3);vk=y.index[(y.index>=inner)&(y.index<issue)&(y.label_end<issue)]
  ap=baseline_model(x,y,inner,'pair');ar=baseline_model(x,y,inner,'rf')
  pp=q.b.predict(ap,x.loc[vk]);pr=q.b.predict(ar,x.loc[vk]);truth=y.loc[vk,'magnitude_log_bps'].to_numpy()
  weight=min([0.,.1,.25,.5],key=lambda w:(float(abs((1-w)*pp+w*pr-truth).mean()),w))
  ap=baseline_model(x,y,issue,'pair');ar=baseline_model(x,y,issue,'rf')
  end=issue+pd.DateOffset(months=3) if issue!=starts[-1] else dev.max()+pd.Timedelta(minutes=30)
  keys=y.index[(y.index>=issue)&(y.index<end)]
  bp=(1-weight)*q.b.predict(ap,x.loc[keys])+weight*q.b.predict(ar,x.loc[keys])
  pieces.append(pd.DataFrame(dict(baseline=bp,baseline_issue=str(issue)),index=keys))
  decisions.append(dict(issue=str(issue),inner=str(inner),end=str(end),weight=weight,rows=len(keys),validation_rows=len(vk)))
  print('BASELINE OOF',issue.date(),'rows',len(keys),'weight',weight,flush=True)
 oof=pd.concat(pieces);assert oof.index.is_unique and oof.index.is_monotonic_increasing
 np.testing.assert_allclose(oof.loc[dev,'baseline'],ref.report_hybrid,rtol=0,atol=1e-7)
 oof.to_csv(OUT/'baseline_oof_predictions.csv',index_label='decision_time');q.dump(OUT/'BASELINE_SELECTIONS.json',decisions)
 return oof

class Net(nn.Module):
 def __init__(self,c):
  super().__init__();layers=[];n=2
  for h in c['widths']:layers.extend([nn.Linear(n,h),nn.Tanh(),nn.Dropout(c['dropout'])]);n=h
  layers.append(nn.Linear(n,1));self.layers=nn.Sequential(*layers)
  nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)
 def forward(self,x):return torch.tanh(self.layers(x)).squeeze(1) # Signed correction capped at +/-100 bps.

def infer(net,t):
 net.eval()
 with torch.no_grad():return np.concatenate([net(t[i:i+4096]).numpy().astype('float64')*100 for i in range(0,len(t),4096)])

def predict(a,x,keys):
 net=Net(a['config']);net.load_state_dict(a['state'])
 return infer(net,helper.design(x,keys,a['prep']))

def fit(x,y,oof,keys,cutoff,c,seed,path,vk=None,epochs=None):
 assert keys.isin(oof.index).all() and y.loc[keys,'label_end'].max()<cutoff
 if path.exists():
  a=torch.load(path,weights_only=False)
  assert a['train_keys'].equals(keys) and a['cutoff']==cutoff and a['config']==c and a['seed']==seed
  if epochs is not None:assert a['epoch']==epochs
  return a
 torch.manual_seed(seed);rng=np.random.default_rng(seed);start=time.monotonic()
 p=q.prep(x,keys,q.b.PAIR);tx=helper.design(x,keys,p)
 residual=(y.loc[keys,'magnitude_log_bps']-oof.loc[keys,'baseline']).to_numpy(np.float32);ty=torch.from_numpy(residual/100)
 net=Net(c);opt=torch.optim.AdamW([{'params':[a for a in net.parameters() if a.ndim>1],'weight_decay':c['decay']},{'params':[a for a in net.parameters() if a.ndim==1],'weight_decay':0}],lr=c['lr'])
 vx=helper.design(x,vk,p) if vk is not None else None
 vy=y.loc[vk,'magnitude_log_bps'].to_numpy() if vk is not None else None
 vb=oof.loc[vk,'baseline'].to_numpy() if vk is not None else None
 best=float(abs(vb-vy).mean()) if vk is not None else np.inf
 state=copy.deepcopy(net.state_dict());best_epoch=0;stale=0
 history=[dict(epoch=0,validation_mae=best if vk is not None else None)]
 for epoch in range(1,(MAX_EPOCHS if epochs is None else epochs)+1):
  net.train();order=rng.permutation(len(keys));total=0.
  for k in range(0,len(keys),512):
   ids=order[k:k+512];opt.zero_grad(set_to_none=True);loss=(net(tx[ids])-ty[ids]).abs().mean()
   assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1);opt.step();total+=float(loss.detach())*len(ids)*100
  val=float(abs(np.maximum(0,vb+infer(net,vx))-vy).mean()) if vk is not None else None
  history.append(dict(epoch=epoch,residual_train_mae=total/len(keys),validation_mae=val))
  if vk is None or val<best-.02:
   best=val if vk is not None else total/len(keys);best_epoch=epoch;state=copy.deepcopy(net.state_dict());stale=0
  else:stale+=1
  if vk is not None and stale>=8:break
 a=dict(state=state,config=c,seed=seed,prep=p,train_keys=keys,cutoff=cutoff,validation_keys=vk,training_residual=residual,epoch=best_epoch,history=history,parameters=sum(p.numel() for p in net.parameters()),seconds=time.monotonic()-start)
 torch.save(a,path);return a

def eligibility(row):
 return row['gain_bps']>=.1 and row['positive_months']>=2 and row['positive_seeds']>=2 and row['worst_month_degradation_bps']<=1.

def rank(row):return (row['mae'],row['strength'],row.get('parameters',0),row.get('id',''))

def run():
 begin=time.monotonic()
 if (OUT/'COMPLETE.json').exists():print((OUT/'COMPLETE.json').read_text(),flush=True);return
 OUT.mkdir(exist_ok=True)
 cached=joblib.load(CACHE);x=cached[0][q.b.PAIR].copy();y,dev,ref=cached[1:4];del cached
 assert x.index.equals(y.index) and ref.index.equals(dev) and len(dev)==18481 and x.index.is_unique
 assert (y.label_end==y.index+pd.Timedelta(hours=48)).all()
 np.testing.assert_allclose(ref.truth,y.loc[dev,'magnitude_log_bps'],atol=1e-8,rtol=0)
 contract=dict(features=q.b.PAIR,correction_input_dimension=2,baseline='Original median regression plus random forest recipe, including its missing-value handling; no new source features',
  correction_target='truth minus chronological held-out baseline prediction; no in-sample baseline residuals',
  baseline_oof_start='2023-01-01',warmup='Baseline uses earlier data; correction training starts with first causal OOF residuals in January 2023',
  outer_starts=list(q.STARTS),inner='Three preceding monthly folds; all training labels mature before issuance; validation labels mature before month end',
  configs=CONFIGS,seeds=SEEDS,strengths=[0.]+STRENGTHS,correction_cap_bps=100,max_epochs=MAX_EPOCHS,patience=8,
  selection='Nonzero requires pooled gain >=0.1 bps, improvements in >=2/3 months and >=2/3 seeds, worst monthly degradation <=1 bps. Lowest eligible pooled MAE; otherwise zero correction',
  checkpoint_selection='Full-strength corrected validation MAE, including zero-initialized epoch0; strength selected afterward',
  optimizer='AdamW; MAE residual loss/100; batch512; clip gradients1; weight decay excludes biases',
  outer_epochs='Per-seed median of selected three inner best epochs',
  stopping='One fixed experiment; no test-driven retries; no time cap',source_sha256=q.sha(__file__),
  limitation='Previously examined historical evaluation, not an untouched holdout')
 if (OUT/'CONTRACT.json').exists():assert json.loads((OUT/'CONTRACT.json').read_text())==json.loads(json.dumps(contract,default=str))
 else:
  q.dump(OUT/'CONTRACT.json',contract);(OUT/'training_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
  fw=Path('C:/Users/Siddh/Desktop/ML/ml_framework')
  docs=[HERE/'PROJECT_CONTEXT.md',fw.parent/'AGENTS.md']+[fw/p for p in ['AGENTS.md','CONTEXT.md','docs/INDEX.md','docs/methods/WORKFLOW.md','docs/methods/DATA_AND_METRICS.md','docs/findings/RECORDED_RESULTS.md','jobs.json']]
  q.dump(OUT/'MANIFEST.json',dict(source_sha256=q.sha(__file__),cache_sha256=q.sha(CACHE),baseline_reference_sha256=q.sha(q.b.BASE/'report_baseline_predictions.csv'),
   original_baseline_source_sha256=q.sha(q.b.BASE/'training_source_snapshot.py'),
   instruction_hashes={str(p):q.sha(p) for p in docs},dependency_hashes={p:q.sha(HERE/p) for p in ['mlp_hyperparameter_tuning.py','lstm_walk_forward.py','targeted_feature_extension.py','nested_ic_rf.py']},
   versions={p:importlib.metadata.version(p) for p in ['torch','numpy','pandas','scipy','scikit-learn','joblib']}))
 oof=baseline_oof(x,y,dev,ref);pieces=[];validations=[];decisions=[]
 for issue in q.STARTS:
  folder=OUT/issue.strftime('%Y%m%d');folder.mkdir(exist_ok=True)
  ff=[(s,e,tk[tk.isin(oof.index)],vk) for s,e,tk,vk in helper.folds(y,issue)]
  vk_all=ff[0][3].append([f[3] for f in ff[1:]])
  truth=y.loc[vk_all,'magnitude_log_bps'].to_numpy();bp=oof.loc[vk_all,'baseline'].to_numpy();base_mae=float(abs(bp-truth).mean())
  trials=[];cache={};corrections={}
  for c in CONFIGS:
   per_fold=[];epochs={str(seed):[] for seed in SEEDS}
   for fi,(start,end,tk,vk) in enumerate(ff):
    preds=[]
    for seed in SEEDS:
     a=fit(x,y,oof,tk,start,c,seed,folder/f"{c['id']}_f{fi}_s{seed}_inner.pt",vk=vk)
     cache[c['id'],fi,seed]=a;preds.append(predict(a,x,vk));epochs[str(seed)].append(a['epoch'])
    per_fold.append(np.stack(preds))
   seed_predictions=np.concatenate(per_fold,axis=1);mean_correction=seed_predictions.mean(axis=0);corrections[c['id']]=mean_correction
   for strength in STRENGTHS:
    pred=np.maximum(0,bp+strength*mean_correction)
    month_gains=[]
    for f,fc in zip(ff,per_fold):
     vk=f[3];mb=oof.loc[vk,'baseline'].to_numpy();my=y.loc[vk,'magnitude_log_bps'].to_numpy()
     month_gains.append(float(abs(mb-my).mean()-abs(np.maximum(0,mb+strength*fc.mean(axis=0))-my).mean()))
    seed_gains=[float(base_mae-abs(np.maximum(0,bp+strength*sc)-truth).mean()) for sc in seed_predictions]
    row=dict(id=c['id'],config=c,strength=strength,epochs=epochs,parameters=a['parameters'],gain_bps=float(base_mae-abs(pred-truth).mean()),
     monthly_gains=month_gains,seed_gains=seed_gains,positive_months=sum(g>0 for g in month_gains),positive_seeds=sum(g>0 for g in seed_gains),
     worst_month_degradation_bps=float(max(0,-min(month_gains))),**q.b.metric(pred,truth))
    row['eligible']=eligibility(row);trials.append(row)
   q.dump(folder/'TRIALS.json',trials)
   print('CORRECTION VALIDATED',issue.date(),c['id'],'best gain',round(max(r['gain_bps'] for r in trials if r['id']==c['id']),4),'elapsed',round(time.monotonic()-begin,1),flush=True)
   q.dump(OUT/'STATUS.json',dict(stage='validation',issue=str(issue),completed_configs=len(corrections),total_configs=len(CONFIGS)))
  good=[r for r in trials if r['eligible']]
  winner=min(good,key=rank) if good else dict(id='baseline_only',config=None,strength=0.,gain_bps=0.,epochs={},parameters=0,**q.b.metric(bp,truth))
  epochs={str(seed):int(np.median(winner['epochs'][str(seed)])) for seed in SEEDS} if good else {}
  decision=dict(issue=str(issue),selected=winner,outer_epochs=epochs,eligible_candidates=len(good),baseline_validation_mae=base_mae)
  q.dump(folder/'selection.json',decision)
  _,_,_,ot,te=q.windows(y,dev,issue);ot=ot[ot.isin(oof.index)]
  if good:
   outer=[]
   for seed in SEEDS:
    a=fit(x,y,oof,ot,issue,winner['config'],seed,folder/f"{winner['id']}_s{seed}_outer.pt",epochs=epochs[str(seed)])
    outer.append(predict(a,x,te))
   correction=np.mean(outer,axis=0)
  else:correction=np.zeros(len(te))
  baseline=ref.loc[te,'report_hybrid'].to_numpy();pred=np.maximum(0,baseline+winner['strength']*correction)
  pieces.append(pd.DataFrame(dict(baseline=baseline,corrected=pred,raw_correction=correction,strength=winner['strength'],issue=str(issue)),index=te))
  vc=corrections[winner['id']] if good else np.zeros(len(vk_all))
  validations.append(pd.DataFrame(dict(baseline=bp,corrected=np.maximum(0,bp+winner['strength']*vc),truth=truth,issue=str(issue)),index=vk_all))
  decisions.append(decision);q.dump(OUT/'SELECTIONS.json',decisions)
  print('WINDOW FROZEN',issue.date(),winner['id'],'strength',winner['strength'],'eligible',len(good),flush=True)
 p=pd.concat(pieces);assert p.index.equals(dev);p['truth']=y.loc[dev,'magnitude_log_bps'];v=pd.concat(validations)
 p.to_csv(OUT/'predictions.csv',index_label='decision_time');v.to_csv(OUT/'validation_predictions.csv',index_label='decision_time')
 scores=[dict(split=split,model=name,**q.b.metric(f[name],f.truth)) for split,f in [('validation',v),('test',p)] for name in ['baseline','corrected']]
 pd.DataFrame(scores).to_csv(OUT/'scores.csv',index=False)
 ws=[dict(issue=issue,model=n,**q.b.metric(f[n],f.truth)) for issue,f in p.groupby('issue') for n in ['baseline','corrected']]
 pd.DataFrame(ws).to_csv(OUT/'window_scores.csv',index=False)
 # Replay every baseline OOF segment, including selection of its earlier validation blend.
 base_count=0
 for d in json.loads((OUT/'BASELINE_SELECTIONS.json').read_text()):
  issue=pd.Timestamp(d['issue']);inner=pd.Timestamp(d['inner']);end=pd.Timestamp(d['end'])
  vk=y.index[(y.index>=inner)&(y.index<issue)&(y.label_end<issue)];keys=oof.index[(oof.index>=issue)&(oof.index<end)]
  ap=baseline_model(x,y,inner,'pair');ar=baseline_model(x,y,inner,'rf');pp=q.b.predict(ap,x.loc[vk]);pr=q.b.predict(ar,x.loc[vk])
  w=min([0.,.1,.25,.5],key=lambda w:(float(abs((1-w)*pp+w*pr-y.loc[vk,'magnitude_log_bps']).mean()),w))
  assert w==d['weight']
  ap=baseline_model(x,y,issue,'pair');ar=baseline_model(x,y,issue,'rf')
  rp=(1-w)*q.b.predict(ap,x.loc[keys])+w*q.b.predict(ar,x.loc[keys])
  np.testing.assert_allclose(rp,oof.loc[keys,'baseline'],atol=1e-10,rtol=0);base_count+=1
 saved=pd.read_csv(OUT/'predictions.csv',index_col=0,parse_dates=True);assert saved.index.equals(dev)
 np.testing.assert_allclose(saved.baseline,ref.report_hybrid,atol=1e-8,rtol=0)
 np.testing.assert_allclose(saved.truth,y.loc[dev,'magnitude_log_bps'],atol=1e-8,rtol=0)
 count=0;outer_count=0;error=0.
 for issue in q.STARTS:
  folder=OUT/issue.strftime('%Y%m%d');s=json.loads((folder/'selection.json').read_text());trials=json.loads((folder/'TRIALS.json').read_text())
  for row in trials:assert row['eligible']==eligibility(row)
  good=[r for r in trials if r['eligible']]
  if good:assert s['selected']==min(good,key=rank)
  else:assert s['selected']['strength']==0
  ff=[(a,b,tk[tk.isin(oof.index)],vk) for a,b,tk,vk in helper.folds(y,issue)]
  _,_,_,ot,te=q.windows(y,dev,issue);ot=ot[ot.isin(oof.index)];replay=[]
  for path in folder.glob('*.pt'):
   a=torch.load(path,weights_only=False);inner=path.stem.endswith('_inner')
   if inner:
    fi=int(path.stem.split('_f')[1].split('_')[0]);cut,end,tk,vk=ff[fi]
    assert a['validation_keys'].equals(vk) and y.loc[vk,'label_end'].max()<end<=issue
   else:
    cut,tk=issue,ot;outer_count+=1
    assert a['config']==s['selected']['config'] and a['epoch']==s['outer_epochs'][str(a['seed'])]
    assert a['epoch']==int(np.median(s['selected']['epochs'][str(a['seed'])]))
    replay.append(predict(a,x,te))
   assert a['train_keys'].equals(tk) and a['cutoff']==cut and y.loc[tk,'label_end'].max()<cut
   assert (pd.to_datetime(oof.loc[tk,'baseline_issue'],utc=True)<=tk).all()
   expected=q.prep(x,tk,q.b.PAIR);assert a['prep']['columns']==q.b.PAIR
   for field in ['median','scale']:pd.testing.assert_series_equal(a['prep'][field],expected[field])
   np.testing.assert_array_equal(a['training_residual'],(y.loc[tk,'magnitude_log_bps']-oof.loc[tk,'baseline']).to_numpy(np.float32))
   assert Net(a['config']).layers[0].in_features==2;count+=1
  correction=np.mean(replay,axis=0) if replay else np.zeros(len(te))
  rp=np.maximum(0,ref.loc[te,'report_hybrid'].to_numpy()+s['selected']['strength']*correction)
  error=max(error,float(abs(rp-saved.loc[te,'corrected']).max()))
  np.testing.assert_allclose(rp,saved.loc[te,'corrected'],atol=1e-10,rtol=0)
 assert count==216+outer_count
 audit=dict(status='PASS',model_checkpoints=count,outer_correction_models=outer_count,baseline_oof_segments=base_count,rows=len(p),correction_inputs=2,
  baseline_reproduced=True,held_out_baseline_residuals=True,strict_label_purges=True,training_only_preprocessing=True,eligibility_selection_reproduced=True,max_replay_error=error)
 q.dump(OUT/'AUDIT.json',audit)
 delta=abs(p.corrected-p.truth)-abs(p.baseline-p.truth);groups=(p.index.asi8-p.index.asi8.min())//pd.Timedelta(days=7).value
 blocks=pd.DataFrame(dict(delta=delta.to_numpy(),group=groups)).groupby('group').delta.agg(['sum','count'])
 ids=np.random.default_rng(2026).integers(0,len(blocks),(2000,len(blocks)));boot=blocks['sum'].to_numpy()[ids].sum(1)/blocks['count'].to_numpy()[ids].sum(1)
 test=pd.DataFrame(scores).query("split=='test'").set_index('model').to_dict('index')
 result=dict(status='EXECUTED_AND_AUDITED',audit=audit,test_scores=test,beats_baseline=bool(delta.mean()<0),
  mae_difference_corrected_minus_baseline=float(delta.mean()),weekly_block_bootstrap_95pct_ci=np.quantile(boot,[.025,.975]).tolist(),
  active_correction_windows=sum(d['selected']['strength']>0 for d in decisions),elapsed_this_invocation_seconds=time.monotonic()-begin,limitation=contract['limitation'])
 q.dump(OUT/'COMPLETE.json',result);q.dump(OUT/'STATUS.json',dict(stage='complete',**result))
 (OUT/'RESULTS.txt').write_text('Two-input MLP correction of causal held-out baseline errors.\n\n'+pd.DataFrame(scores).to_string(index=False)+'\n\n'+pd.DataFrame(ws).to_string(index=False)+'\n\n'+json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 with threadpool_limits(limits=2):run()

