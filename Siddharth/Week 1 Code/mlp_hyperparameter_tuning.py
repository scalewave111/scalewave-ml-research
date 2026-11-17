"""Nested small-MLP hyperparameter tuning; exactly two original liquidation inputs."""
from pathlib import Path
import copy,itertools,json,time,importlib.metadata
import numpy as np
import pandas as pd
import torch
from torch import nn
import joblib
from sklearn.linear_model import QuantileRegressor
from threadpoolctl import threadpool_limits
import lstm_walk_forward as q

HERE=Path(__file__).resolve().parent
OUT=HERE/'experiments/mlp_hyperparameter_tuning_v4'
CACHE=HERE/'experiments/lstm_feature_architecture_v3/prepared.joblib'
SEEDS=[48,149,250]
MAX_EPOCHS=100
LINEAR={}
def configs():
 anchors=[
 ([8],'silu',.001,.01,.05,1024,'median'),
 ([8,4],'relu',.001,.001,0.,1024,'linear'),
 ([16,8],'relu',.001,.001,0.,1024,'linear'),
 ([4],'tanh',.0003,.1,0.,1024,'median'),
 ([8],'relu',.003,.01,0.,256,'median'),
 ([16],'silu',.0003,.01,.05,1024,'median'),
 ([8,4],'tanh',.001,.1,.05,256,'median'),
 ([16,8],'silu',.001,.01,.15,1024,'median')]
 pool=list(itertools.product([(4,),(8,),(16,),(8,4),(16,8)],['relu','tanh','silu'],[.0001,.0003,.001,.003],[0.,.001,.01,.1],[0.,.05,.15],[256,1024]))
 rng=np.random.default_rng(20261104);rng.shuffle(pool)
 rows=list(anchors)
 for width,act,lr,decay,drop,batch in pool:
  item=(list(width),act,lr,decay,drop,batch,'median')
  if item not in rows:rows.append(item)
  if len(rows)==24:break
 return [dict(id=f'C{i:02d}',widths=w,activation=a,lr=l,decay=d,dropout=p,batch=b,initialization=z) for i,(w,a,l,d,p,b,z) in enumerate(rows)]
CONFIGS=configs()

class Net(nn.Module):
 def __init__(self,c,median):
  super().__init__();self.config=c;layers=[];n=2
  for h in c['widths']:
   layers.extend([nn.Linear(n,h),{'relu':nn.ReLU,'silu':nn.SiLU,'tanh':nn.Tanh}[c['activation']](),nn.Dropout(c['dropout'])]);n=h
  layers.append(nn.Linear(n,1));self.layers=nn.Sequential(*layers)
  nn.init.zeros_(self.layers[-1].weight);nn.init.constant_(self.layers[-1].bias,median/100)
 def forward(self,x):return self.layers(x).squeeze(1).clamp(min=0)
 def linear_start(self,coef,intercept):
  assert self.config['activation']=='relu' and all(h>=4 for h in self.config['widths'])
  linear=[a for a in self.layers if isinstance(a,nn.Linear)]
  with torch.no_grad():
   linear[0].weight[:4].copy_(torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]));linear[0].bias[:4].zero_()
   for a in linear[1:-1]:
    a.weight[:4].zero_();a.bias[:4].zero_();a.weight[:4,:4].copy_(torch.eye(4))
   linear[-1].weight.zero_();linear[-1].weight[0,:4].copy_(torch.tensor([coef[0],-coef[0],coef[1],-coef[1]],dtype=torch.float32))
   linear[-1].bias.fill_(intercept)

def design(x,keys,p):
 z=((x.loc[keys,p['columns']].fillna(p['median'])-p['median'])/p['scale']).clip(-8,8).to_numpy(np.float32)
 assert z.shape==(len(keys),2) and np.isfinite(z).all()
 return torch.from_numpy(z)

def infer(net,t):
 net.eval()
 with torch.no_grad():return np.concatenate([net(t[i:i+4096]).numpy().astype('float64')*100 for i in range(0,len(t),4096)])

def predict(a,x,keys):
 net=Net(a['config'],a['median']);net.load_state_dict(a['state'])
 return infer(net,design(x,keys,a['prep']))

def fit(x,y,keys,cutoff,c,seed,path,vk=None,epochs=None):
 assert y.loc[keys,'label_end'].max()<cutoff
 if path.exists():
  a=torch.load(path,weights_only=False)
  assert a['train_keys'].equals(keys) and a['cutoff']==cutoff and a['config']==c and a['seed']==seed
  assert (vk is None and a['validation_keys'] is None) or (vk is not None and a['validation_keys'].equals(vk))
  if epochs is not None:assert a['epoch']==epochs
  return a
 begin=time.monotonic();torch.manual_seed(seed);rng=np.random.default_rng(seed)
 p=q.prep(x,keys,q.b.PAIR);tx=design(x,keys,p);target=y.loc[keys,'magnitude_log_bps'].to_numpy(np.float32)
 ty=torch.from_numpy(target/100);median=float(np.median(target));net=Net(c,median)
 if c['initialization']=='linear':
  key=str(cutoff)
  if key not in LINEAR:
   reg=QuantileRegressor(quantile=.5,alpha=.0001,solver='highs').fit(tx.numpy().astype(float),ty.numpy().astype(float))
   LINEAR[key]=(reg.coef_,float(reg.intercept_))
  net.linear_start(*LINEAR[key])
 opt=torch.optim.AdamW([{'params':[p for p in net.parameters() if p.ndim>1],'weight_decay':c['decay']},{'params':[p for p in net.parameters() if p.ndim==1],'weight_decay':0.}],lr=c['lr'])
 vx=design(x,vk,p) if vk is not None else None
 vy=y.loc[vk,'magnitude_log_bps'].to_numpy() if vk is not None else None
 best=float(abs(infer(net,vx)-vy).mean()) if vx is not None else np.inf
 state=copy.deepcopy(net.state_dict());epoch_best=0;stale=0
 history=[dict(epoch=0,train_mae=float(abs(infer(net,tx)-target).mean()),validation_mae=best if vx is not None else None)]
 for epoch in range(1,(MAX_EPOCHS if epochs is None else epochs)+1):
  net.train();order=rng.permutation(len(keys));total=0.
  for k in range(0,len(keys),c['batch']):
   ids=order[k:k+c['batch']];opt.zero_grad(set_to_none=True)
   loss=(net(tx[ids])-ty[ids]).abs().mean();assert torch.isfinite(loss)
   loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1);opt.step();total+=float(loss.detach())*len(ids)*100
  val=float(abs(infer(net,vx)-vy).mean()) if vx is not None else None
  history.append(dict(epoch=epoch,train_mae=total/len(keys),validation_mae=val))
  if vx is None or val<best-.02:
   best=val if vx is not None else total/len(keys);state=copy.deepcopy(net.state_dict());epoch_best=epoch;stale=0
  else:stale+=1
  if vx is not None and stale>=10:break
 a=dict(state=state,config=c,seed=seed,median=median,prep=p,train_keys=keys,validation_keys=vk,cutoff=cutoff,epoch=epoch_best,history=history,parameters=sum(p.numel() for p in net.parameters()),seconds=time.monotonic()-begin)
 torch.save(a,path)
 return a

def folds(y,issue):
 out=[]
 for back in [3,2,1]:
  start=issue-pd.DateOffset(months=back);end=start+pd.DateOffset(months=1)
  train=y.index[y.label_end<start]
  valid=y.index[(y.index>=start)&(y.index<end)&(y.label_end<end)]
  assert len(train)>0 and len(valid)>0 and y.loc[valid,'label_end'].max()<end<=issue
  out.append((start,end,train,valid))
 return out

def run():
 begin=time.monotonic()
 if (OUT/'COMPLETE.json').exists():
  print((OUT/'COMPLETE.json').read_text(),flush=True);return
 OUT.mkdir(exist_ok=True)
 cache=joblib.load(CACHE);x=cache[0][q.b.PAIR].copy();y,dev,ref=cache[1:4];del cache
 assert x.index.equals(y.index) and x.index.is_unique and ref.index.equals(dev) and len(dev)==18481
 assert (y.label_end==y.index+pd.Timedelta(hours=48)).all()
 np.testing.assert_allclose(ref.truth,y.loc[dev,'magnitude_log_bps'],atol=1e-8,rtol=0)
 np.testing.assert_allclose(y.magnitude_log_bps,abs(y.signed_return_log_bps),atol=1e-8,rtol=0)
 contract=dict(features=q.b.PAIR,input_dimension=2,target='48h absolute endpoint log return in bps',configs=CONFIGS,seeds=SEEDS,
  outer_starts=list(q.STARTS),inner='Three nonoverlapping preceding monthly validation folds; expanding training; strict label_end purges at each start/end',
  screening='24 predeclared settings, seed48, pooled inner-fold MAE; shortlist top3',
  confirmation='Shortlisted settings rerun with seeds149,250; choose pooled MAE of 3-seed predictions, parameter-count tie break',
  outer_epochs='Per-seed median of three selected inner best epochs; all outer training rows retained',
  objective='MAE only; report Spearman and Pearson IC',max_epochs=MAX_EPOCHS,patience=10,min_improvement_bps=.02,
  optimizer='AdamW, weight decay on weight tensors only, gradient clip 1, constant learning rate',
  preprocessing='Train-only median imputation and IQR/std scaling; clip [-8,8]; exactly two inputs; no flags, lags, or additional features',
  stopping='One frozen search; no outer-test-driven retries; no wall-clock cap',
  source_sha256=q.sha(__file__),limitation='Previously examined historical evaluation, not untouched holdout')
 if (OUT/'CONTRACT.json').exists():assert json.loads((OUT/'CONTRACT.json').read_text())==json.loads(json.dumps(contract,default=str))
 else:
  q.dump(OUT/'CONTRACT.json',contract);(OUT/'training_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
  fw=Path('C:/Users/Siddh/Desktop/ML/ml_framework')
  docs=[HERE/'PROJECT_CONTEXT.md',fw.parent/'AGENTS.md']+[fw/p for p in ['AGENTS.md','CONTEXT.md','docs/INDEX.md','docs/methods/WORKFLOW.md','docs/methods/DATA_AND_METRICS.md','docs/findings/RECORDED_RESULTS.md','jobs.json']]
  q.dump(OUT/'MANIFEST.json',dict(source_sha256=q.sha(__file__),cache_sha256=q.sha(CACHE),baseline_sha256=q.sha(q.b.BASE/'report_baseline_predictions.csv'),
   instruction_hashes={str(p):q.sha(p) for p in docs},dependency_hashes={p:q.sha(HERE/p) for p in ['lstm_walk_forward.py','targeted_feature_extension.py','nested_ic_rf.py']},
   versions={p:importlib.metadata.version(p) for p in ['torch','numpy','pandas','scipy','scikit-learn','joblib']}))
 predictions=[];validations=[];decisions=[]
 for issue in q.STARTS:
  folder=OUT/issue.strftime('%Y%m%d');folder.mkdir(exist_ok=True);ff=folds(y,issue);cache={};screen=[]
  def candidate(c,seeds):
   pp=[];truth=[];epoch_map={str(seed):[] for seed in seeds};params=None
   for fi,(start,end,tk,vk) in enumerate(ff):
    preds=[]
    for seed in seeds:
     tag=(c['id'],fi,seed)
     if tag not in cache:cache[tag]=fit(x,y,tk,start,c,seed,folder/f"{c['id']}_f{fi}_s{seed}_inner.pt",vk=vk)
     a=cache[tag];preds.append(predict(a,x,vk));epoch_map[str(seed)].append(a['epoch']);params=a['parameters']
    pp.append(np.mean(preds,axis=0));truth.append(y.loc[vk,'magnitude_log_bps'].to_numpy())
   pred=np.concatenate(pp);truth=np.concatenate(truth)
   row=dict(config=c,parameters=params,epochs=epoch_map,fold_mae=[float(abs(p-y.loc[f[3],'magnitude_log_bps']).mean()) for p,f in zip(pp,ff)],**q.b.metric(pred,truth))
   return row,pp
  for c in CONFIGS:
   row,_=candidate(c,[48]);screen.append(row)
   q.dump(folder/'screening.json',screen)
   q.dump(OUT/'STATUS.json',dict(stage='screening',issue=str(issue),completed_configs=len(screen),total_configs=24,elapsed_seconds=time.monotonic()-begin))
   print('SCREEN',issue.date(),c['id'],'MAE',round(row['mae'],4),'elapsed',round(time.monotonic()-begin,1),flush=True)
  rank=lambda r:(r['mae'],r['parameters'],r['config']['id'])
  shortlist=sorted(screen,key=rank)[:3];q.dump(folder/'shortlist.json',shortlist);confirm=[];vp={}
  for row in shortlist:
   r,pp=candidate(row['config'],SEEDS);confirm.append(r);vp[r['config']['id']]=pp
   print('CONFIRM',issue.date(),r['config']['id'],'MAE',round(r['mae'],4),flush=True)
  q.dump(folder/'confirmation.json',confirm);winner=min(confirm,key=rank);c=winner['config']
  _,_,_,ot,te=q.windows(y,dev,issue)
  epochs={str(s):int(np.median(winner['epochs'][str(s)])) for s in SEEDS}
  decision=dict(issue=str(issue),selected=winner,outer_epochs=epochs,train_rows=len(ot),test_rows=len(te))
  q.dump(folder/'selection.json',decision) # Frozen before test prediction.
  outer=[]
  for seed in SEEDS:
   a=fit(x,y,ot,issue,c,seed,folder/f"{c['id']}_s{seed}_outer.pt",epochs=epochs[str(seed)])
   outer.append(predict(a,x,te))
  q.base_window(x,y,dev,ref,issue)
  predictions.append(pd.DataFrame(dict(mlp=np.mean(outer,axis=0),baseline=ref.loc[te,'report_hybrid'],issue=str(issue)),index=te))
  for pp,f in zip(vp[c['id']],ff):
   vk=f[3];validations.append(pd.DataFrame(dict(mlp=pp,truth=y.loc[vk,'magnitude_log_bps'],issue=str(issue),validation_start=str(f[0])),index=vk))
  decisions.append(decision);q.dump(OUT/'SELECTIONS.json',decisions)
  print('WINDOW FROZEN',issue.date(),c,'epochs',epochs,flush=True)
 p=pd.concat(predictions);assert p.index.equals(dev);p['truth']=y.loc[dev,'magnitude_log_bps'];v=pd.concat(validations)
 p.to_csv(OUT/'predictions.csv',index_label='decision_time');v.to_csv(OUT/'validation_predictions.csv',index_label='decision_time')
 scores=[dict(split='test',model=n,**q.b.metric(p[n],p.truth)) for n in ['baseline','mlp']]
 scores.append(dict(split='validation',model='mlp',**q.b.metric(v.mlp,v.truth)))
 pd.DataFrame(scores).to_csv(OUT/'scores.csv',index=False)
 ws=[dict(issue=i,model=n,**q.b.metric(f[n],f.truth)) for i,f in p.groupby('issue') for n in ['baseline','mlp']]
 pd.DataFrame(ws).to_csv(OUT/'window_scores.csv',index=False)
 # Check all artifacts, frozen selection, fold membership, preprocessing, and full outer replay.
 saved=pd.read_csv(OUT/'predictions.csv',index_col=0,parse_dates=True);assert saved.index.equals(dev)
 np.testing.assert_allclose(saved.baseline,ref.report_hybrid,rtol=0,atol=1e-8)
 np.testing.assert_allclose(saved.truth,y.loc[dev,'magnitude_log_bps'],rtol=0,atol=1e-8)
 count=0;error=0.
 for issue in q.STARTS:
  folder=OUT/issue.strftime('%Y%m%d');ff=folds(y,issue);_,_,_,ot,te=q.windows(y,dev,issue)
  s=json.loads((folder/'selection.json').read_text());conf=json.loads((folder/'confirmation.json').read_text())
  assert s['selected']==min(conf,key=rank)
  replay=[]
  for path in folder.glob('*.pt'):
   a=torch.load(path,weights_only=False);inner=path.stem.endswith('_inner')
   if inner:
    fi=int(path.stem.split('_f')[1].split('_')[0]);cut,end,tk,vk=ff[fi]
    assert a['validation_keys'].equals(vk) and y.loc[vk,'label_end'].max()<end<=issue
   else:
    cut,tk=issue,ot
    assert a['config']==s['selected']['config'] and a['epoch']==s['outer_epochs'][str(a['seed'])]
    assert a['epoch']==int(np.median(s['selected']['epochs'][str(a['seed'])]))
    replay.append(predict(a,x,te))
   assert a['train_keys'].equals(tk) and a['cutoff']==cut and y.loc[tk,'label_end'].max()<cut
   expected=q.prep(x,tk,q.b.PAIR);assert a['prep']['columns']==q.b.PAIR
   for field in ['median','scale']:pd.testing.assert_series_equal(a['prep'][field],expected[field])
   assert Net(a['config'],a['median']).layers[0].in_features==2
   count+=1
  rp=np.mean(replay,axis=0);error=max(error,float(abs(rp-saved.loc[te,'mlp']).max()))
  np.testing.assert_allclose(rp,saved.loc[te,'mlp'],atol=1e-10,rtol=0)
 assert count==372
 audit=dict(status='PASS',model_checkpoints=count,rows=len(p),input_dimension=2,strict_label_purges=True,training_only_preprocessing=True,selection_reproduced=True,baseline_reproduced=True,max_replay_error=error)
 q.dump(OUT/'AUDIT.json',audit)
 delta=abs(p.mlp-p.truth)-abs(p.baseline-p.truth);groups=(p.index.asi8-p.index.asi8.min())//pd.Timedelta(days=7).value
 blocks=pd.DataFrame(dict(delta=delta.to_numpy(),group=groups)).groupby('group').delta.agg(['sum','count'])
 ids=np.random.default_rng(2026).integers(0,len(blocks),(2000,len(blocks)))
 boot=blocks['sum'].to_numpy()[ids].sum(1)/blocks['count'].to_numpy()[ids].sum(1)
 test=pd.DataFrame(scores).query("split=='test'").set_index('model').to_dict('index')
 result=dict(status='EXECUTED_AND_AUDITED',audit=audit,test_scores=test,beats_baseline=bool(delta.mean()<0),mae_difference_mlp_minus_baseline=float(delta.mean()),weekly_block_bootstrap_95pct_ci=np.quantile(boot,[.025,.975]).tolist(),elapsed_this_invocation_seconds=time.monotonic()-begin,limitation=contract['limitation'])
 q.dump(OUT/'COMPLETE.json',result);q.dump(OUT/'STATUS.json',dict(stage='complete',**result))
 (OUT/'RESULTS.txt').write_text('Two-feature small MLP hyperparameter tuning. MAE selected settings; IC is Spearman.\n\n'+pd.DataFrame(scores).to_string(index=False)+'\n\n'+pd.DataFrame(ws).to_string(index=False)+'\n\n'+json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 with threadpool_limits(limits=2):run()

