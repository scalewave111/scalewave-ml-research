"""Two-input small MLP: frozen search, quarterly purged expanding walk-forward."""
from pathlib import Path
import copy, json, time, importlib.metadata
import numpy as np
import pandas as pd
import torch
from torch import nn
import joblib
from threadpoolctl import threadpool_limits
import lstm_walk_forward as q

HERE=Path(__file__).resolve().parent
OUT=HERE/'experiments/mlp_two_liquidation_warmstart_v2'
CACHE=HERE/'experiments/lstm_feature_architecture_v3/prepared.joblib'
ARCHS={'h8_4_slow':[8,4],'h8_4_fast':[8,4],'h16_8_slow':[16,8],'h16_8_fast':[16,8]}
LINEAR_CACHE={}
SEEDS=[48,149,250]
MAX_EPOCHS=128
BATCH=1024

class Net(nn.Module):
 def __init__(self,widths,median):
  super().__init__()
  self.first=nn.Linear(2,widths[0]);self.second=nn.Linear(widths[0],widths[1]);self.head=nn.Linear(widths[1],1)
 def forward(self,x):return self.head(torch.relu(self.second(torch.relu(self.first(x))))).squeeze(1).clamp(min=0)
 def warm_start(self,coef,intercept):
  with torch.no_grad():
   self.first.weight[:4].copy_(torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]))
   self.first.bias[:4].zero_();self.second.weight[:4].zero_();self.second.bias[:4].zero_()
   self.second.weight[:4,:4].copy_(torch.eye(4))
   self.head.weight.zero_();self.head.weight[0,:4].copy_(torch.tensor([coef[0],-coef[0],coef[1],-coef[1]],dtype=torch.float32))
   self.head.bias.fill_(float(intercept))

def initial_linear(tx,ty,cutoff):
 from sklearn.linear_model import QuantileRegressor
 key=str(cutoff)
 if key not in LINEAR_CACHE:
  # Exactly two columns; no flags, baseline predictions, or other regressors.
  linear=QuantileRegressor(quantile=.5,alpha=.0001,solver='highs').fit(tx.numpy().astype(float),ty.numpy().astype(float))
  LINEAR_CACHE[key]=(linear.coef_,float(linear.intercept_))
 return LINEAR_CACHE[key]

def design(x,keys,p):
 z=((x.loc[keys,p['columns']].fillna(p['median'])-p['median'])/p['scale']).clip(-8,8).to_numpy(np.float32)
 assert z.shape==(len(keys),2) and np.isfinite(z).all()
 return torch.from_numpy(z)

def infer(model,t):
 model.eval()
 with torch.no_grad():return np.concatenate([model(t[i:i+4096]).numpy().astype('float64')*100 for i in range(0,len(t),4096)])

def fit(x,y,keys,cutoff,name,seed,path,vk=None,epochs=None):
 assert y.loc[keys,'label_end'].max()<cutoff
 p=q.prep(x,keys,q.b.PAIR);tx=design(x,keys,p)
 target=y.loc[keys,'magnitude_log_bps'].to_numpy(np.float32)
 ty=torch.from_numpy(target/100);median=float(np.median(target))
 torch.manual_seed(seed);rng=np.random.default_rng(seed);model=Net(ARCHS[name],median)
 coef,intercept=initial_linear(tx,ty,cutoff);model.warm_start(coef,intercept)
 np.testing.assert_allclose(infer(model,tx),np.maximum(0,tx.numpy()@coef+intercept)*100,atol=1e-4,rtol=1e-6)
 decay=[v for v in model.parameters() if v.ndim>1];bias=[v for v in model.parameters() if v.ndim==1]
 opt=torch.optim.AdamW([{'params':decay,'weight_decay':.001},{'params':bias,'weight_decay':0}],lr=.0001 if name.endswith('slow') else .001)
 vx=design(x,vk,p) if vk is not None else None
 vy=y.loc[vk,'magnitude_log_bps'].to_numpy() if vk is not None else None
 best=float(np.mean(abs(infer(model,vx)-vy))) if vx is not None else np.inf
 stale=0;hist=[dict(epoch=0,train_mae=float(np.mean(abs(infer(model,tx)-target))),validation_mae=best if vx is not None else None)]
 state=copy.deepcopy(model.state_dict());best_epoch=0;start=time.monotonic()
 for epoch in range(1,(MAX_EPOCHS if epochs is None else epochs)+1):
  # No wall-clock limit: user explicitly removed it.
  model.train();order=rng.permutation(len(keys));total=0.
  for k in range(0,len(keys),BATCH):
   ids=order[k:k+BATCH];opt.zero_grad(set_to_none=True)
   loss=(model(tx[ids])-ty[ids]).abs().mean();assert torch.isfinite(loss)
   loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
   total+=float(loss.detach())*len(ids)*100
  val=float(np.mean(abs(infer(model,vx)-vy))) if vx is not None else None
  hist.append(dict(epoch=epoch,train_mae=total/len(keys),validation_mae=val))
  if vx is None or val<best-.01:
   best=val if vx is not None else total/len(keys);state=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
  else:stale+=1
  if vx is not None and stale>=12:break
 a=dict(state=state,prep=p,name=name,widths=ARCHS[name],seed=seed,median=median,
        train_keys=keys,cutoff=cutoff,validation_keys=vk,epoch=best_epoch,history=hist,
        parameters=sum(v.numel() for v in model.parameters()),seconds=time.monotonic()-start)
 torch.save(a,path)
 return a

def predict(a,x,keys):
 model=Net(a['widths'],a['median']);model.load_state_dict(a['state'])
 return infer(model,design(x,keys,a['prep']))

def run():
 # Fixed search, no wall-clock deadline.
 begin=time.monotonic()
 if (OUT/'CONTRACT.json').exists():raise RuntimeError('Run folder exists; preserve prior experiment')
 OUT.mkdir(parents=True,exist_ok=True)
 # Cache has other columns; only the original two are retained, preprocessed or supplied to models.
 cached=joblib.load(CACHE);x=cached[0][q.b.PAIR].copy();y,dev,ref=cached[1:4];del cached
 assert list(x.columns)==q.b.PAIR and x.index.equals(y.index)
 assert (y.label_end==y.index+pd.Timedelta(hours=48)).all()
 assert x.index.is_unique and ref.index.equals(dev) and len(dev)==18481
 np.testing.assert_allclose(y.magnitude_log_bps,abs(y.signed_return_log_bps),atol=1e-8,rtol=0)
 np.testing.assert_allclose(ref.truth,y.loc[dev,'magnitude_log_bps'],atol=1e-8,rtol=0)
 contract=dict(features=q.b.PAIR,input_dimension=2,no_added_indicators=True,no_lags=True,
  target='48 hour absolute endpoint log return, bps',architectures=ARCHS,seeds=SEEDS,
  outer_starts=list(q.STARTS),validation='preceding quarter, labels mature strictly before outer issuance',
  training='expanding, label_end strictly before model issuance',selection='lowest pooled prior-quarter MAE of three-seed mean; parameter count tie break',
  epochs=MAX_EPOCHS,early_stopping_patience=12,minimum_improvement_bps=.01,
  optimizer='AdamW lr .0001 or .001, weight decay .001 on weights only, MAE loss, gradient clip 1; include initialization epoch 0',
  dropout=0,activation='ReLU',initialization='Two-input median regression embedded exactly in MLP weights; alpha .0001 on target/100; initialization uses training data only',preprocessing='train-only median/IQR with std fallback; clip [-8,8]; exactly two values',
  stopping='one frozen search; no test-driven retries; no wall-clock cap per user instruction',
  limitation='Previously examined historical evaluation, not untouched holdout')
 q.dump(OUT/'CONTRACT.json',contract)
 (OUT/'training_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
 fw=Path('C:/Users/Siddh/Desktop/ML/ml_framework')
 docs=[HERE/'PROJECT_CONTEXT.md',fw.parent/'AGENTS.md']+[fw/p for p in ['AGENTS.md','CONTEXT.md','docs/INDEX.md','docs/methods/WORKFLOW.md','docs/methods/DATA_AND_METRICS.md','docs/findings/RECORDED_RESULTS.md','jobs.json']]
 q.dump(OUT/'MANIFEST.json',dict(source_sha256=q.sha(__file__),cache_sha256=q.sha(CACHE),
  baseline_sha256=q.sha(q.b.BASE/'report_baseline_predictions.csv'),
  instruction_hashes={str(p):q.sha(p) for p in docs},
  dependency_hashes={p:q.sha(HERE/p) for p in ['lstm_walk_forward.py','targeted_feature_extension.py','nested_ic_rf.py']},
  versions={p:importlib.metadata.version(p) for p in ['torch','numpy','pandas','scipy','joblib','scikit-learn']},
  missing_counts=x.isna().sum().to_dict(),unique_counts=x.nunique().to_dict()))
 predictions=[];validations=[];decisions=[];trials=[];fits=0
 for issue in q.STARTS:
  inner,ik,vk,ot,te=q.windows(y,dev,issue)
  assert y.loc[vk,'label_end'].max()<issue
  folder=OUT/issue.strftime('%Y%m%d');folder.mkdir()
  # Reconstruct saved baseline on identical training/test rows.
  _,bv,_,bw=q.base_window(x,y,dev,ref,issue)
  candidates={};models={}
  for name in ARCHS:
   models[name]=[]
   for seed in SEEDS:
    a=fit(x,y,ik,inner,name,seed,folder/f'{name}_{seed}_inner.pt',vk=vk)
    models[name].append(a);fits+=1
   vp=np.mean([predict(a,x,vk) for a in models[name]],axis=0)
   candidates[name]=vp
   row=dict(issue=str(issue),name=name,parameters=models[name][0]['parameters'],
    epochs=[a['epoch'] for a in models[name]],**q.b.metric(vp,y.loc[vk,'magnitude_log_bps']))
   trials.append(row)
   print('VALIDATED',issue.date(),name,'MAE',round(row['mae'],4),'epochs',row['epochs'],flush=True)
  selected=min([r for r in trials if r['issue']==str(issue)],key=lambda r:(r['mae'],r['parameters'],r['name']))
  name=selected['name']
  decision=dict(issue=str(issue),selected=selected,features=q.b.PAIR,
   train_rows=len(ot),validation_rows=len(vk),test_rows=len(te),
   max_training_label_end=str(y.loc[ot,'label_end'].max()),baseline_weight=bw)
  q.dump(folder/'selection.json',decision) # Freeze before outer test prediction.
  outer=[]
  for a in models[name]:
   path=folder/f"{name}_{a['seed']}_outer.pt"
   oa=fit(x,y,ot,issue,name,a['seed'],path,epochs=a['epoch']);fits+=1
   outer.append(predict(oa,x,te))
  predictions.append(pd.DataFrame(dict(mlp=np.mean(outer,axis=0),baseline=ref.loc[te,'report_hybrid'],issue=str(issue)),index=te))
  validations.append(pd.DataFrame(dict(mlp=candidates[name],baseline=bv,truth=y.loc[vk,'magnitude_log_bps'],issue=str(issue)),index=vk))
  decisions.append(decision);q.dump(OUT/'SELECTIONS.json',decisions);q.dump(OUT/'TRIALS.json',trials)
  print('WINDOW FROZEN',issue.date(),name,'elapsed',round(time.monotonic()-begin,1),flush=True)
 p=pd.concat(predictions);assert p.index.equals(dev)
 p['truth']=y.loc[dev,'magnitude_log_bps'];v=pd.concat(validations)
 p.to_csv(OUT/'predictions.csv',index_label='decision_time');v.to_csv(OUT/'validation_predictions.csv',index_label='decision_time')
 scores=[dict(split=split,model=n,**q.b.metric(f[n],f.truth)) for split,f in [('validation',v),('test',p)] for n in ['baseline','mlp']]
 pd.DataFrame(scores).to_csv(OUT/'scores.csv',index=False)
 ws=[dict(issue=issue,model=n,**q.b.metric(f[n],f.truth)) for issue,f in p.groupby('issue') for n in ['baseline','mlp']]
 pd.DataFrame(ws).to_csv(OUT/'window_scores.csv',index=False)
 # Independent checkpoint/preprocessing replay audit, including saved CSV predictions.
 saved=pd.read_csv(OUT/'predictions.csv',index_col=0,parse_dates=True);assert saved.index.equals(dev)
 np.testing.assert_allclose(saved.baseline,ref.report_hybrid,rtol=0,atol=1e-8)
 np.testing.assert_allclose(saved.truth,y.loc[dev,'magnitude_log_bps'],rtol=0,atol=1e-8)
 count=0;error=0.
 for issue in q.STARTS:
  inner,ik,vk,ot,te=q.windows(y,dev,issue);folder=OUT/issue.strftime('%Y%m%d');replay=[]
  selection=json.loads((folder/'selection.json').read_text())['selected']
  for path in folder.glob('*.pt'):
   a=torch.load(path,weights_only=False);is_inner=path.stem.endswith('_inner')
   keys,cut=(ik,inner) if is_inner else (ot,issue)
   assert a['train_keys'].equals(keys) and a['cutoff']==cut and y.loc[keys,'label_end'].max()<cut
   assert a['prep']['columns']==q.b.PAIR
   expected=q.prep(x,keys,q.b.PAIR)
   for f in ['median','scale']:pd.testing.assert_series_equal(a['prep'][f],expected[f])
   if is_inner:assert a['validation_keys'].equals(vk)
   else:
    assert a['name']==selection['name'] and a['epoch']==selection['epochs'][SEEDS.index(a['seed'])]
    replay.append(predict(a,x,te))
   count+=1
  rp=np.mean(replay,axis=0);error=max(error,float(np.max(abs(rp-saved.loc[te,'mlp']))))
  np.testing.assert_allclose(rp,saved.loc[te,'mlp'],rtol=0,atol=1e-10)
 # Paired weekly-block uncertainty: descriptive, does not undo historical test exposure.
 delta=(abs(p.mlp-p.truth)-abs(p.baseline-p.truth))
 group=(p.index.asi8-p.index.asi8.min())//pd.Timedelta(days=7).value
 blocks=pd.DataFrame(dict(delta=delta.to_numpy(),group=group)).groupby('group').delta.agg(['sum','count'])
 rng=np.random.default_rng(2026);ids=rng.integers(0,len(blocks),size=(2000,len(blocks)))
 boot=blocks['sum'].to_numpy()[ids].sum(axis=1)/blocks['count'].to_numpy()[ids].sum(axis=1)
 ci=np.quantile(boot,[.025,.975]).tolist()
 audit=dict(status='PASS',fits=count,rows=len(p),input_dimension=2,exact_feature_allowlist=True,
  baseline_reproduced=True,training_only_preprocessing=True,strict_label_purge=True,max_replay_error=error)
 assert count==fits==60
 q.dump(OUT/'AUDIT.json',audit)
 test=pd.DataFrame(scores).query("split=='test'").set_index('model')
 result=dict(status='EXECUTED_AND_AUDITED',audit=audit,test_scores=test.to_dict('index'),
  beats_baseline=bool(test.loc['mlp','mae']<test.loc['baseline','mae']),
  mae_difference_mlp_minus_baseline=float(delta.mean()),weekly_block_bootstrap_95pct_ci=ci,
  elapsed_seconds=time.monotonic()-begin,limitation=contract['limitation'])
 q.dump(OUT/'COMPLETE.json',result)
 report='Two liquidation input warm-start MLP: executed on real data and audited.\n\n'+pd.DataFrame(scores).to_string(index=False)+'\n\n'+pd.DataFrame(ws).to_string(index=False)+'\n\n'+json.dumps(result,indent=2)
 (OUT/'RESULTS.txt').write_text(report,encoding='utf-8')
 nb=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},cells=[
  dict(cell_type='markdown',metadata={},source=['# Two-input liquidation warm-start MLP results\n','Uses only the original 5m and 15m historical liquidation activity rates. Same quarterly expanding walk-forward, strict label purges. Previously examined evaluation period.\n']),
  dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=["from pathlib import Path\n","import pandas as pd, json\n","run = Path('experiments/mlp_two_liquidation_warmstart_v2')\n","display(pd.read_csv(run/'scores.csv'))\n","display(pd.read_csv(run/'window_scores.csv'))\n","print((run/'COMPLETE.json').read_text())\n"]),
  dict(cell_type='markdown',metadata={},source=['Training source: mlp_two_liquidation_warmstart.py. Completed run is protected against accidental overwriting. This notebook views saved results; it does not start training.'])])
 q.dump(HERE/'BTC_48h_MLP_Two_Liquidation_Warmstart.ipynb',nb)
 print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 with threadpool_limits(limits=2):run()

