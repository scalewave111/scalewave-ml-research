"""LSTM 48h magnitude: same quarterly expanding walk-forward as baseline."""
from pathlib import Path
import argparse,copy,hashlib,importlib.metadata,json,time
import numpy as np
import pandas as pd
import torch
from torch import nn
import joblib
from threadpoolctl import threadpool_limits
import targeted_feature_extension as b
import canonical_return_features as cr
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=HERE/'experiments/lstm_walk_forward_v1'
STARTS=pd.to_datetime(['2024-01-01','2024-04-01','2024-07-01','2024-10-01'],utc=True)
FLOW='F4Z__flow__flow_activity_rz__4h__value'
RAW=b.PAIR+[FLOW,'F4Z__volatility__realized_variance_level_rz__4h__value','F4Z__volatility__realized_variance_level_rz__1d__value','F4Z__funding__funding_persistence_percentile__15m__value','F4Z__basis__basis_level_rz__1h__value']
CONFIGS={
 'pair_short':dict(pack='pair',steps=16,stride=1,hidden=16,decay=.01),
 'pair_medium':dict(pack='pair',steps=24,stride=4,hidden=16,decay=.01),
 'flow_medium':dict(pack='flow',steps=24,stride=4,hidden=16,decay=.01),
 'market_medium':dict(pack='market',steps=24,stride=4,hidden=16,decay=.02),
 'market_long':dict(pack='market',steps=42,stride=8,hidden=16,decay=.02),
 'market_wide':dict(pack='market',steps=24,stride=4,hidden=32,decay=.03)}
SEED=48
BATCH=512
MAX_EPOCHS=24
PATIENCE=6

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False),encoding='utf-8')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
 return h.hexdigest()

def load():
 raw,y,dev,_=b.old.load();r=cr.load_returns(ROOT/'Dataset/canonical_returns_5m.parquet');checks=cr.validate(r,y)
 prices=cr.features(r,raw.index).astype('float32');x=pd.concat([raw[RAW],prices],axis=1).replace([np.inf,-np.inf],np.nan)
 packs=dict(pair=b.PAIR,flow=b.PAIR+[FLOW],market=list(x.columns))
 ref=pd.read_csv(b.BASE/'report_baseline_predictions.csv',index_col=0,parse_dates=True)
 assert ref.index.equals(dev) and len(dev)==18481
 np.testing.assert_allclose(ref.truth,y.loc[dev,'magnitude_log_bps'],rtol=0,atol=1e-8)
 np.testing.assert_allclose(y.magnitude_log_bps,y.signed_return_log_bps.abs(),rtol=0,atol=1e-8)
 return x,y,dev,ref,packs,checks

def prep(x,keys,cols):
 a=x.loc[keys,cols];med=a.median().fillna(0);filled=a.fillna(med);scale=filled.quantile(.75)-filled.quantile(.25)
 scale=scale.where(scale>1e-6,filled.std(ddof=0)).fillna(1).clip(lower=1e-6)
 return dict(columns=cols,median=med,scale=scale)

def seq(x,keys,p,c):
 offsets=np.arange(c['steps']-1,-1,-1)*c['stride']*pd.Timedelta(minutes=30).value
 stamps=keys.asi8[:,None]-offsets[None,:];a=x[p['columns']].reindex(pd.to_datetime(stamps.ravel(),utc=True))
 z=((a.fillna(p['median'])-p['median'])/p['scale']).clip(-8,8)
 v=np.column_stack([z.to_numpy(np.float32),a.isna().to_numpy(np.float32)])
 assert np.isfinite(v).all() and (stamps<=keys.asi8[:,None]).all()
 return torch.from_numpy(v.reshape(len(keys),c['steps'],-1))

class Net(nn.Module):
 def __init__(self,n,h,median):
  super().__init__();self.lstm=nn.LSTM(n,h,batch_first=True);self.head=nn.Sequential(nn.Dropout(.1),nn.Linear(h+n,1))
  nn.init.normal_(self.head[1].weight,std=.01);nn.init.constant_(self.head[1].bias,float(np.log(np.expm1(median/100))))
 def forward(self,x):
  z,_=self.lstm(x)
  return nn.functional.softplus(self.head(torch.cat([z[:,-1],x[:,-1]],dim=1))).squeeze(1)

def infer(model,x):
 model.eval()
 with torch.no_grad():return np.concatenate([model(x[i:i+BATCH]).numpy()*100 for i in range(0,len(x),BATCH)])

def predict(a,x,keys):
 model=Net(a['inputs'],a['config']['hidden'],a['median']);model.load_state_dict(a['state'])
 return infer(model,seq(x,keys,a['prep'],a['config']))

def fit(x,y,keys,issue,name,cols,path,vk=None,epochs=None):
 c=CONFIGS[name];assert y.loc[keys,'label_end'].max()<issue
 if path.exists():
  a=torch.load(path,weights_only=False,map_location='cpu')
  assert a['train_keys'].equals(keys) and a['issuance']==issue and a['config']==c
  return a
 torch.manual_seed(SEED);rng=np.random.default_rng(SEED)
 p=prep(x,keys,cols);tx=seq(x,keys,p,c);target=y.loc[keys,'magnitude_log_bps'].to_numpy(np.float32)
 ty=torch.from_numpy(target/100);median=float(np.median(target));model=Net(tx.shape[-1],c['hidden'],median)
 opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=c['decay'])
 vx=seq(x,vk,p,c) if vk is not None else None
 vy=y.loc[vk,'magnitude_log_bps'].to_numpy() if vk is not None else None
 best=np.inf;best_epoch=0;stale=0;history=[];best_state=None;begin=time.monotonic()
 for epoch in range(1,(epochs or MAX_EPOCHS)+1):
  model.train();order=rng.permutation(len(keys));total=0.
  for start in range(0,len(keys),BATCH):
   idx=order[start:start+BATCH];opt.zero_grad(set_to_none=True);loss=(model(tx[idx])-ty[idx]).abs().mean()
   assert torch.isfinite(loss)
   loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=loss.item()*len(idx)*100
  val=float(abs(infer(model,vx)-vy).mean()) if vx is not None else None
  history.append(dict(epoch=epoch,train_mae=total/len(keys),validation_mae=val))
  if vx is None or val<best-.01:
   best=val if val is not None else total/len(keys);best_epoch=epoch;best_state=copy.deepcopy(model.state_dict());stale=0
  else:stale+=1
  if epoch==1 or epoch%4==0:print('EPOCH',issue.date(),path.stem,epoch,'val',val,'seconds',round(time.monotonic()-begin,1),flush=True)
  if vx is not None and stale>=PATIENCE:break
 a=dict(state=best_state,prep=p,name=name,config=c,train_keys=keys,issuance=issue,max_label_end=y.loc[keys,'label_end'].max(),epochs=best_epoch,history=history,inputs=tx.shape[-1],median=median,seconds=time.monotonic()-begin)
 torch.save(a,path);dump(path.with_suffix('.json'),{k:v for k,v in a.items() if k not in ['state','prep','train_keys']})
 print('FIT',path.name,'epochs',best_epoch,'seconds',round(a['seconds'],1),flush=True)
 return a

def windows(y,dev,issue):
 inner=issue-pd.DateOffset(months=3);ik=y.index[y.label_end<inner]
 vk=y.index[(y.index>=inner)&(y.index<issue)&(y.label_end<issue)];ot=y.index[y.label_end<issue]
 end=issue+pd.DateOffset(months=3) if issue!=STARTS[-1] else dev.max()+pd.Timedelta(minutes=30)
 return inner,ik,vk,ot,dev[(dev>=issue)&(dev<end)]

def base_window(x,y,dev,ref,issue):
 inner,ik,vk,ot,te=windows(y,dev,issue);art={}
 for phase,date,keys in [('inner',inner,ik),('outer',issue,ot)]:
  for kind in ['pair','rf']:
   a=joblib.load(b.BASE/'models'/f'{date:%Y-%m}__report_{phase}_{kind}__expanding.joblib')
   assert a['train_keys'].equals(keys);art[phase,kind]=a
 pp=b.predict(art['inner','pair'],x.loc[vk]);rr=b.predict(art['inner','rf'],x.loc[vk])
 w=min([0.,.1,.25,.5],key=lambda v:(b.metric((1-v)*pp+v*rr,y.loc[vk,'magnitude_log_bps'])['mae'],v))
 bp=(1-w)*b.predict(art['outer','pair'],x.loc[te])+w*b.predict(art['outer','rf'],x.loc[te])
 np.testing.assert_allclose(bp,ref.loc[te,'report_hybrid'],atol=1e-7,rtol=0)
 return pp,(1-w)*pp+w*rr,art['outer','pair'],w

def register(packs,checks):
 OUT.mkdir(parents=True,exist_ok=True)
 contract=dict(target='absolute 48h endpoint log return in bps',primary='test MAE',starts=list(STARTS),training='all supplied origins with label_end strictly before issuance',validation='preceding quarter; label_end strictly before outer issuance',test_rows=18481,features=packs,configs=CONFIGS,max_epochs=MAX_EPOCHS,patience=PATIENCE,seed=SEED,batch=BATCH,optimizer='AdamW lr .002; MAE/100; gradient clip 1',selection='lowest validation MAE; IC tie break; single models or mean of validation top3',hybrid='LSTM weights .1/.25/.5/.75/1; remainder median regression; no zero-LSTM fallback',outer_epochs='inner best epoch count',sequences='exact timestamp lags; gaps missing',preprocessing='training-only median and IQR/std, clip [-8,8], missing flags',stopping='one frozen search; no test-driven retries',limitation='Previously exposed historical evaluation',source_sha256=sha(__file__))
 if (OUT/'CONTRACT.json').exists():assert json.loads((OUT/'CONTRACT.json').read_text())['source_sha256']==sha(__file__)
 else:
  dump(OUT/'CONTRACT.json',contract);(OUT/'training_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
 framework=ROOT.parents[1]/'ml_framework'
 docs=[HERE/'PROJECT_CONTEXT.md',ROOT.parents[1]/'AGENTS.md']+[framework/p for p in ['AGENTS.md','CONTEXT.md','docs/INDEX.md','docs/methods/WORKFLOW.md','docs/methods/DATA_AND_METRICS.md','docs/findings/RECORDED_RESULTS.md','jobs.json']]
 inputs=[b.old.PACKAGE/'data'/p for p in ['X_train.parquet','y_train.parquet','X_development.parquet','y_development.parquet']]+[ROOT/'Dataset/canonical_returns_5m.parquet',b.BASE/'report_baseline_predictions.csv']
 dump(OUT/'MANIFEST.json',dict(instruction_hashes={str(p):sha(p) for p in docs},input_hashes={str(p):sha(p) for p in inputs},versions={p:importlib.metadata.version(p) for p in ['torch','numpy','pandas','scipy','scikit-learn','joblib']},canonical_checks=checks,device='cpu'))

def run():
 if (OUT/'COMPLETE.json').exists():raise RuntimeError('Completed run exists; use audit')
 x,y,dev,ref,packs,checks=load();register(packs,checks)
 decisions=[];trials=[];predictions=[];validations=[]
 for issue in STARTS:
  inner,ik,vk,ot,te=windows(y,dev,issue);folder=OUT/issue.strftime('%Y%m%d');folder.mkdir(exist_ok=True)
  pp,bv,op,bw=base_window(x,y,dev,ref,issue);cache={};candidates={}
  for name,c in CONFIGS.items():
   a=fit(x,y,ik,inner,name,packs[c['pack']],folder/(name+'_inner.pt'),vk=vk)
   cache[name]=a;candidates[name]=dict(components=[name],prediction=predict(a,x,vk))
  ranked=sorted(candidates,key=lambda n:(b.metric(candidates[n]['prediction'],y.loc[vk,'magnitude_log_bps'])['mae'],n));top=ranked[:3]
  candidates['top3_ensemble']=dict(components=top,prediction=np.mean([candidates[n]['prediction'] for n in top],axis=0))
  choices=[]
  for name,c in candidates.items():
   for weight in [.1,.25,.5,.75,1.]:
    rec=dict(issue=str(issue),name=name,components=c['components'],weight=weight,**b.metric((1-weight)*pp+weight*c['prediction'],y.loc[vk,'magnitude_log_bps']))
    choices.append(rec);trials.append(rec)
  key=lambda a:(a['mae'],-a['rank_ic'],a['name'],a['weight'])
  single=min([a for a in choices if a['weight']==1.],key=key);hybrid=min(choices,key=key)
  needed=sorted(set(single['components']+hybrid['components']))
  s=dict(issue=str(issue),inner=str(inner),standalone=single,hybrid=hybrid,baseline_weight=bw,inner_train_rows=len(ik),train_rows=len(ot),validation_rows=len(vk),test_rows=len(te),max_training_label_end=str(y.loc[ot,'label_end'].max()),max_validation_label_end=str(y.loc[vk,'label_end'].max()),components={n:dict(epochs=cache[n]['epochs'],features=packs[CONFIGS[n]['pack']]) for n in needed})
  dump(folder/'selection.json',s);outer={}
  for name in needed:
   a=fit(x,y,ot,issue,name,packs[CONFIGS[name]['pack']],folder/(name+'_outer.pt'),epochs=cache[name]['epochs']);outer[name]=predict(a,x,te)
  sp=np.mean([outer[n] for n in single['components']],axis=0);w=hybrid['weight']
  hp=w*np.mean([outer[n] for n in hybrid['components']],axis=0)+(1-w)*b.predict(op,x.loc[te])
  p=pd.DataFrame(dict(lstm=sp,lstm_hybrid=hp,baseline=ref.loc[te,'report_hybrid'],issue=str(issue)),index=te)
  p.to_csv(folder/'predictions_unscored.csv',index_label='decision_time');predictions.append(p)
  validations.append(pd.DataFrame(dict(lstm=candidates[single['name']]['prediction'],lstm_hybrid=w*candidates[hybrid['name']]['prediction']+(1-w)*pp,baseline=bv,truth=y.loc[vk,'magnitude_log_bps'],issue=str(issue)),index=vk))
  decisions.append(s);dump(OUT/'SELECTIONS.json',decisions);dump(OUT/'VALIDATION_TRIALS.json',trials)
  print('WINDOW FROZEN',issue.date(),'LSTM',single['name'],'hybrid',hybrid['name'],w,flush=True)
 p=pd.concat(predictions);assert p.index.equals(dev) and p.index.is_unique
 p['truth']=y.loc[dev,'magnitude_log_bps'];v=pd.concat(validations)
 p.to_csv(OUT/'predictions.csv',index_label='decision_time');v.to_csv(OUT/'validation_predictions.csv',index_label='decision_time')
 scores=[dict(split=split,model=name,**b.metric(f[name],f.truth)) for split,f in [('validation',v),('test',p),('last3months',p.loc['2024-11-01':])] for name in ['lstm','lstm_hybrid','baseline']]
 pd.DataFrame(scores).to_csv(OUT/'scores.csv',index=False)
 rows=[dict(issue=issue,model=name,**b.metric(f[name],f.truth)) for issue,f in p.groupby('issue') for name in ['lstm','lstm_hybrid','baseline']]
 pd.DataFrame(rows).to_csv(OUT/'window_scores.csv',index=False)
 dump(OUT/'TRAINING_FINISHED.json',dict(status='TRAINED_AWAITING_AUDIT',scores=scores))
 print(pd.DataFrame(scores).to_string(index=False),flush=True);audit()

def audit():
 x,y,dev,ref,packs,checks=load();p=pd.read_csv(OUT/'predictions.csv',index_col=0,parse_dates=True)
 assert p.index.equals(dev) and p.index.is_unique and len(p)==18481
 np.testing.assert_allclose(p.baseline,ref.report_hybrid,rtol=0,atol=1e-8)
 np.testing.assert_allclose(p.truth,y.loc[dev,'magnitude_log_bps'],rtol=0,atol=1e-8)
 count=0;error=0.
 for issue in STARTS:
  inner,ik,vk,ot,te=windows(y,dev,issue);folder=OUT/issue.strftime('%Y%m%d');s=json.loads((folder/'selection.json').read_text());outer={}
  for path in folder.glob('*.pt'):
   a=torch.load(path,weights_only=False,map_location='cpu');is_inner=path.stem.endswith('_inner');keys,cutoff=(ik,inner) if is_inner else (ot,issue)
   assert a['train_keys'].equals(keys) and a['max_label_end']<cutoff and y.loc[keys,'label_end'].max()<cutoff
   expected=prep(x,keys,packs[CONFIGS[a['name']]['pack']])
   for field in ['median','scale']:pd.testing.assert_series_equal(a['prep'][field],expected[field])
   sample=te[::max(1,len(te)//16)]
   np.testing.assert_array_equal(seq(x,sample,a['prep'],a['config']).numpy(),seq(x.loc[:sample.max()],sample,a['prep'],a['config']).numpy())
   if not is_inner:
    assert a['epochs']==s['components'][a['name']]['epochs'];outer[a['name']]=predict(a,x,te)
   count+=1
  assert y.loc[vk,'label_end'].max()<issue
  _,_,op,_=base_window(x,y,dev,ref,issue);sp=np.mean([outer[n] for n in s['standalone']['components']],axis=0);w=s['hybrid']['weight']
  hp=w*np.mean([outer[n] for n in s['hybrid']['components']],axis=0)+(1-w)*b.predict(op,x.loc[te])
  for name,pred in [('lstm',sp),('lstm_hybrid',hp)]:
   error=max(error,float(np.max(abs(pred-p.loc[te,name]))))
   if pred.dtype==np.float32:
    np.testing.assert_array_equal(pred,p.loc[te,name].to_numpy(dtype=np.float32))
   else:
    np.testing.assert_allclose(pred,p.loc[te,name],rtol=0,atol=1e-10)
 result=dict(status='PASS',rows=len(p),new_fits=count,baseline_reproduction=True,training_only_preprocessing=True,purged_training_validation=True,causal_sequences=True,canonical_checks=checks,max_replay_error=error)
 dump(OUT/'AUDIT.json',result);scores=pd.read_csv(OUT/'scores.csv');test=scores[scores.split=='test'].set_index('model')
 dump(OUT/'COMPLETE.json',dict(status='EXECUTED_AND_AUDITED',audit=result,test_scores=test.to_dict('index'),standalone_beats_baseline=bool(test.loc['lstm','mae']<test.loc['baseline','mae']),hybrid_beats_baseline=bool(test.loc['lstm_hybrid','mae']<test.loc['baseline','mae']),training_source_sha256=json.loads((OUT/'CONTRACT.json').read_text())['source_sha256'],audit_source_sha256=sha(__file__),limitation='Previously exposed historical evaluation; not fresh holdout'))
 print('AUDIT PASS',count,'fits; replay error',error,flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','audit']);args=parser.parse_args()
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 with threadpool_limits(limits=2):run() if args.action=='run' else audit()
