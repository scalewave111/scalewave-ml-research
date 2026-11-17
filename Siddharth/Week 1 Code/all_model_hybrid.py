"""Monthly hybrid selection from earlier, genuinely out-of-sample 48h forecasts.
No final-test weight fitting. Historical period is retrospective, not untouched."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import linprog
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'Experiments/all_model_hybrid_v1'
EXPOSURES=[.025,.05,.1,.2,.35,.5]
def dump(p,v):p.write_text(json.dumps(v,indent=2,default=str,allow_nan=False),encoding='utf-8')
def metric(p,y):
 a=np.asarray(p);b=np.asarray(y)
 return dict(mae=float(abs(a-b).mean()),rank_ic=float(spearmanr(np.round(a,10),b).statistic) if np.ptp(a)>0 else 0.,pearson_ic=float(np.corrcoef(a,b)[0,1]) if np.ptp(a)>0 else None,rows=len(a))
def mix(frame,weights):
 return sum(float(w)*frame[name] for name,w in weights.items())
def family_mix(names,registry):
 groups={}
 for name in names:groups.setdefault(registry[name]['family'],[]).append(name)
 return groups
def expand_family(groups,weights):
 return {name:float(weights[group])/len(names) for group,names in groups.items() for name in names}
def fit_mae_weights(frame,y,penalty):
 n,k=frame.shape
 if k==1:return np.ones(1),dict(status=0,message='One family; no optimization',objective=float(abs(frame.iloc[:,0]-y).mean()))
 A=sparse.csr_matrix(frame.to_numpy());I=sparse.eye(n,format='csr');Z=sparse.csr_matrix((n,k));J=sparse.eye(k,format='csr');ZN=sparse.csr_matrix((k,n))
 aub=sparse.vstack([sparse.hstack([A,-I,Z]),sparse.hstack([-A,-I,Z]),sparse.hstack([J,ZN,-J]),sparse.hstack([-J,ZN,-J])],format='csr')
 uniform=np.full(k,1/k);bub=np.r_[np.asarray(y),-np.asarray(y),uniform,-uniform]
 objective=np.r_[np.zeros(k),np.ones(n)/n,np.full(k,penalty)]
 ae=sparse.csr_matrix(np.r_[np.ones(k),np.zeros(n+k)][None,:])
 result=linprog(objective,A_ub=aub,b_ub=bub,A_eq=ae,b_eq=[1.],bounds=[(.02,1.)]*k+[(0,None)]*(n+k),method='highs',options={'time_limit':30})
 if not result.success:return None,dict(status=int(result.status),message=result.message)
 w=result.x[:k];w=np.maximum(w,0);w/=w.sum()
 return w,dict(status=0,message=result.message,objective=float(result.fun))
def gate(p,baseline,y,index):
 a=metric(p,y);b=metric(baseline,y)
 gains=[]
 for month in sorted(set(index.strftime('%Y-%m'))):
  ix=index.strftime('%Y-%m')==month
  gains.append(float(np.mean(abs(np.asarray(baseline)[ix]-np.asarray(y)[ix])-abs(np.asarray(p)[ix]-np.asarray(y)[ix]))))
 return dict(mae_gain=b['mae']-a['mae'],ic_gain=a['rank_ic']-b['rank_ic'],monthly_mae_wins=int(sum(z>.01 for z in gains)),passed=bool(b['mae']-a['mae']>.01 and a['rank_ic']-b['rank_ic']>.0001 and sum(z>.01 for z in gains)>=2))
def decision(matrix,targets,registry,issue):
 cutoff=issue-pd.DateOffset(months=2);start=issue-pd.DateOffset(months=5)
 train=matrix.index[(matrix.index>=start)&(matrix.index<cutoff)&(targets.label_end<cutoff)]
 val=matrix.index[(matrix.index>=cutoff)&(matrix.index<issue)&(targets.label_end<issue)]
 end=issue+pd.DateOffset(months=1);test=matrix.index[(matrix.index>=issue)&(matrix.index<end)]
 info=dict(issue=str(issue),weight_train_start=str(start),validation_start=str(cutoff),validation_rows=len(val),max_validation_label_end=str(targets.loc[val,'label_end'].max()) if len(val) else None,eligible=[],excluded=[],duplicate_map=[],weight_training_rows=len(train))
 if len(val)<40*48:
  info.update(weights={'baseline':1.},method='baseline_warmup',validation_gate=None)
  return info,[],[],train,val,test
 # Eligibility/dedup uses only historical forecast values; current-month values never rank candidates.
 names=[];seen={'baseline':matrix.loc[val,'baseline'].to_numpy()}
 for name in matrix.columns:
  if name=='baseline':continue
  values=matrix.loc[val,name]
  if values.notna().mean()<.95 or not matrix.loc[test[0],name]==matrix.loc[test[0],name]:
   info['excluded'].append(dict(name=name,reason='insufficient_prior_forecast_coverage_or_unavailable_at_issue'));continue
  # Use a fixed baseline replacement only for sparse historical forecast gaps, recorded.
  values=values.fillna(matrix.loc[val,'baseline']).to_numpy()
  duplicate=next((other for other,b in seen.items() if np.allclose(values,b,rtol=0,atol=1e-8)),None)
  if duplicate:info['duplicate_map'].append(dict(name=name,same_prior_forecast_as=duplicate));continue
  seen[name]=values;names.append(name)
 info['eligible']=names
 if not names:
  info.update(weights={'baseline':1.},method='baseline_no_candidates',validation_gate=None)
  return info,[],[],train,val,test
 # Current availability is an integrity condition, not a performance-selection criterion.
 assert matrix.loc[test,names].notna().all().all(),'Unexpected missing future component forecasts'
 history=matrix.loc[val,['baseline']+names].copy()
 for name in names:history[name]=history[name].fillna(history.baseline)
 groups=family_mix(names,registry)
 candidates=[dict(method='all_models_equal',weights={name:1/len(names) for name in names}),
             dict(method='family_balanced',weights=expand_family(groups,{g:1/len(groups) for g in groups}))]
 for group,cols in groups.items():candidates.append(dict(method='family:'+group,weights={name:1/len(cols) for name in cols}))
 for name in names:candidates.append(dict(method='individual:'+name,weights={name:1.}))
 solverlog=[]
 # Learned weights only see components with adequate coverage in the earlier fitting window.
 fitnames=[name for name in names if len(train)>=25*48 and matrix.loc[train,name].notna().mean()>=.95]
 if fitnames:
  ft=matrix.loc[train,['baseline']+fitnames].copy()
  for name in fitnames:ft[name]=ft[name].fillna(ft.baseline)
  fg=family_mix(fitnames,registry);A=pd.DataFrame({g:ft[cols].mean(axis=1) for g,cols in fg.items()},index=train)
  for penalty in [0.,.5,2.]:
   w,record=fit_mae_weights(A,targets.loc[train,'truth'],penalty)
   record.update(penalty=penalty,train_rows=len(train),train_start=str(train.min()),train_end=str(train.max()),max_label_end=str(targets.loc[train,'label_end'].max()),groups=fg)
   if w is not None:
    record['family_weights']={g:float(v) for g,v in zip(A.columns,w)}
    candidates.append(dict(method='trained_family_L1_'+str(penalty),weights=expand_family(fg,record['family_weights'])))
   solverlog.append(record)
 b=history.baseline;y=targets.loc[val,'truth'];trials=[]
 for candidate in candidates:
  for exposure in EXPOSURES:
   weights={'baseline':1-exposure,**{name:exposure*w for name,w in candidate['weights'].items()}}
   pred=mix(history,weights);g=gate(pred,b,y,val)
   trials.append(dict(method=candidate['method'],exposure=exposure,weights=weights,**metric(pred,y),**g))
 passing=[a for a in trials if a['passed']]
 if passing:
  best=min(passing,key=lambda a:(a['mae'],-a['rank_ic'],a['exposure'],a['method']))
  info.update(weights=best['weights'],method=best['method'],exposure=best['exposure'],validation_gate={k:best[k] for k in ['mae_gain','ic_gain','monthly_mae_wins','passed']})
 else:info.update(weights={'baseline':1.},method='baseline_no_validated_improvement',exposure=0.,validation_gate=None)
 return info,trials,solverlog,train,val,test

def run(matrix,targets,registry):
 predictions=[];validations=[];training=[];decisions=[];alltrials=[];allsolvers=[]
 for issue in pd.date_range('2024-01-01','2025-01-01',freq='MS',tz='UTC'):
  info,trials,solvers,tr,va,te=decision(matrix,targets,registry,issue)
  weights=info['weights'];assert min(weights.values())>=0 and abs(sum(weights.values())-1)<1e-10
  pred=mix(matrix.loc[te],weights)
  assert np.isfinite(pred).all()
  predictions.append(pd.DataFrame({'truth':targets.loc[te,'truth'],'prediction':pred,'baseline':matrix.loc[te,'baseline'],'issue':str(issue)}))
  if len(va):
   a=matrix.loc[va,list(weights)].copy()
   for name in a:a[name]=a[name].fillna(matrix.loc[va,'baseline'])
   validations.append(pd.DataFrame({'truth':targets.loc[va,'truth'],'prediction':mix(a,weights),'baseline':matrix.loc[va,'baseline'],'issue':str(issue)}))
  # Report actual complete calibration rows only; no invented training history.
  if len(tr):
   good=matrix.loc[tr,list(weights)].notna().all(axis=1);tk=tr[good]
   if len(tk):training.append(pd.DataFrame({'truth':targets.loc[tk,'truth'],'prediction':mix(matrix.loc[tk],weights),'baseline':matrix.loc[tk,'baseline'],'issue':str(issue)}))
  decisions.append(info);alltrials.extend(dict(issue=str(issue),**a) for a in trials);allsolvers.extend(dict(issue=str(issue),**a) for a in solvers)
  print(issue.date(),info['method'],info.get('exposure',0),'eligible',len(info['eligible']),flush=True)
 return pd.concat(predictions),pd.concat(training),pd.concat(validations),decisions,alltrials,allsolvers
def main():
 matrix=pd.read_csv(OUT/'component_predictions.csv',index_col=0,parse_dates=True)
 targets=pd.read_csv(OUT/'targets.csv',index_col=0,parse_dates=True);targets.label_end=pd.to_datetime(targets.label_end,utc=True)
 registry={r['name']:r for r in json.loads((OUT/'REGISTRY.json').read_text())}
 assert matrix.index.equals(targets.index) and matrix.index.is_unique and len(matrix)==18481
 p,t,v,choices,trials,solvers=run(matrix,targets,registry)
 for name,f in [('predictions',p),('weight_training_predictions',t),('validation_predictions',v)]:f.to_csv(OUT/(name+'.csv'),index_label='decision_time')
 scores=[]
 for split,f,col in [('weight_calibration',t,'prediction'),('validation',v,'prediction'),('test',p,'prediction'),('test_baseline',p,'baseline')]:
  scores.append(dict(split=split,unique_timestamps=f.index.nunique(),**metric(f[col],f.truth)))
 late=p.loc[p.index>=pd.Timestamp('2024-11-01',tz='UTC')]
 for name in ['prediction','baseline']:scores.append(dict(split='last3months_'+name,unique_timestamps=len(late),**metric(late[name],late.truth)))
 pd.DataFrame(scores).to_csv(OUT/'scores.csv',index=False)
 dump(OUT/'SELECTIONS.json',choices);dump(OUT/'VALIDATION_TRIALS.json',trials);dump(OUT/'WEIGHT_FITS.json',solvers)
 dump(OUT/'COMPLETE.json',dict(status='EXECUTED',scores=scores,test_rows=len(p),new_base_model_training='seePREPARATION.json',combiner_weight_fits=len(solvers),validation_comparisons=len(trials),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),component_matrix_sha256=hashlib.sha256((OUT/'component_predictions.csv').read_bytes()).hexdigest()))
 print(json.dumps(scores),flush=True)
if __name__=='__main__':main()
