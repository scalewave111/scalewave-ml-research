"""Read-only saved forecast audit and combined MAE report; never fits models."""
from pathlib import Path
import json,re,hashlib,ast,math
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'Code/experiments'
SUP=ROOT/'Report/combined_magnitude_support_2026-11-14'
SUP.mkdir(parents=True,exist_ok=True)
HORIZONS=['15m','1h','4h','8h','16h','48h','72h','120h','200h','300h']
audit={}
def js(p):
 audit[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 return json.loads(p.read_text(encoding='utf-8-sig'))
def audit_predictions():
 records=[];frames={}; baselines={}
 for h in HORIZONS:
  candidates=[]
  for directory in EXP.iterdir():
   if '_'+h+'_' not in directory.name:continue
   if h in ['15m','1h','4h','8h','16h','48h'] and 'excursion' not in directory.name:continue
   for c in directory.rglob('COMPLETE.json'):
    d=c.parent;p=d/'predictions.csv'
    if not p.exists() or not (d/'CONTRACT.json').exists():continue
    done=js(c)
    if done.get('status')!='COMPLETE':continue
    expected=done.get('predictions_sha256',done.get('prediction_sha256'))
    if not expected:continue
    digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest==expected,p
    audit[str(p.relative_to(ROOT))]=digest
    f=pd.read_csv(p);ycol='actual' if 'actual' in f else 'actual_log_bps'
    if ycol not in f or 'target' not in f:continue
    if not set(f.target).issubset({'up_magnitude_log_bps','down_magnitude_log_bps'}):continue
    candidates.append((d,f,ycol))
  for target in ['up_magnitude_log_bps','down_magnitude_log_bps']:
   n=max(len(f[f.target.eq(target)]) for d,f,y in candidates)
   ref=None
   for d,f,ycol in candidates:
    g=f[f.target.eq(target)].sort_values('decision_time').reset_index(drop=True)
    if len(g)!=n:continue
    assert not g.decision_time.duplicated().any()
    if ref is None:ref=g[['decision_time',ycol]].copy();ref.columns=['decision_time','actual']
    assert list(g.decision_time)==list(ref.decision_time)
    np.testing.assert_allclose(g[ycol],ref.actual,rtol=0,atol=1e-8)
    key=str(d.relative_to(EXP));frames[(h,key,target)]=g
    for col in g:
     if col in ['actual','actual_log_bps','effective_weight','rolling_count'] or col.endswith('_count') or not pd.api.types.is_numeric_dtype(g[col]):continue
     if not np.isfinite(g[col]).all():continue
     e=g[col]-g[ycol]
     record=dict(horizon=h,target=target,run=key,forecast=col,rows=n,MAE=float(abs(e).mean()),MedianAE=float(abs(e).median()),RMSE=float(np.sqrt(np.mean(e**2))),IC=float(g[col].corr(g[ycol],method='spearman')),Bias=float(e.mean()))
     records.append(record)
     if col in ['training_mean','quarterly_mean','prior_mean','mean','training_median','quarterly_median','prior_median','median']:
      kind='mean' if col.endswith('mean') else 'median'
      if (h,target,kind) not in baselines:baselines[(h,target,kind)]=record
   assert ref is not None
 rankings=pd.DataFrame(records).sort_values(['horizon','target','MAE','run','forecast'])
 rankings['MAE_rank']=rankings.groupby(['horizon','target']).MAE.rank(method='min')
 rankings.to_csv(SUP/'all_saved_full_coverage_forecasts.csv',index=False)
 winners=[]
 for h in HORIZONS:
  for target in ['up_magnitude_log_bps','down_magnitude_log_bps']:
   w=rankings[rankings.horizon.eq(h)&rankings.target.eq(target)].iloc[0].to_dict()
   w['baselines']={k:baselines[(h,target,k)] for k in ['mean','median']}
   winners.append(w)
 (SUP/'winners_metrics.json').write_text(json.dumps(winners,indent=2))
 (SUP/'prediction_audit.json').write_text(json.dumps(audit,indent=2))
 return winners
def model_metadata(winners):
 import joblib
 def load(p,expected=None):
  digest=hashlib.sha256(p.read_bytes()).hexdigest()
  if expected:assert digest==expected,p
  audit[str(p.relative_to(ROOT))]=digest
  return joblib.load(p)
 def estimator(m):
  while isinstance(m,dict):m=m['model']
  return dict(architecture=type(m).__name__,hyperparameters=m.get_params())
 for w in winners:
  h,t,col=w['horizon'],w['target'],w['forecast'];d=EXP/w['run'];side='UP' if t.startswith('up') else 'DOWN';quarters=[]
  w['status']='Validation-selected policy' if col in ['selected','mae_selected','mlp_selected'] else 'Best observed diagnostic (chosen retrospectively by MAE)'
  if h=='72h' and side=='DOWN':w['status']='Candidate equals validation-selected policy'
  if h=='200h':w['status']='Diagnostic candidate; DOWN also equals later selected forecast' if side=='DOWN' else 'Diagnostic candidate'
  for month in ['2024-01','2024-04','2024-07','2024-10']:
   q=dict(issue=month,features=[]);folder=d/(month+'_'+t)
   if h in ['4h','8h','16h'] or (h=='72h' and side=='UP'):
    card=js(folder/'selection.json');variant=card['selected_variant'] if col=='selected' else col;s=card['variants'][variant]
    q.update(features=s['features'],specification={'variant':variant},**estimator(load(folder/s['artifact'],s['artifact_sha256'])))
    q['formula']='Training-median imputation + missing flags; training mean/std scaling. yhat=max(0,b0+eta*sum tree leaf updates).'
   elif h=='15m':
    card=js(folder/'selection.json');candidate=card['candidate'] if col=='selected' else col;s=card['specs'][candidate];weight=card['weight'] if col=='selected' else 1
    q.update(features=list(s['features']),specification={'candidate':candidate,'weight':weight},**estimator(load(folder/s['artifact'],js(folder/'COMPLETE.json')['hashes'][s['artifact']])))
    if weight<1:q['features']=list(dict.fromkeys(q['features']+card['specs']['current_features__unchanged']['features']))
    q['formula']=f'yhat={weight:g}*candidate + {1-weight:g}*previous model. Both use median imputation, missing flags and mean/std scaling; each tree prediction is max(0,b0+eta*sum leaf updates).'
    if weight<1:q['reference_parameters']=estimator(load(folder/card['current_artifact']))['hyperparameters']
   elif h=='1h':
    card=js(folder/'selection.json');choice=card['choices'][col];s=card['specs'][choice['candidate']];p=folder/(choice['candidate']+'.joblib')
    q.update(features=list(s['features']),specification={'candidate':choice['candidate'],'weight':choice['weight']},**estimator(load(p,js(folder/(choice['candidate']+'_HASH.json'))['sha256'])))
    if choice['weight']<1:
     ref=pd.read_csv(ROOT/'Report/one_hour_tuned_report_sources/exact_tuned_features.csv');q['features']=list(dict.fromkeys(q['features']+ref[(ref.issue==month)&(ref.target==t)&(ref.policy==col)].feature.tolist()))
    q['formula']=f"yhat={choice['weight']:g}*candidate + {1-choice['weight']:g}*previous model. New candidate uses raw inputs and native missing-value splits; max(0,boosted tree sum). Reference uses imputation + flags + mean/std scaling, HGB 80 iterations, eta .05, depth3, leaves7, minleaf128, L2=1."
   elif h=='48h':
    s=next(s for s in js(d/'SELECTIONS.json') if s['target']==t and s['issue'].startswith(month) and s['policy']==col)
    m=load(d/s['artifact'],s['sha256']);q.update(features=m['columns'],specification={'policy':col},**estimator(m))
    q['formula']='235 engineered values -> native missing-value histogram bins -> additive absolute-error boosted trees. yhat=max(0,b0+eta*sum leaf updates). No input scaling or missing flags.'
   elif h=='72h':
    folder=d/t/month;m=load(folder/'model.joblib',js(folder/'MODEL_HASH.json')['sha256']);b=m[col];q.update(features=b['features'],architecture='GRU16',specification=m['plan'][col],hyperparameters=dict(**b['config'],epochs=24,batch=512,training_cap=4096,seed=48,gradient_clip=1))
    q['formula']='4 snapshots (18h,12h,6h,0h), 30 raw values + 30 missing flags per step -> GRU(60,16) -> Linear(16,1). yhat=m+w*0.75*max(m,1)*tanh(output); m=training median. Input median imputation, mean/std scaling, clip8. Insufficient coverage falls back to m.'
   elif h=='120h' and side=='UP':
    folder=d/t/month;m=load(folder/'model.joblib',js(folder/'MODEL_HASH.json')['sha256']);s=m['plan'][col];b=m[col];u=s['update_weight'];v=s['mean_weight']
    q.update(features=list(dict.fromkeys(m['anchor_features']+b['features'])),architecture='Anchor HGB + paired L1/L2 HGB',specification={k:s[k] for k in ['config','pack','mean_weight','update_weight','eligible']},hyperparameters={'anchor':estimator(m['anchor'])['hyperparameters'],'L1':b['median'].get_params(),'L2':b['mean'].get_params()})
    q['formula']=f'yhat={1-u:g}*A + {u*(1-v):g}*L + {u*v:g}*M. A=previous HGB; L/M=nonnegative L1/L2 HGB predictions. Imputation + flags + mean/std scaling. Correction uses capped training rows; <80% available inputs falls back to A.'
   elif h=='120h':
    m=load(folder/'model.joblib',js(folder/'MODEL_HASH.json')['sha256']);b=m['model'];s=b['spec'];q.update(features=m['features'],architecture='Two-seed MLP ensemble: 8 -> 8 ReLU -> 1 Softplus',specification={k:s[k] for k in ['pack','config','mode','weight']},hyperparameters=dict(epochs=40,optimizer='AdamW',lr=.001,weight_decay=.1,batch=256,clip=1,seeds=[48,148]))
    if s['mode']=='volatility':q['features']=q['features']+['LONG__vol_per_sqrt_hour__120h']
    q['formula']='Median imputation + missing flags -> mean/std scaling -> clip8. yhat=w*a(t)*u*mean_s Softplus(W2_s ReLU(W1_s z+b1_s)+b2_s)+(1-w)*m. u=max(training median(y/a),.001). Volatility mode: a0=LONG__vol_per_sqrt_hour__120h*sqrt(120); finite scale floored at max(training P10 positive scale,1), nonfinite uses training positive-scale median. Raw mode a=1. Extra scale input is included in the matrix.'
   elif h=='200h':
    sup=ROOT/'Report/200h_report_support_2026-11-14';roots=pd.read_csv(sup/'quarterly_model_roots.csv');nodes={x['id']:x for x in js(sup/'model_components.json')};label='F' if side=='UP' else 'E';r=roots[(roots.run==label)&(roots.target==side)&(roots.issue==month)&(roots.forecast==col)].iloc[0];ids=ast.literal_eval(r.nodes);root=nodes[r.root]
    q.update(features=root['features'],architecture='Nested transformed/dual ensemble' if side=='UP' else 'GRU or median anchor + paired HGB',specification={'root':r.root},components=[nodes[i] for i in ids],formula=root['formula'],hyperparameters='Exact component settings below')
    folder=d/t/month;load(folder/'model.joblib',js(folder/'MODEL_HASH.json')['sha256'])
   else:
    details=js(ROOT/'Report/300h_report_support_2026-11-14/model_details.json');s=next(s for s in details if s['run']=='G' and s['target']==side and s['issue']==month and s['forecast']==col);q.update({k:s[k] for k in ['features','architecture','specification','hyperparameters']})
    q['formula']='HGB: yhat=(1-w)*a+w*max(0,tree prediction); a=training mean or median from specification. GRU16: p=max(0,m+.75*max(m,1)*tanh(Linear(last hidden))); yhat=(1-w)*a+w*p; m=training median; a=(1-anchor_weight)*m+anchor_weight*training mean; 4 snapshots 90/60/30/0 hours, train-only imputation/flags/standardization, clip8; low coverage uses anchor. Median constant uses no predictors.'
    folder=d/t/month;load(folder/'model.joblib',js(folder/'MODEL_HASH.json')['sha256'])
   q['features']=list(dict.fromkeys(q['features']));quarters.append(q)
  w['quarters']=quarters;w['features']=sorted(set(c for q in quarters for c in q['features']))
 (SUP/'winner_models_and_features.json').write_text(json.dumps(winners,indent=2,default=str))
 (SUP/'source_hashes.json').write_text(json.dumps(audit,indent=2))
 return winners

if __name__=='__main__':
 winners=json.loads((SUP/'winners_metrics.json').read_text()) if (SUP/'winners_metrics.json').exists() else audit_predictions()
 winners=model_metadata(winners)
 for w in winners:print(w['horizon'],w['target'][:4],round(w['MAE'],6),len(w['features']))
