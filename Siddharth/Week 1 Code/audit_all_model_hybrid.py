"""Audit source forecasts, completed base artifacts, and causal hybrid selection."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,joblib
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
import all_model_hybrid as m
import tree_improvement_campaign as c
from audit_tree_improvement_campaign import validate_art
def rank_ic_matrix(P,y):
 a=rankdata(np.round(P,10),axis=0);b=rankdata(y);a-=a.mean(axis=0);b-=b.mean()
 denom=np.sqrt((a*a).sum(axis=0)*(b*b).sum())
 return np.divide((a*b[:,None]).sum(axis=0),denom,out=np.zeros(P.shape[1]),where=denom>0)
def main():
 out=m.OUT;complete=json.loads((out/'COMPLETE.json').read_text())
 assert complete['source_sha256']==hashlib.sha256((m.ROOT/'Code/all_model_hybrid.py').read_bytes()).hexdigest()
 assert complete['component_matrix_sha256']==hashlib.sha256((out/'component_predictions.csv').read_bytes()).hexdigest()
 X=pd.read_csv(out/'component_predictions.csv',index_col=0,parse_dates=True);T=pd.read_csv(out/'targets.csv',index_col=0,parse_dates=True);T.label_end=pd.to_datetime(T.label_end,utc=True)
 registry=json.loads((out/'REGISTRY.json').read_text());rmap={r['name']:r for r in registry}
 raw,y,dev,report=c.old.old.load();assert X.index.equals(dev) and X.index.is_unique
 np.testing.assert_allclose(T.truth,y.loc[dev,'magnitude_log_bps'],rtol=0,atol=1e-8)
 checked_sources=0;frames={}
 for rec in registry:
  if rec['source']=='components ledger':continue
  path=Path(rec['source'])
  assert hashlib.sha256(path.read_bytes()).hexdigest()==rec['source_sha256']
  if str(path) not in frames:frames[str(path)]=pd.read_csv(path,index_col=0,parse_dates=True)
  expected=frames[str(path)][rec['column']].reindex(dev)
  np.testing.assert_allclose(expected,X[rec['name']],rtol=0,atol=1e-8,equal_nan=True);checked_sources+=1
 reference=pd.read_csv(m.ROOT/'Experiments/nested_boost_48h_v1/report_baseline_predictions.csv',index_col=0,parse_dates=True)
 np.testing.assert_allclose(X.baseline,reference.report_hybrid,rtol=0,atol=1e-8)
 returns=c.crf.load_returns(m.ROOT/'Dataset/canonical_returns_5m.parquet');parents=list(dict.fromkeys(c.PAIR+[n for n in report if c.old.family(n)!='other']))
 frame=pd.concat([raw,c.engineering(raw,parents),c.state_frame(returns,raw.index)],axis=1).replace([np.inf,-np.inf],np.nan).astype('float32')
 canonical_y=c.crf.make_labels(returns,raw.index,48)
 np.testing.assert_allclose(canonical_y.loc[y.index,'magnitude_log_bps'],y.magnitude_log_bps,rtol=0,atol=1e-8)
 ledger=json.loads((out/'COMPONENT_FITS.json').read_text());checked=set();full_configs=set();new_artifacts=0
 for rec in ledger:
  art=joblib.load(rec['artifact']);issue=pd.Timestamp(rec['issue'])
  assert pd.Timestamp(art['max_label_end'])<issue and art['columns']==rec['columns']
  if rec['artifact'] not in checked:
   full=rec['config'] not in full_configs and rec['status']!='REUSED_ORIGINAL'
   validate_art(art,frame,canonical_y if rec['status']=='REUSED_ORIGINAL' else y,full=full);checked.add(rec['artifact'])
   if full:full_configs.add(rec['config'])
   new_artifacts+=rec['status']!='REUSED_ORIGINAL'
  idx=dev[(dev>=issue)&(dev<issue+pd.DateOffset(months=1))]
  np.testing.assert_allclose(c.predict(art,frame.loc[idx]),X.loc[idx,rec['name']],rtol=1e-10,atol=1e-8)
 choices=json.loads((out/'SELECTIONS.json').read_text());trials=json.loads((out/'VALIDATION_TRIALS.json').read_text());solvers=json.loads((out/'WEIGHT_FITS.json').read_text())
 p=pd.read_csv(out/'predictions.csv',index_col=0,parse_dates=True);assert p.index.equals(dev)
 validation_checks=0
 for dec in choices:
  issue=pd.Timestamp(dec['issue']);cut=issue-pd.DateOffset(months=2)
  va=X.index[(X.index>=cut)&(X.index<issue)&(T.label_end<issue)]
  tr=X.index[(X.index>=issue-pd.DateOffset(months=5))&(X.index<cut)&(T.label_end<cut)]
  te=X.index[(X.index>=issue)&(X.index<issue+pd.DateOffset(months=1))]
  assert len(va)==dec['validation_rows']
  if len(va):assert T.loc[va,'label_end'].max()<issue
  if len(tr):assert T.loc[tr,'label_end'].max()<cut
  weights=dec['weights'];assert all(v>=0 for v in weights.values()) and abs(sum(weights.values())-1)<1e-10 and weights.get('baseline',0)>=.5-1e-10
  np.testing.assert_allclose(m.mix(X.loc[te],weights),p.loc[te,'prediction'],rtol=1e-10,atol=1e-8)
  tests=[a for a in trials if a['issue']==dec['issue']]
  if tests:
   columns=list(X);C=np.zeros((len(columns),len(tests)))
   for j,test in enumerate(tests):
    for name,w in test['weights'].items():C[columns.index(name),j]=w
   A=X.loc[va].to_numpy();b=X.loc[va,'baseline'].to_numpy();target=T.loc[va,'truth'].to_numpy()
   A=np.where(np.isfinite(A),A,b[:,None]);P=A@C
   maes=abs(P-target[:,None]).mean(axis=0);ics=rank_ic_matrix(P,target);base=m.metric(b,target)
   wins=np.zeros(len(tests),dtype=int)
   for month in sorted(set(va.strftime('%Y-%m'))):
    ix=va.strftime('%Y-%m')==month
    gains=abs(b[ix]-target[ix]).mean()-abs(P[ix]-target[ix,None]).mean(axis=0);wins+=gains>.01
   for j,test in enumerate(tests):
    np.testing.assert_allclose([maes[j],ics[j]],[test['mae'],test['rank_ic']],rtol=1e-8,atol=1e-8)
    passed=base['mae']-maes[j]>.01 and ics[j]-base['rank_ic']>.0001 and wins[j]>=2
    assert bool(passed)==test['passed'] and wins[j]==test['monthly_mae_wins']
   passing=[t for t in tests if t['passed']]
   if passing:
    best=min(passing,key=lambda a:(a['mae'],-a['rank_ic'],a['exposure'],a['method']))
    assert dec['weights']==best['weights'] and dec['method']==best['method']
   else:assert weights=={'baseline':1.}
   validation_checks+=len(tests)
  else:assert weights=={'baseline':1.}
  for fit in [a for a in solvers if a['issue']==dec['issue']]:
   assert pd.Timestamp(fit['max_label_end'])<cut
   if fit['status']!=0:continue
   fw=fit['family_weights'];assert abs(sum(fw.values())-1)<1e-8 and min(fw.values())>=.02-1e-8
   F={}
   for group,names in fit['groups'].items():
    z=X.loc[tr,names].copy()
    for name in names:z[name]=z[name].fillna(X.loc[tr,'baseline'])
    F[group]=z.mean(axis=1)
   pred=sum(fw[group]*series for group,series in F.items())
   objective=float(abs(pred-T.loc[tr,'truth']).mean()+fit['penalty']*sum(abs(w-1/len(fw)) for w in fw.values()))
   np.testing.assert_allclose(objective,fit['objective'],rtol=1e-7,atol=1e-7)
  print('Hybrid audit',issue.date(),'PASS',flush=True)
 # Perturb all not-yet-known labels and all future forecasts at an early nontrivial issue.
 issue=pd.Timestamp('2024-03-01',tz='UTC');changed_y=T.copy();changed_y.loc[changed_y.label_end>=issue,'truth']+=1e7
 changed_x=X.copy();changed_x.loc[changed_x.index>=issue]+=1e7
 replay,*_=m.decision(changed_x,changed_y,rmap,issue)
 expected=next(d for d in choices if d['issue']==str(issue))
 assert replay['weights']==expected['weights'] and replay['method']==expected['method']
 for score in complete['scores']:
  split=score['split'];f=pd.read_csv(out/('weight_training_predictions.csv' if split=='weight_calibration' else 'validation_predictions.csv' if split=='validation' else 'predictions.csv'),index_col=0,parse_dates=True)
  if split.startswith('last3months'):f=f.loc[f.index>=pd.Timestamp('2024-11-01',tz='UTC')]
  col='baseline' if split in ['test_baseline','last3months_baseline'] else 'prediction'
  actual=m.metric(f[col],f.truth)
  for k,v in actual.items():
   if v is not None:np.testing.assert_allclose(v,score[k],rtol=1e-8,atol=1e-8)
 m.dump(out/'AUDIT.json',dict(status='PASS',source_forecast_columns_checked=checked_sources,component_artifacts_checked=len(checked),new_component_artifacts=new_artifacts,full_leaf_checks_configurations=sorted(full_configs),validation_candidates_recomputed=validation_checks,all_monthly_weights_replayed=True,future_target_and_forecast_perturbation=True,weight_training_label_purges=True,unique_test_rows=len(p),no_base_model_refits=True,legacy_limitation='Some historical recipe and blend choices were development-selected; this remains retrospective even though new combiner weights use earlier matured labels only.'))
 print('Hybrid audit PASS',flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
